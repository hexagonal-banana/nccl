/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// C++ port of convert_to_chrome_trace.py. Reads an inspector JSON log
// file (one record per line) and writes a Chrome Trace Event Format
// JSON file. This runs in the cold path (finalize), so STL usage is
// unrestricted here.

#include "inspector_trace.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <sys/stat.h>

namespace {

// ---------------------------------------------------------------------------
// Minimal JSON value + recursive-descent parser
// ---------------------------------------------------------------------------
struct JsonValue {
  enum Type { Null, Bool, Num, Str, Arr, Obj } type;
  bool b;
  double num;
  long long inum;
  bool isInt;
  std::string str;
  std::vector<JsonValue> arr;
  std::vector<std::pair<std::string, JsonValue>> obj;

  JsonValue() : type(Null), b(false), num(0), inum(0), isInt(false) {}

  const JsonValue* find(const char* key) const {
    if (type != Obj) return nullptr;
    for (size_t i = 0; i < obj.size(); ++i) {
      if (obj[i].first == key) return &obj[i].second;
    }
    return nullptr;
  }

  long long asInt(long long def = 0) const {
    if (type == Num) return isInt ? inum : (long long)num;
    return def;
  }
  double asDouble(double def = 0.0) const {
    if (type == Num) return isInt ? (double)inum : num;
    return def;
  }
  const char* asStr(const char* def = "") const {
    return type == Str ? str.c_str() : def;
  }
};

static void skipWs(const char*& p) {
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
}

static bool parseValue(const char*& p, JsonValue& out);

static bool parseString(const char*& p, std::string& out) {
  if (*p != '"') return false;
  ++p;
  while (*p && *p != '"') {
    if (*p == '\\') {
      ++p;
      switch (*p) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'n': out.push_back('\n'); break;
        case 't': out.push_back('\t'); break;
        case 'r': out.push_back('\r'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'u': {
          // Copy the 4 hex digits through verbatim as an escape; we do not
          // need to decode since output re-escapes only control chars.
          char buf[7] = "\\u";
          for (int i = 0; i < 4 && p[1]; ++i) buf[2 + i] = *(++p);
          out.append(buf);
          break;
        }
        default: out.push_back(*p); break;
      }
      if (*p) ++p;
    } else {
      out.push_back(*p++);
    }
  }
  if (*p != '"') return false;
  ++p;
  return true;
}

static bool parseNumber(const char*& p, JsonValue& out) {
  const char* start = p;
  bool isInt = true;
  if (*p == '-' || *p == '+') ++p;
  while (*p) {
    char c = *p;
    if (c >= '0' && c <= '9') {
      ++p;
    } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
      isInt = false;
      ++p;
    } else {
      break;
    }
  }
  std::string tok(start, (size_t)(p - start));
  if (tok.empty()) return false;
  out.type = JsonValue::Num;
  out.isInt = isInt;
  if (isInt) {
    out.inum = strtoll(tok.c_str(), nullptr, 10);
    out.num = (double)out.inum;
  } else {
    out.num = strtod(tok.c_str(), nullptr);
    out.inum = (long long)out.num;
  }
  return true;
}

static bool parseValue(const char*& p, JsonValue& out) {
  skipWs(p);
  if (*p == '{') {
    ++p;
    out.type = JsonValue::Obj;
    skipWs(p);
    if (*p == '}') { ++p; return true; }
    while (true) {
      skipWs(p);
      std::string key;
      if (!parseString(p, key)) return false;
      skipWs(p);
      if (*p != ':') return false;
      ++p;
      JsonValue val;
      if (!parseValue(p, val)) return false;
      out.obj.push_back(std::make_pair(std::move(key), std::move(val)));
      skipWs(p);
      if (*p == ',') { ++p; continue; }
      if (*p == '}') { ++p; return true; }
      return false;
    }
  } else if (*p == '[') {
    ++p;
    out.type = JsonValue::Arr;
    skipWs(p);
    if (*p == ']') { ++p; return true; }
    while (true) {
      JsonValue val;
      if (!parseValue(p, val)) return false;
      out.arr.push_back(std::move(val));
      skipWs(p);
      if (*p == ',') { ++p; continue; }
      if (*p == ']') { ++p; return true; }
      return false;
    }
  } else if (*p == '"') {
    out.type = JsonValue::Str;
    return parseString(p, out.str);
  } else if (strncmp(p, "true", 4) == 0) {
    p += 4; out.type = JsonValue::Bool; out.b = true; return true;
  } else if (strncmp(p, "false", 5) == 0) {
    p += 5; out.type = JsonValue::Bool; out.b = false; return true;
  } else if (strncmp(p, "null", 4) == 0) {
    p += 4; out.type = JsonValue::Null; return true;
  } else {
    return parseNumber(p, out);
  }
}

// ---------------------------------------------------------------------------
// JsonValue construction helpers
// ---------------------------------------------------------------------------
static JsonValue jObj() { JsonValue v; v.type = JsonValue::Obj; return v; }
static JsonValue jStr(const std::string& s) { JsonValue v; v.type = JsonValue::Str; v.str = s; return v; }
static JsonValue jInt(long long n) { JsonValue v; v.type = JsonValue::Num; v.isInt = true; v.inum = n; v.num = (double)n; return v; }
static JsonValue jChar(char ph) { return jStr(std::string(1, ph)); }

static void put(JsonValue& o, const char* k, JsonValue v) {
  o.obj.push_back(std::make_pair(std::string(k), std::move(v)));
}

// Copy a parsed sub-value into args under key k if it exists.
static void putIf(JsonValue& args, const char* k, const JsonValue* src) {
  if (src != nullptr) args.obj.push_back(std::make_pair(std::string(k), *src));
}

// ---------------------------------------------------------------------------
// tid allocator: sequential integer tids per pid, keyed by lane name
// ---------------------------------------------------------------------------
struct TidAlloc {
  // pid -> ordered list of (name, tid)
  std::map<int, std::vector<std::pair<std::string, int>>> maps;
  int get(int pid, const std::string& name) {
    std::vector<std::pair<std::string, int>>& v = maps[pid];
    for (size_t i = 0; i < v.size(); ++i) {
      if (v[i].first == name) return v[i].second;
    }
    int id = (int)v.size() + 1;
    v.push_back(std::make_pair(name, id));
    return id;
  }
};

// ---------------------------------------------------------------------------
// Serializer (JsonValue -> FILE*)
// ---------------------------------------------------------------------------
static void emitString(FILE* f, const std::string& s) {
  fputc('"', f);
  for (size_t i = 0; i < s.size(); ++i) {
    unsigned char c = (unsigned char)s[i];
    switch (c) {
      case '"': fputs("\\\"", f); break;
      case '\\': fputs("\\\\", f); break;
      case '\n': fputs("\\n", f); break;
      case '\r': fputs("\\r", f); break;
      case '\t': fputs("\\t", f); break;
      case '\b': fputs("\\b", f); break;
      case '\f': fputs("\\f", f); break;
      default:
        if (c < 0x20) {
          fprintf(f, "\\u%04x", (unsigned)c);
        } else {
          fputc((int)c, f);
        }
        break;
    }
  }
  fputc('"', f);
}

static void emitValue(FILE* f, const JsonValue& v) {
  switch (v.type) {
    case JsonValue::Null: fputs("null", f); break;
    case JsonValue::Bool: fputs(v.b ? "true" : "false", f); break;
    case JsonValue::Num:
      if (v.isInt) {
        fprintf(f, "%lld", v.inum);
      } else if (v.num != v.num) {
        fputs("\"nan\"", f);
      } else {
        fprintf(f, "%g", v.num);
      }
      break;
    case JsonValue::Str: emitString(f, v.str); break;
    case JsonValue::Arr:
      fputc('[', f);
      for (size_t i = 0; i < v.arr.size(); ++i) {
        if (i) fputc(',', f);
        emitValue(f, v.arr[i]);
      }
      fputc(']', f);
      break;
    case JsonValue::Obj:
      fputc('{', f);
      for (size_t i = 0; i < v.obj.size(); ++i) {
        if (i) fputc(',', f);
        emitString(f, v.obj[i].first);
        fputc(':', f);
        emitValue(f, v.obj[i].second);
      }
      fputc('}', f);
      break;
  }
}

// ---------------------------------------------------------------------------
// Trace event builders (mirror convert_to_chrome_trace.py)
// ---------------------------------------------------------------------------

// Create a complete X (duration) event object.
static JsonValue mkSpan(const std::string& name, const char* cat,
                        long long ts, long long dur, int pid, int tid,
                        JsonValue args) {
  JsonValue e = jObj();
  put(e, "name", jStr(name));
  put(e, "cat", jStr(cat));
  put(e, "ph", jChar('X'));
  put(e, "ts", jInt(ts));
  put(e, "dur", jInt(dur));
  put(e, "pid", jInt(pid));
  put(e, "tid", jInt(tid));
  put(e, "args", std::move(args));
  return e;
}

// Create an instant (i) event with thread scope.
static JsonValue mkInstant(const std::string& name, long long ts,
                           int pid, int tid, JsonValue args) {
  JsonValue e = jObj();
  put(e, "name", jStr(name));
  put(e, "cat", jStr("state"));
  put(e, "ph", jChar('i'));
  put(e, "s", jChar('t'));
  put(e, "ts", jInt(ts));
  put(e, "pid", jInt(pid));
  put(e, "tid", jInt(tid));
  put(e, "args", std::move(args));
  return e;
}

static void convertProxyCtrl(const JsonValue& proxy, std::vector<JsonValue>& events,
                             int pid, TidAlloc& tids) {
  long long start = proxy.find("proxy_start_ts") ? proxy.find("proxy_start_ts")->asInt() : 0;
  long long stop = proxy.find("proxy_stop_ts") ? proxy.find("proxy_stop_ts")->asInt() : 0;
  long long dur = stop - start;
  const JsonValue* states = proxy.find("proxy_states");
  std::string name = "ProxyCtrl";
  if (states && states->type == JsonValue::Arr && !states->arr.empty()) {
    const JsonValue* st = states->arr[0].find("state");
    if (st) name = st->asStr("ProxyCtrl");
  }
  int tid = tids.get(pid, "ProxyCtrl");

  JsonValue args = jObj();
  putIf(args, "proxy_id", proxy.find("proxy_id"));
  put(args, "duration_us", jInt(dur));
  events.push_back(mkSpan(name, "proxy_ctrl", start, dur, pid, tid, std::move(args)));

  if (states && states->type == JsonValue::Arr) {
    for (size_t i = 0; i < states->arr.size(); ++i) {
      const JsonValue& s = states->arr[i];
      JsonValue a = jObj();
      putIf(a, "state_id", s.find("state_id"));
      putIf(a, "appended_proxy_ops", s.find("appended_proxy_ops"));
      long long sts = s.find("state_ts") ? s.find("state_ts")->asInt() : 0;
      const char* sn = s.find("state") ? s.find("state")->asStr("") : "";
      events.push_back(mkInstant(sn, sts, pid, tid, std::move(a)));
    }
  }
}

static void convertKernelCh(const JsonValue& ev, const JsonValue* opSn,
                            int pid, TidAlloc& tids, std::vector<JsonValue>& events) {
  const JsonValue* ch = ev.find("channel_id");
  const JsonValue* tsO = ev.find("event_trace_ts");
  long long kStart = 0, kStop = 0;
  if (tsO) {
    if (tsO->find("kernel_start_ts")) kStart = tsO->find("kernel_start_ts")->asInt();
    if (tsO->find("kernel_stop_ts")) kStop = tsO->find("kernel_stop_ts")->asInt();
  }
  int tid = tids.get(pid, "Kernel");
  char nameBuf[32];
  snprintf(nameBuf, sizeof(nameBuf), "KernelCh%lld", ch ? ch->asInt() : 0);

  JsonValue args = jObj();
  putIf(args, "channel_id", ch);
  if (opSn) args.obj.push_back(std::make_pair(std::string("op_sn"), *opSn));
  putIf(args, "trace_sn", ev.find("event_trace_sn"));
  events.push_back(mkSpan(nameBuf, "kernel", kStart, kStop - kStart, pid, tid, std::move(args)));
}

// Emit ProxyStep instant states from a step record list.
static void convertProxySteps(const JsonValue& container, const JsonValue* proxyId,
                              const JsonValue* opSn, int pid, int tid,
                              std::vector<JsonValue>& events) {
  const JsonValue* steps = container.find("proxy_step_records");
  std::vector<const JsonValue*> stepPtrs;
  if (steps && steps->type == JsonValue::Arr) {
    for (size_t i = 0; i < steps->arr.size(); ++i) stepPtrs.push_back(&steps->arr[i]);
  } else {
    const JsonValue* legacy = container.find("events");
    if (legacy && legacy->type == JsonValue::Arr) {
      for (size_t i = 0; i < legacy->arr.size(); ++i) {
        const JsonValue* et = legacy->arr[i].find("event_type");
        if (et && strcmp(et->asStr(""), "ProxyStep") == 0) stepPtrs.push_back(&legacy->arr[i]);
      }
    }
  }
  for (size_t i = 0; i < stepPtrs.size(); ++i) {
    const JsonValue* step = stepPtrs[i];
    const JsonValue* stStates = step->find("proxy_states");
    if (!stStates || stStates->type != JsonValue::Arr) continue;
    for (size_t j = 0; j < stStates->arr.size(); ++j) {
      const JsonValue& s = stStates->arr[j];
      JsonValue a = jObj();
      putIf(a, "state_id", s.find("state_id"));
      if (proxyId) a.obj.push_back(std::make_pair(std::string("proxy_id"), *proxyId));
      putIf(a, "step", step->find("proxy_step"));
      if (opSn) a.obj.push_back(std::make_pair(std::string("op_sn"), *opSn));
      putIf(a, "trans_size", s.find("trans_size"));
      long long sts = s.find("state_ts") ? s.find("state_ts")->asInt() : 0;
      const char* sn = s.find("state") ? s.find("state")->asStr("") : "";
      events.push_back(mkInstant(sn, sts, pid, tid, std::move(a)));
    }
  }
}

// Build the "Proxy chN dir->peerP" lane name.
static std::string proxyLaneName(long long channelId, const char* direction, long long peer) {
  char buf[96];
  // U+2192 (right arrow) encoded as UTF-8 bytes.
  snprintf(buf, sizeof(buf), "Proxy ch%lld %s\xe2\x86\x92peer%lld", channelId, direction, peer);
  return std::string(buf);
}

static void convertProxyOp(const JsonValue& ev, const JsonValue* opSn,
                           int pid, TidAlloc& tids, std::vector<JsonValue>& events) {
  long long channelId = ev.find("proxy_channel_id") ? ev.find("proxy_channel_id")->asInt() : 0;
  const char* direction = ev.find("proxy_direction") ? ev.find("proxy_direction")->asStr("Unknown") : "Unknown";
  long long peer = ev.find("proxy_peer") ? ev.find("proxy_peer")->asInt() : -1;
  long long opStart = ev.find("proxy_start_ts") ? ev.find("proxy_start_ts")->asInt() : 0;
  long long opStop = ev.find("proxy_stop_ts") ? ev.find("proxy_stop_ts")->asInt() : 0;
  const JsonValue* proxyId = ev.find("proxy_id");

  int tid = tids.get(pid, proxyLaneName(channelId, direction, peer));

  char nameBuf[64];
  snprintf(nameBuf, sizeof(nameBuf), "ProxyOp %s ch%lld", direction, channelId);

  JsonValue args = jObj();
  putIf(args, "proxy_id", proxyId);
  put(args, "direction", jStr(direction));
  put(args, "channel_id", jInt(channelId));
  put(args, "peer", jInt(peer));
  putIf(args, "n_steps", ev.find("proxy_n_steps"));
  putIf(args, "chunk_size", ev.find("proxy_chunk_size"));
  putIf(args, "trans_size", ev.find("proxy_trans_size"));
  putIf(args, "duration_us", ev.find("proxy_duration_us"));
  putIf(args, "step_count", ev.find("proxy_step_count"));
  if (opSn) args.obj.push_back(std::make_pair(std::string("op_sn"), *opSn));
  events.push_back(mkSpan(nameBuf, "proxy", opStart, opStop - opStart, pid, tid, std::move(args)));

  const JsonValue* states = ev.find("proxy_states");
  if (states && states->type == JsonValue::Arr) {
    for (size_t i = 0; i < states->arr.size(); ++i) {
      const JsonValue& s = states->arr[i];
      JsonValue a = jObj();
      putIf(a, "state_id", s.find("state_id"));
      if (proxyId) a.obj.push_back(std::make_pair(std::string("proxy_id"), *proxyId));
      if (opSn) a.obj.push_back(std::make_pair(std::string("op_sn"), *opSn));
      long long sts = s.find("state_ts") ? s.find("state_ts")->asInt() : 0;
      const char* sn = s.find("state") ? s.find("state")->asStr("") : "";
      events.push_back(mkInstant(sn, sts, pid, tid, std::move(a)));
    }
  }
  convertProxySteps(ev, proxyId, opSn, pid, tid, events);
}

static void convertStandaloneProxyOp(const JsonValue& proxy, std::vector<JsonValue>& events,
                                     int pid, TidAlloc& tids) {
  long long channelId = proxy.find("proxy_channel_id") ? proxy.find("proxy_channel_id")->asInt() : 0;
  const char* direction = proxy.find("proxy_direction") ? proxy.find("proxy_direction")->asStr("Unknown") : "Unknown";
  long long peer = proxy.find("proxy_peer") ? proxy.find("proxy_peer")->asInt() : -1;
  long long opStart = proxy.find("proxy_start_ts") ? proxy.find("proxy_start_ts")->asInt() : 0;
  long long opStop = proxy.find("proxy_stop_ts") ? proxy.find("proxy_stop_ts")->asInt() : 0;
  const JsonValue* proxyId = proxy.find("proxy_id");

  int tid = tids.get(pid, proxyLaneName(channelId, direction, peer));

  char nameBuf[64];
  snprintf(nameBuf, sizeof(nameBuf), "ProxyOp %s ch%lld", direction, channelId);

  JsonValue args = jObj();
  putIf(args, "proxy_id", proxyId);
  put(args, "direction", jStr(direction));
  put(args, "channel_id", jInt(channelId));
  put(args, "peer", jInt(peer));
  putIf(args, "n_steps", proxy.find("proxy_n_steps"));
  putIf(args, "duration_us", proxy.find("proxy_duration_us"));
  putIf(args, "step_count", proxy.find("proxy_step_count"));
  events.push_back(mkSpan(nameBuf, "proxy", opStart, opStop - opStart, pid, tid, std::move(args)));

  const JsonValue* states = proxy.find("proxy_states");
  if (states && states->type == JsonValue::Arr) {
    for (size_t i = 0; i < states->arr.size(); ++i) {
      const JsonValue& s = states->arr[i];
      JsonValue a = jObj();
      putIf(a, "state_id", s.find("state_id"));
      if (proxyId) a.obj.push_back(std::make_pair(std::string("proxy_id"), *proxyId));
      long long sts = s.find("state_ts") ? s.find("state_ts")->asInt() : 0;
      const char* sn = s.find("state") ? s.find("state")->asStr("") : "";
      events.push_back(mkInstant(sn, sts, pid, tid, std::move(a)));
    }
  }
  convertProxySteps(proxy, proxyId, nullptr, pid, tid, events);
}

static void convertCollPerf(const JsonValue& rec, std::vector<JsonValue>& events,
                            int pid, TidAlloc& tids) {
  const JsonValue* header = rec.find("header");
  const JsonValue* coll = rec.find("coll_perf");
  if (!coll) return;
  const char* collName = coll->find("coll") ? coll->find("coll")->asStr("") : "";
  const JsonValue* collSn = coll->find("coll_sn");
  const JsonValue* ts = coll->find("event_trace_ts");
  long long start = 0, stop = 0;
  if (ts) {
    if (ts->find("coll_start_ts")) start = ts->find("coll_start_ts")->asInt();
    if (ts->find("coll_stop_ts")) stop = ts->find("coll_stop_ts")->asInt();
  }
  int tid = tids.get(pid, "Collective");

  char nameBuf[96];
  snprintf(nameBuf, sizeof(nameBuf), "%s #%lld", collName, collSn ? collSn->asInt() : 0);

  JsonValue args = jObj();
  putIf(args, "coll_sn", collSn);
  putIf(args, "msg_size_bytes", coll->find("coll_msg_size_bytes"));
  putIf(args, "exec_time_us", coll->find("coll_exec_time_us"));
  putIf(args, "timing_source", coll->find("coll_timing_source"));
  putIf(args, "algobw_gbs", coll->find("coll_algobw_gbs"));
  putIf(args, "busbw_gbs", coll->find("coll_busbw_gbs"));
  if (header) {
    putIf(args, "comm_id", header->find("id"));
    putIf(args, "rank", header->find("rank"));
  }
  events.push_back(mkSpan(nameBuf, "coll", start, stop - start, pid, tid, std::move(args)));

  const JsonValue* evs = coll->find("events");
  if (evs && evs->type == JsonValue::Arr) {
    for (size_t i = 0; i < evs->arr.size(); ++i) {
      const JsonValue& ev = evs->arr[i];
      const JsonValue* et = ev.find("event_type");
      const char* type = et ? et->asStr("") : "";
      if (strcmp(type, "KernelCh") == 0) {
        convertKernelCh(ev, collSn, pid, tids, events);
      } else if (strcmp(type, "ProxyOp") == 0) {
        convertProxyOp(ev, collSn, pid, tids, events);
      }
    }
  }
}

static void convertP2pPerf(const JsonValue& rec, std::vector<JsonValue>& events,
                           int pid, TidAlloc& tids) {
  const JsonValue* header = rec.find("header");
  const JsonValue* p2p = rec.find("p2p_perf");
  if (!p2p) return;
  const char* p2pName = p2p->find("p2p") ? p2p->find("p2p")->asStr("") : "";
  const JsonValue* p2pSn = p2p->find("p2p_sn");
  long long peer = p2p->find("p2p_peer") ? p2p->find("p2p_peer")->asInt() : -1;
  const JsonValue* ts = p2p->find("event_trace_ts");
  long long start = 0, stop = 0;
  if (ts) {
    if (ts->find("p2p_start_ts")) start = ts->find("p2p_start_ts")->asInt();
    if (ts->find("p2p_stop_ts")) stop = ts->find("p2p_stop_ts")->asInt();
  }
  int tid = tids.get(pid, "P2P");

  char nameBuf[96];
  snprintf(nameBuf, sizeof(nameBuf), "%s #%lld peer%lld", p2pName, p2pSn ? p2pSn->asInt() : 0, peer);

  JsonValue args = jObj();
  putIf(args, "p2p_sn", p2pSn);
  put(args, "peer", jInt(peer));
  putIf(args, "msg_size_bytes", p2p->find("p2p_msg_size_bytes"));
  putIf(args, "exec_time_us", p2p->find("p2p_exec_time_us"));
  putIf(args, "timing_source", p2p->find("p2p_timing_source"));
  putIf(args, "algobw_gbs", p2p->find("p2p_algobw_gbs"));
  putIf(args, "busbw_gbs", p2p->find("p2p_busbw_gbs"));
  if (header) {
    putIf(args, "comm_id", header->find("id"));
    putIf(args, "rank", header->find("rank"));
  }
  events.push_back(mkSpan(nameBuf, "p2p", start, stop - start, pid, tid, std::move(args)));

  const JsonValue* evs = p2p->find("events");
  if (evs && evs->type == JsonValue::Arr) {
    for (size_t i = 0; i < evs->arr.size(); ++i) {
      const JsonValue& ev = evs->arr[i];
      const JsonValue* et = ev.find("event_type");
      const char* type = et ? et->asStr("") : "";
      if (strcmp(type, "KernelCh") == 0) {
        convertKernelCh(ev, p2pSn, pid, tids, events);
      } else if (strcmp(type, "ProxyOp") == 0) {
        convertProxyOp(ev, p2pSn, pid, tids, events);
      }
    }
  }
}

struct SimpleKey { const char* key; const char* typeName; };
static const SimpleKey kSimpleKeys[] = {
  {"group", "Group"},
  {"group_api", "GroupApi"},
  {"coll_api", "CollApi"},
  {"p2p_api", "P2pApi"},
  {"kernel_launch", "KernelLaunch"},
  {"net_plugin", "NetPlugin"},
};

static void convertSimpleEvent(const char* typeName, const JsonValue& ev,
                               std::vector<JsonValue>& events, int pid, TidAlloc& tids) {
  long long start = ev.find("start_ts") ? ev.find("start_ts")->asInt() : 0;
  long long stop = ev.find("stop_ts") ? ev.find("stop_ts")->asInt() : 0;
  long long dur = stop - start;
  if (dur < 0) dur = 0;
  int tid = tids.get(pid, typeName);

  std::string name = typeName;
  const JsonValue* func = ev.find("func");
  if (func && func->type == JsonValue::Str) {
    name = std::string(typeName) + ": " + func->str;
  }

  std::string cat = typeName;
  for (size_t i = 0; i < cat.size(); ++i) cat[i] = (char)tolower((unsigned char)cat[i]);

  static const char* kArgKeys[] = {
    "rank", "func", "count", "datatype", "root",
    "group_depth", "graph_captured", "net_plugin_id", "duration_us",
  };
  JsonValue args = jObj();
  for (size_t i = 0; i < sizeof(kArgKeys) / sizeof(kArgKeys[0]); ++i) {
    putIf(args, kArgKeys[i], ev.find(kArgKeys[i]));
  }

  events.push_back(mkSpan(name, cat.c_str(), start, dur, pid, tid, std::move(args)));
}

// ---------------------------------------------------------------------------
// pid allocation keyed by (rank, hostname)
// ---------------------------------------------------------------------------
struct PidAlloc {
  std::map<std::string, int> pidMap;                       // "rank\x1fhost" -> pid
  std::vector<std::pair<int, std::pair<int, std::string>>> order;  // pid -> (rank, host)
  int next;
  PidAlloc() : next(1000) {}
  int get(int rank, const std::string& host) {
    std::string key = std::to_string(rank) + "\x1f" + host;
    std::map<std::string, int>::iterator it = pidMap.find(key);
    if (it != pidMap.end()) return it->second;
    int pid = next++;
    pidMap[key] = pid;
    order.push_back(std::make_pair(pid, std::make_pair(rank, host)));
    return pid;
  }
};

}  // namespace

inspectorResult_t inspectorConvertLogToChromeTrace(const char* logPath,
                                                   const char* outPath) {
  if (!logPath || !outPath) return inspectorMemoryError;

  std::ifstream in(logPath);
  if (!in.is_open()) {
    INFO_INSPECTOR("NCCL Inspector: trace converter cannot open log %s", logPath);
    return inspectorFileOpenError;
  }

  PidAlloc pids;
  TidAlloc tids;
  std::vector<JsonValue> events;

  std::string line;
  size_t lineNo = 0;
  size_t recordCount = 0;
  while (std::getline(in, line)) {
    ++lineNo;
    // Trim leading whitespace to test for empty lines.
    size_t s = 0;
    while (s < line.size() && (line[s] == ' ' || line[s] == '\t' || line[s] == '\r')) ++s;
    if (s >= line.size()) continue;

    JsonValue rec;
    const char* p = line.c_str();
    if (!parseValue(p, rec) || rec.type != JsonValue::Obj) {
      INFO_INSPECTOR("NCCL Inspector: trace converter parse error at %s:%zu",
                     logPath, lineNo);
      continue;
    }
    ++recordCount;

    const JsonValue* header = rec.find("header");
    const JsonValue* metadata = rec.find("metadata");
    int rank = (header && header->find("rank")) ? (int)header->find("rank")->asInt() : 0;
    std::string host = "unknown";
    if (metadata && metadata->find("hostname")) host = metadata->find("hostname")->asStr("unknown");
    int pid = pids.get(rank, host);

    if (rec.find("proxy_ctrl")) {
      convertProxyCtrl(*rec.find("proxy_ctrl"), events, pid, tids);
    } else if (rec.find("proxy_op")) {
      convertStandaloneProxyOp(*rec.find("proxy_op"), events, pid, tids);
    } else if (rec.find("coll_perf")) {
      convertCollPerf(rec, events, pid, tids);
    } else if (rec.find("p2p_perf")) {
      convertP2pPerf(rec, events, pid, tids);
    } else {
      for (size_t i = 0; i < sizeof(kSimpleKeys) / sizeof(kSimpleKeys[0]); ++i) {
        const JsonValue* ev = rec.find(kSimpleKeys[i].key);
        if (ev) {
          convertSimpleEvent(kSimpleKeys[i].typeName, *ev, events, pid, tids);
          break;
        }
      }
    }
  }
  in.close();

  if (recordCount == 0) {
    INFO_INSPECTOR("NCCL Inspector: trace converter found no records in %s", logPath);
    return inspectorSuccess;
  }

  // Build metadata events (process_name, process_sort_index, thread_name),
  // ordered by pid.
  std::vector<JsonValue> meta;
  for (size_t i = 0; i < pids.order.size(); ++i) {
    int pid = pids.order[i].first;
    int rank = pids.order[i].second.first;
    const std::string& host = pids.order[i].second.second;

    char pname[320];
    snprintf(pname, sizeof(pname), "Rank %d (%s)", rank, host.c_str());
    JsonValue pn = jObj();
    put(pn, "name", jStr("process_name"));
    put(pn, "ph", jChar('M'));
    put(pn, "pid", jInt(pid));
    JsonValue pnArgs = jObj();
    put(pnArgs, "name", jStr(pname));
    put(pn, "args", std::move(pnArgs));
    meta.push_back(std::move(pn));

    JsonValue si = jObj();
    put(si, "name", jStr("process_sort_index"));
    put(si, "ph", jChar('M'));
    put(si, "pid", jInt(pid));
    JsonValue siArgs = jObj();
    put(siArgs, "sort_index", jInt(rank));
    put(si, "args", std::move(siArgs));
    meta.push_back(std::move(si));

    std::map<int, std::vector<std::pair<std::string, int>>>::iterator lit = tids.maps.find(pid);
    if (lit != tids.maps.end()) {
      std::vector<std::pair<std::string, int>> lanes = lit->second;
      // Order lanes by tid.
      for (size_t a = 0; a < lanes.size(); ++a) {
        for (size_t b = a + 1; b < lanes.size(); ++b) {
          if (lanes[b].second < lanes[a].second) std::swap(lanes[a], lanes[b]);
        }
      }
      for (size_t j = 0; j < lanes.size(); ++j) {
        JsonValue tn = jObj();
        put(tn, "name", jStr("thread_name"));
        put(tn, "ph", jChar('M'));
        put(tn, "pid", jInt(pid));
        put(tn, "tid", jInt(lanes[j].second));
        JsonValue tnArgs = jObj();
        put(tnArgs, "name", jStr(lanes[j].first));
        put(tn, "args", std::move(tnArgs));
        meta.push_back(std::move(tn));
      }
    }
  }

  FILE* out = fopen(outPath, "w");
  if (!out) {
    INFO_INSPECTOR("NCCL Inspector: trace converter cannot open output %s", outPath);
    return inspectorFileOpenError;
  }

  fputs("{\"traceEvents\":[", out);
  bool first = true;
  for (size_t i = 0; i < meta.size(); ++i) {
    if (!first) fputc(',', out);
    first = false;
    emitValue(out, meta[i]);
  }
  for (size_t i = 0; i < events.size(); ++i) {
    if (!first) fputc(',', out);
    first = false;
    emitValue(out, events[i]);
  }
  fputs("],\"displayTimeUnit\":\"us\"}", out);
  fclose(out);
  chmod(outPath, 0666);

  INFO_INSPECTOR("NCCL Inspector: wrote Chrome trace %s (%zu records, %zu events)",
                 outPath, recordCount, meta.size() + events.size());
  return inspectorSuccess;
}
