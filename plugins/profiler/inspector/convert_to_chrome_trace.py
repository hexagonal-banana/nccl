#!/usr/bin/env python3
"""Convert NCCL Inspector profiler log files to Chrome Trace Event Format JSON.

Usage:
    python convert_to_chrome_trace.py <input_log_or_dir> [<input_log_or_dir> ...] [-o output.json]

Examples:
    # Single file
    python convert_to_chrome_trace.py results/dump/node-pid12345.log -o trace.json

    # All logs in a directory
    python convert_to_chrome_trace.py results/dump/ -o trace.json

    # Limit output to first 500 records (for large dumps)
    python convert_to_chrome_trace.py results/dump/ -o trace.json --max-records 500

Output is Chrome Trace JSON loadable in Perfetto UI or chrome://tracing.
Uses numeric pid/tid as required by Perfetto.
"""

import argparse
import json
import sys
from pathlib import Path

# Simple event keys in current inspector format
SIMPLE_EVENT_KEYS = {
    "group": "Group",
    "group_api": "GroupApi",
    "coll_api": "CollApi",
    "p2p_api": "P2pApi",
    "kernel_launch": "KernelLaunch",
    "net_plugin": "NetPlugin",
}


class TidAlloc:
    """Allocate sequential integer tids per pid."""
    def __init__(self):
        self._maps = {}  # pid -> {name: int}

    def get(self, pid, name):
        m = self._maps.setdefault(pid, {})
        if name not in m:
            m[name] = len(m) + 1
        return m[name]

    def items(self, pid):
        return sorted(self._maps.get(pid, {}).items(), key=lambda x: x[1])


def collect_log_files(paths):
    """Resolve input paths to a list of .log files."""
    files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("*.log")))
        elif path.is_file():
            files.append(path)
        else:
            print(f"WARNING: skipping non-existent path: {p}", file=sys.stderr)
    return files


def parse_log_file(filepath):
    """Parse a log file, yielding one parsed JSON dict per line."""
    with open(filepath, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: {filepath}:{line_no}: JSON parse error: {e}", file=sys.stderr)


def convert_proxy_ctrl(proxy, trace_events, pid, tids):
    """Convert a proxy_ctrl record to trace events."""
    proxy_id = proxy["proxy_id"]
    start_ts = proxy["proxy_start_ts"]
    stop_ts = proxy["proxy_stop_ts"]
    dur = stop_ts - start_ts
    states = proxy.get("proxy_states", [])
    name = states[0]["state"] if states else "ProxyCtrl"
    tid = tids.get(pid, "ProxyCtrl")

    trace_events.append({
        "name": name, "cat": "proxy_ctrl", "ph": "X",
        "ts": start_ts, "dur": dur, "pid": pid, "tid": tid,
        "args": {"proxy_id": proxy_id, "duration_us": dur},
    })
    for state in states:
        a = {"state_id": state["state_id"]}
        if "appended_proxy_ops" in state:
            a["appended_proxy_ops"] = state["appended_proxy_ops"]
        trace_events.append({
            "name": state["state"], "cat": "state", "ph": "i", "s": "t",
            "ts": state["state_ts"], "pid": pid, "tid": tid, "args": a,
        })


def convert_coll_perf(record, trace_events, pid, tids):
    """Convert a coll_perf record to trace events."""
    header = record["header"]
    coll = record["coll_perf"]
    rank = header["rank"]
    coll_name = coll["coll"]
    coll_sn = coll["coll_sn"]
    msg_size = coll["coll_msg_size_bytes"]

    ts = coll.get("event_trace_ts", {})
    coll_start = ts.get("coll_start_ts", 0)
    coll_stop = ts.get("coll_stop_ts", 0)
    tid = tids.get(pid, "Collective")

    trace_events.append({
        "name": f"{coll_name} #{coll_sn}", "cat": "coll", "ph": "X",
        "ts": coll_start, "dur": coll_stop - coll_start, "pid": pid, "tid": tid,
        "args": {
            "coll_sn": coll_sn, "msg_size_bytes": msg_size,
            "exec_time_us": coll["coll_exec_time_us"],
            "timing_source": coll.get("coll_timing_source", ""),
            "algobw_gbs": coll.get("coll_algobw_gbs"),
            "busbw_gbs": coll.get("coll_busbw_gbs"),
            "comm_id": header["id"], "rank": rank,
        },
    })

    for ev in coll.get("events", []):
        if ev["event_type"] == "KernelCh":
            convert_kernel_ch(ev, coll_sn, pid, tids, trace_events)
        elif ev["event_type"] == "ProxyOp":
            convert_proxy_op(ev, coll_sn, pid, tids, trace_events)


def convert_p2p_perf(record, trace_events, pid, tids):
    """Convert a p2p_perf record to trace events."""
    header = record["header"]
    p2p = record["p2p_perf"]
    rank = header["rank"]
    p2p_name = p2p["p2p"]
    p2p_sn = p2p["p2p_sn"]
    msg_size = p2p["p2p_msg_size_bytes"]
    peer = p2p.get("p2p_peer", -1)

    ts = p2p.get("event_trace_ts", {})
    start = ts.get("p2p_start_ts", 0)
    stop = ts.get("p2p_stop_ts", 0)
    tid = tids.get(pid, "P2P")

    trace_events.append({
        "name": f"{p2p_name} #{p2p_sn} peer{peer}", "cat": "p2p", "ph": "X",
        "ts": start, "dur": stop - start, "pid": pid, "tid": tid,
        "args": {
            "p2p_sn": p2p_sn, "peer": peer, "msg_size_bytes": msg_size,
            "exec_time_us": p2p["p2p_exec_time_us"],
            "timing_source": p2p.get("p2p_timing_source", ""),
            "algobw_gbs": p2p.get("p2p_algobw_gbs"),
            "busbw_gbs": p2p.get("p2p_busbw_gbs"),
            "comm_id": header["id"], "rank": rank,
        },
    })

    for ev in p2p.get("events", []):
        if ev["event_type"] == "KernelCh":
            convert_kernel_ch(ev, p2p_sn, pid, tids, trace_events)
        elif ev["event_type"] == "ProxyOp":
            convert_proxy_op(ev, p2p_sn, pid, tids, trace_events)


def convert_kernel_ch(ev, op_sn, pid, tids, trace_events):
    """Convert a KernelCh event."""
    ch = ev["channel_id"]
    k_start = ev["event_trace_ts"]["kernel_start_ts"]
    k_stop = ev["event_trace_ts"]["kernel_stop_ts"]
    tid = tids.get(pid, "Kernel")
    trace_events.append({
        "name": f"KernelCh{ch}", "cat": "kernel", "ph": "X",
        "ts": k_start, "dur": k_stop - k_start, "pid": pid, "tid": tid,
        "args": {"channel_id": ch, "op_sn": op_sn, "trace_sn": ev.get("event_trace_sn", {})},
    })


def convert_proxy_op(ev, op_sn, pid, tids, trace_events):
    """Convert a ProxyOp event (nested in coll_perf/p2p_perf)."""
    proxy_id = ev["proxy_id"]
    direction = ev["proxy_direction"]
    channel_id = ev["proxy_channel_id"]
    peer = ev["proxy_peer"]
    op_start = ev["proxy_start_ts"]
    op_stop = ev["proxy_stop_ts"]

    tid_name = f"Proxy ch{channel_id} {direction}\u2192peer{peer}"
    tid = tids.get(pid, tid_name)

    trace_events.append({
        "name": f"ProxyOp {direction} ch{channel_id}", "cat": "proxy", "ph": "X",
        "ts": op_start, "dur": op_stop - op_start, "pid": pid, "tid": tid,
        "args": {
            "proxy_id": proxy_id, "direction": direction,
            "channel_id": channel_id, "peer": peer,
            "n_steps": ev.get("proxy_n_steps"),
            "chunk_size": ev.get("proxy_chunk_size"),
            "trans_size": ev.get("proxy_trans_size"),
            "duration_us": ev.get("proxy_duration_us"),
            "step_count": ev.get("proxy_step_count"),
            "op_sn": op_sn,
        },
    })

    for state in ev.get("proxy_states", []):
        trace_events.append({
            "name": state["state"], "cat": "state", "ph": "i", "s": "t",
            "ts": state["state_ts"], "pid": pid, "tid": tid,
            "args": {"state_id": state["state_id"], "proxy_id": proxy_id, "op_sn": op_sn},
        })

    # ProxyStep sub-events (current: proxy_step_records, legacy: events)
    steps = ev.get("proxy_step_records", None)
    if steps is None:
        steps = [e for e in ev.get("events", []) if e.get("event_type") == "ProxyStep"]
    for step_ev in steps:
        for state in step_ev.get("proxy_states", []):
            a = {"state_id": state["state_id"], "proxy_id": proxy_id,
                 "step": step_ev["proxy_step"], "op_sn": op_sn}
            if "trans_size" in state:
                a["trans_size"] = state["trans_size"]
            trace_events.append({
                "name": state["state"], "cat": "state", "ph": "i", "s": "t",
                "ts": state["state_ts"], "pid": pid, "tid": tid, "args": a,
            })


def convert_standalone_proxy_op(proxy, trace_events, pid, tids):
    """Convert a standalone proxy_op record."""
    proxy_id = proxy["proxy_id"]
    direction = proxy.get("proxy_direction", "Unknown")
    channel_id = proxy.get("proxy_channel_id", 0)
    peer = proxy.get("proxy_peer", -1)
    op_start = proxy["proxy_start_ts"]
    op_stop = proxy["proxy_stop_ts"]

    tid_name = f"Proxy ch{channel_id} {direction}\u2192peer{peer}"
    tid = tids.get(pid, tid_name)

    trace_events.append({
        "name": f"ProxyOp {direction} ch{channel_id}", "cat": "proxy", "ph": "X",
        "ts": op_start, "dur": op_stop - op_start, "pid": pid, "tid": tid,
        "args": {
            "proxy_id": proxy_id, "direction": direction,
            "channel_id": channel_id, "peer": peer,
            "n_steps": proxy.get("proxy_n_steps"),
            "duration_us": proxy.get("proxy_duration_us"),
            "step_count": proxy.get("proxy_step_count"),
        },
    })
    for state in proxy.get("proxy_states", []):
        trace_events.append({
            "name": state["state"], "cat": "state", "ph": "i", "s": "t",
            "ts": state["state_ts"], "pid": pid, "tid": tid,
            "args": {"state_id": state["state_id"], "proxy_id": proxy_id},
        })
    steps = proxy.get("proxy_step_records", None)
    if steps is None:
        steps = [e for e in proxy.get("events", []) if e.get("event_type") == "ProxyStep"]
    for step_ev in steps:
        for state in step_ev.get("proxy_states", []):
            a = {"state_id": state["state_id"], "proxy_id": proxy_id,
                 "step": step_ev["proxy_step"]}
            if "trans_size" in state:
                a["trans_size"] = state["trans_size"]
            trace_events.append({
                "name": state["state"], "cat": "state", "ph": "i", "s": "t",
                "ts": state["state_ts"], "pid": pid, "tid": tid, "args": a,
            })


def convert_simple_event(event_type, ev, trace_events, pid, tids):
    """Convert a simple event (coll_api, kernel_launch, etc.) to an X span."""
    start_ts = ev.get("start_ts", 0)
    stop_ts = ev.get("stop_ts", 0)
    dur = max(stop_ts - start_ts, 0)
    tid = tids.get(pid, event_type)

    name = event_type
    if "func" in ev:
        name = f"{event_type}: {ev['func']}"

    args = {}
    for k in ("rank", "func", "count", "datatype", "root",
              "group_depth", "graph_captured", "net_plugin_id", "duration_us"):
        if k in ev:
            args[k] = ev[k]

    trace_events.append({
        "name": name, "cat": event_type.lower(), "ph": "X",
        "ts": start_ts, "dur": dur, "pid": pid, "tid": tid, "args": args,
    })


def build_metadata(pid_map, tids):
    """Generate process_name and thread_name metadata events."""
    meta = []
    for (rank, hostname), pid in sorted(pid_map.items(), key=lambda x: x[1]):
        meta.append({"name": "process_name", "ph": "M", "pid": pid,
                     "args": {"name": f"Rank {rank} ({hostname})"}})
        meta.append({"name": "process_sort_index", "ph": "M", "pid": pid,
                     "args": {"sort_index": rank}})
        for tname, tid_int in tids.items(pid):
            meta.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid_int,
                         "args": {"name": tname}})
    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Convert NCCL Inspector profiler logs to Chrome Trace JSON."
    )
    parser.add_argument("inputs", nargs="+",
                        help="Input log file(s) or director(ies) containing .log files")
    parser.add_argument("-o", "--output", default="trace.json",
                        help="Output Chrome Trace JSON file (default: trace.json)")
    parser.add_argument("--max-records", type=int, default=0,
                        help="Max input records to process (0 = no limit)")
    args = parser.parse_args()

    log_files = collect_log_files(args.inputs)
    if not log_files:
        print("ERROR: no .log files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(log_files)} log file(s)...", file=sys.stderr)

    # Numeric pid allocation (offset by 1000 to avoid pid==tid collision
    # which causes Perfetto to show "main thread" label)
    pid_map = {}  # (rank, hostname) -> pid_int
    next_pid = [1000]

    def get_pid(rank, hostname):
        key = (rank, hostname)
        if key not in pid_map:
            pid_map[key] = next_pid[0]
            next_pid[0] += 1
        return pid_map[key]

    tids = TidAlloc()
    all_records = []  # (pid, rec)
    limit = args.max_records

    for log_file in log_files:
        for rec in parse_log_file(log_file):
            header = rec.get("header", {})
            metadata = rec.get("metadata", {})
            rank = header.get("rank", 0)
            hostname = metadata.get("hostname", "unknown")
            pid = get_pid(rank, hostname)
            all_records.append((pid, rec))
            if limit > 0 and len(all_records) >= limit:
                break
        if limit > 0 and len(all_records) >= limit:
            break

    if not all_records:
        print("ERROR: no valid records found.", file=sys.stderr)
        sys.exit(1)

    print(f"  Total records: {len(all_records)}", file=sys.stderr)
    print(f"  Processes: {len(pid_map)}", file=sys.stderr)

    # Convert
    trace_events = []
    for pid, rec in all_records:
        if "proxy_ctrl" in rec:
            convert_proxy_ctrl(rec["proxy_ctrl"], trace_events, pid, tids)
        elif "proxy_op" in rec:
            convert_standalone_proxy_op(rec["proxy_op"], trace_events, pid, tids)
        elif "coll_perf" in rec:
            convert_coll_perf(rec, trace_events, pid, tids)
        elif "p2p_perf" in rec:
            convert_p2p_perf(rec, trace_events, pid, tids)
        else:
            # Simple events: coll_api, group_api, group, kernel_launch, etc.
            for key, type_name in SIMPLE_EVENT_KEYS.items():
                if key in rec:
                    convert_simple_event(type_name, rec[key], trace_events, pid, tids)
                    break

    # Metadata
    meta = build_metadata(pid_map, tids)
    trace_events = meta + trace_events

    output = {"traceEvents": trace_events, "displayTimeUnit": "us"}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    sz = output_path.stat().st_size / (1024 * 1024)
    print(f"  Output: {output_path} ({sz:.2f} MB, {len(trace_events)} events)", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
