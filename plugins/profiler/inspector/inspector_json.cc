#include "inspector_json.h"
#include "profiler.h"

#include <unistd.h>
#include <vector>

static const char* inspectorSimpleEventTypeToKeyJson(uint64_t type) {
  switch (type) {
  case ncclProfileGroup:        return "group";
  case ncclProfileGroupApi:     return "group_api";
  case ncclProfileCollApi:      return "coll_api";
  case ncclProfileP2pApi:       return "p2p_api";
  case ncclProfileKernelLaunch: return "kernel_launch";
  case ncclProfileNetPlugin:    return "net_plugin";
  default:                      return "unknown";
  }
}

static const char* inspectorProxyEventTypeToKeyJson(inspectorProxyEventType_t type) {
  switch (type) {
  case inspectorProxyEventTypeOp:   return "proxy_op";
  case inspectorProxyEventTypeStep: return "proxy_step";
  case inspectorProxyEventTypeCtrl: return "proxy_ctrl";
  default:                          return "unknown";
  }
}

#define JSON_CHK(expr)                                          \
  do {                                                          \
    const jsonResult_t res = (expr);                            \
    if (res != jsonSuccess) {                                   \
      INFO_INSPECTOR("jsonError: %s\n", jsonErrorString(res));  \
      return inspectorJsonError;                                \
    }                                                           \
  } while (0)

#define JSON_CHK_GOTO(expr, res, label)                                 \
  do {                                                                  \
    const jsonResult_t macro_res = (expr);                              \
    if (macro_res != jsonSuccess) {                                     \
      INFO_INSPECTOR("jsonError: %s\n", jsonErrorString(macro_res));    \
      res = inspectorJsonError;                                         \
      goto label;                                                       \
      }                                                                   \
    } while (0)

static void inspectorCompletedOpVectorCleanup(std::vector<inspectorCompletedOpInfo>& ops) {
  ops.clear();
}

static inline inspectorResult_t inspectorCompletedOpChildEventList(
    jsonFileOutput* jfo,
    const struct inspectorCompletedOpInfo* op);

static inspectorResult_t inspectorCommInfoHeader(jsonFileOutput* jfo,
                                                 struct inspectorCommInfo* commInfo) {
  JSON_CHK(jsonStartObject(jfo));
  JSON_CHK(jsonKey(jfo, "id")); JSON_CHK(jsonStr(jfo, commInfo->commHashStr));
  const char* commName
    = (commInfo->commName && commInfo->commName[0]) ? commInfo->commName : "unknown";
  JSON_CHK(jsonKey(jfo, "comm_name")); JSON_CHK(jsonStr(jfo, commName));
  JSON_CHK(jsonKey(jfo, "rank")); JSON_CHK(jsonInt(jfo, commInfo->rank));
  JSON_CHK(jsonKey(jfo, "n_ranks")); JSON_CHK(jsonInt(jfo, commInfo->nranks));
  JSON_CHK(jsonKey(jfo, "nnodes")); JSON_CHK(jsonUint64(jfo, commInfo->nnodes));
  JSON_CHK(jsonFinishObject(jfo));
  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Writes metadata header information to the JSON output.
 *
 * Thread Safety:
 *   Not thread-safe (should be called with proper locking).
 *
 * Input:
 *   jsonFileOutput* jfo - JSON output handle.
 *
 * Output:
 *   Metadata header is written to JSON output.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
static inspectorResult_t inspectorCommInfoMetaHeader(jsonFileOutput* jfo) {
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "inspector_output_format_version")); JSON_CHK(jsonStr(jfo, "v4.1"));
    JSON_CHK(jsonKey(jfo, "git_rev")); JSON_CHK(jsonStr(jfo, get_git_version_info()));
    JSON_CHK(jsonKey(jfo, "rec_mechanism")); JSON_CHK(jsonStr(jfo, "nccl_profiler_interface"));
    JSON_CHK(jsonKey(jfo, "dump_timestamp_us")); JSON_CHK(jsonUint64(jfo, inspectorGetTime()));
    char hostname[256];
    gethostname(hostname, 255);
    JSON_CHK(jsonKey(jfo, "hostname")); JSON_CHK(jsonStr(jfo, hostname));
    JSON_CHK(jsonKey(jfo, "pid")); JSON_CHK(jsonUint64(jfo, getpid()));
  }
  JSON_CHK(jsonFinishObject(jfo));
  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Writes verbose information (event_trace) for a completed
 *   collective operation to the JSON output.
 *
 * Thread Safety:
 *   Not thread-safe (should be called with proper locking).
 *
 * Input:
 *   jsonFileOutput* jfo - JSON output handle.
 *   const struct inspectorCompletedOpInfo* op - completed collective info.
 *
 * Output:
 *   Verbose collective info is written to JSON output.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
static inline inspectorResult_t inspectorCompletedCollVerbose(jsonFileOutput* jfo,
                                                              const struct inspectorCompletedOpInfo* op) {
  JSON_CHK(jsonKey(jfo, "event_trace_sn"));
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "coll_start_sn")); JSON_CHK(jsonUint64(jfo, op->evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_START].sn));
    JSON_CHK(jsonKey(jfo, "coll_stop_sn")); JSON_CHK(jsonUint64(jfo, op->evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_STOP].sn));
  }
  JSON_CHK(jsonFinishObject(jfo));

  JSON_CHK(jsonKey(jfo, "event_trace_ts"));
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "coll_start_ts")); JSON_CHK(jsonUint64(jfo, op->evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_START].ts));
    JSON_CHK(jsonKey(jfo, "coll_stop_ts")); JSON_CHK(jsonUint64(jfo, op->evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_STOP].ts));
  }
  JSON_CHK(jsonFinishObject(jfo));

  JSON_CHK(jsonKey(jfo, "events"));
  INS_CHK(inspectorCompletedOpChildEventList(jfo, op));

  return inspectorSuccess;
}

static inline inspectorResult_t inspectorCompletedP2pVerbose(jsonFileOutput* jfo,
                                                             const struct inspectorCompletedOpInfo* op) {
  JSON_CHK(jsonKey(jfo, "event_trace_sn"));
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "p2p_start_sn")); JSON_CHK(jsonUint64(jfo, op->evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_START].sn));
    JSON_CHK(jsonKey(jfo, "p2p_stop_sn")); JSON_CHK(jsonUint64(jfo, op->evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_STOP].sn));
  }
  JSON_CHK(jsonFinishObject(jfo));

  JSON_CHK(jsonKey(jfo, "event_trace_ts"));
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "p2p_start_ts")); JSON_CHK(jsonUint64(jfo, op->evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_START].ts));
    JSON_CHK(jsonKey(jfo, "p2p_stop_ts")); JSON_CHK(jsonUint64(jfo, op->evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_STOP].ts));
  }
  JSON_CHK(jsonFinishObject(jfo));

  JSON_CHK(jsonKey(jfo, "events"));
  INS_CHK(inspectorCompletedOpChildEventList(jfo, op));

  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Writes completed collective operation information to the JSON
 *   output.
 *
 * Thread Safety:
 *   Not thread-safe (should be called with proper locking).
 *
 * Input:
 *   jsonFileOutput* jfo - JSON output handle.
 *   const struct inspectorCompletedOpInfo* op - completed collective info.
 *
 * Output:
 *   Collective info is written to JSON output.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
static inline inspectorResult_t inspectorCompletedColl(jsonFileOutput* jfo,
                                                       const struct inspectorCompletedOpInfo* op) {
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "coll")); JSON_CHK(jsonStr(jfo, ncclFuncToString(op->func)));

    JSON_CHK(jsonKey(jfo, "coll_sn")); JSON_CHK(jsonUint64(jfo, op->sn));

    JSON_CHK(jsonKey(jfo, "coll_msg_size_bytes")); JSON_CHK(jsonUint64(jfo, op->msgSizeBytes));

    JSON_CHK(jsonKey(jfo, "coll_exec_time_us")); JSON_CHK(jsonUint64(jfo, op->execTimeUsecs));

    JSON_CHK(jsonKey(jfo, "coll_timing_source")); JSON_CHK(jsonStr(jfo, inspectorTimingSourceToString(op->timingSource)));

    JSON_CHK(jsonKey(jfo, "coll_algobw_gbs")); JSON_CHK(jsonDouble(jfo, op->algoBwGbs));

    JSON_CHK(jsonKey(jfo, "coll_busbw_gbs")); JSON_CHK(jsonDouble(jfo, op->busBwGbs));

    if (inspectorIsDumpVerboseEnabled()) {
      INS_CHK(inspectorCompletedCollVerbose(jfo, op));
    }
  }
  JSON_CHK(jsonFinishObject(jfo));

  return inspectorSuccess;
}

static inline inspectorResult_t inspectorCompletedP2p(jsonFileOutput* jfo,
                                                      const struct inspectorCompletedOpInfo* op) {
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "p2p")); JSON_CHK(jsonStr(jfo, ncclFuncToString(op->func)));

    JSON_CHK(jsonKey(jfo, "p2p_sn")); JSON_CHK(jsonUint64(jfo, op->sn));

    JSON_CHK(jsonKey(jfo, "p2p_peer")); JSON_CHK(jsonInt(jfo, op->peer));

    JSON_CHK(jsonKey(jfo, "p2p_msg_size_bytes")); JSON_CHK(jsonUint64(jfo, op->msgSizeBytes));

    JSON_CHK(jsonKey(jfo, "p2p_exec_time_us")); JSON_CHK(jsonUint64(jfo, op->execTimeUsecs));

    JSON_CHK(jsonKey(jfo, "p2p_timing_source")); JSON_CHK(jsonStr(jfo, inspectorTimingSourceToString(op->timingSource)));

    JSON_CHK(jsonKey(jfo, "p2p_algobw_gbs")); JSON_CHK(jsonDouble(jfo, op->algoBwGbs));

    JSON_CHK(jsonKey(jfo, "p2p_busbw_gbs")); JSON_CHK(jsonDouble(jfo, op->busBwGbs));

    if (inspectorIsDumpVerboseEnabled()) {
      INS_CHK(inspectorCompletedP2pVerbose(jfo, op));
    }
  }
  JSON_CHK(jsonFinishObject(jfo));

  return inspectorSuccess;
}

static const char* inspectorProxyEventTypeToString(inspectorProxyEventType_t type) {
  switch (type) {
  case inspectorProxyEventTypeOp: return "ProxyOp";
  case inspectorProxyEventTypeStep: return "ProxyStep";
  case inspectorProxyEventTypeCtrl: return "ProxyCtrl";
  default: return "Unknown";
  }
}

static const char* inspectorProxyEventStateToString(int state) {
  switch ((ncclProfilerEventState_t)state) {
  case ncclProfilerProxyOpSendPosted: return "ProxyOpSendPosted";
  case ncclProfilerProxyOpSendRemFifoWait: return "ProxyOpSendRemFifoWait";
  case ncclProfilerProxyOpSendTransmitted: return "ProxyOpSendTransmitted";
  case ncclProfilerProxyOpSendDone: return "ProxyOpSendDone";
  case ncclProfilerProxyOpRecvPosted: return "ProxyOpRecvPosted";
  case ncclProfilerProxyOpRecvReceived: return "ProxyOpRecvReceived";
  case ncclProfilerProxyOpRecvTransmitted: return "ProxyOpRecvTransmitted";
  case ncclProfilerProxyOpRecvDone: return "ProxyOpRecvDone";
  case ncclProfilerProxyOpInProgress_v4: return "ProxyOpInProgress";
  case ncclProfilerProxyStepSendGPUWait: return "ProxyStepSendGPUWait";
  case ncclProfilerProxyStepSendPeerWait_v4: return "ProxyStepSendPeerWait";
  case ncclProfilerProxyStepSendWait: return "ProxyStepSendWait";
  case ncclProfilerProxyStepRecvWait: return "ProxyStepRecvWait";
  case ncclProfilerProxyStepRecvFlushWait: return "ProxyStepRecvFlushWait";
  case ncclProfilerProxyStepRecvGPUWait: return "ProxyStepRecvGPUWait";
  case ncclProfilerProxyCtrlIdle: return "ProxyCtrlIdle";
  case ncclProfilerProxyCtrlActive: return "ProxyCtrlActive";
  case ncclProfilerProxyCtrlSleep: return "ProxyCtrlSleep";
  case ncclProfilerProxyCtrlWakeup: return "ProxyCtrlWakeup";
  case ncclProfilerProxyCtrlAppend: return "ProxyCtrlAppend";
  case ncclProfilerProxyCtrlAppendEnd: return "ProxyCtrlAppendEnd";
  default: return "Unknown";
  }
}

static const char* inspectorProxyDirectionToString(int isSend) {
  if (isSend == 1) return "Send";
  if (isSend == 0) return "Recv";
  return "Unknown";
}

static inline inspectorResult_t inspectorProxyStateList(jsonFileOutput* jfo,
                                                        const inspectorProxyEventStateList* states) {
  JSON_CHK(jsonStartList(jfo));
  uint32_t count = states ? static_cast<uint32_t>(states->size()) : 0;
  for (uint32_t i = 0; i < count; i++) {
    const struct inspectorProxyEventStateInfo* state = &(*states)[i];
    if (state == nullptr) return inspectorMemoryError;

    JSON_CHK(jsonStartObject(jfo));
    JSON_CHK(jsonKey(jfo, "state")); JSON_CHK(jsonStr(jfo, inspectorProxyEventStateToString(state->state)));
    JSON_CHK(jsonKey(jfo, "state_id")); JSON_CHK(jsonInt(jfo, state->state));
    JSON_CHK(jsonKey(jfo, "state_ts")); JSON_CHK(jsonUint64(jfo, state->tsUsec));
    if (state->transSize != 0) {
      JSON_CHK(jsonKey(jfo, "trans_size")); JSON_CHK(jsonUint64(jfo, state->transSize));
    }
    if (state->appendedProxyOps != 0) {
      JSON_CHK(jsonKey(jfo, "appended_proxy_ops")); JSON_CHK(jsonInt(jfo, state->appendedProxyOps));
    }
    JSON_CHK(jsonFinishObject(jfo));
  }
  JSON_CHK(jsonFinishList(jfo));
  return inspectorSuccess;
}

static inline inspectorResult_t inspectorProxyStepRecord(jsonFileOutput* jfo,
                                                         const struct inspectorProxyStepRecordInfo* step) {
  if (step == nullptr) return inspectorMemoryError;

  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "event_type")); JSON_CHK(jsonStr(jfo, "ProxyStep"));
    JSON_CHK(jsonKey(jfo, "proxy_step")); JSON_CHK(jsonInt(jfo, step->step));
    JSON_CHK(jsonKey(jfo, "proxy_start_ts")); JSON_CHK(jsonUint64(jfo, step->tsStartUsec));
    JSON_CHK(jsonKey(jfo, "proxy_stop_ts")); JSON_CHK(jsonUint64(jfo, step->tsCompletedUsec));
    uint64_t durationUsec =
      (step->tsCompletedUsec >= step->tsStartUsec) ?
      (step->tsCompletedUsec - step->tsStartUsec) : 0;
    JSON_CHK(jsonKey(jfo, "proxy_duration_us")); JSON_CHK(jsonUint64(jfo, durationUsec));
    JSON_CHK(jsonKey(jfo, "proxy_trans_size")); JSON_CHK(jsonUint64(jfo, step->transSize));
    JSON_CHK(jsonKey(jfo, "proxy_states"));
    INS_CHK(inspectorProxyStateList(jfo, &step->states));
  }
  JSON_CHK(jsonFinishObject(jfo));
  return inspectorSuccess;
}

static inline inspectorResult_t inspectorProxyStepList(jsonFileOutput* jfo,
                                                       const inspectorProxyStepRecordList* steps) {
  JSON_CHK(jsonStartList(jfo));
  uint32_t count = steps ? static_cast<uint32_t>(steps->size()) : 0;
  for (uint32_t i = 0; i < count; i++) {
    const struct inspectorProxyStepRecordInfo* step = &(*steps)[i];
    INS_CHK(inspectorProxyStepRecord(jfo, step));
  }
  JSON_CHK(jsonFinishList(jfo));
  return inspectorSuccess;
}

static inline inspectorResult_t inspectorCompletedProxy(jsonFileOutput* jfo,
                                                        const struct inspectorCompletedProxyEventInfo* proxy) {
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "event_type")); JSON_CHK(jsonStr(jfo, inspectorProxyEventTypeToString(proxy->proxyType)));

    JSON_CHK(jsonKey(jfo, "proxy_id")); JSON_CHK(jsonUint64(jfo, proxy->proxyId));

    JSON_CHK(jsonKey(jfo, "proxy_rank")); JSON_CHK(jsonInt(jfo, proxy->rank));

    JSON_CHK(jsonKey(jfo, "proxy_pid")); JSON_CHK(jsonInt(jfo, proxy->pid));

    JSON_CHK(jsonKey(jfo, "proxy_start_ts")); JSON_CHK(jsonUint64(jfo, proxy->tsStartUsec));

    JSON_CHK(jsonKey(jfo, "proxy_stop_ts")); JSON_CHK(jsonUint64(jfo, proxy->tsCompletedUsec));

    uint64_t durationUsec =
      (proxy->tsCompletedUsec >= proxy->tsStartUsec) ?
      (proxy->tsCompletedUsec - proxy->tsStartUsec) : 0;
    JSON_CHK(jsonKey(jfo, "proxy_duration_us")); JSON_CHK(jsonUint64(jfo, durationUsec));

    if (proxy->proxyType == inspectorProxyEventTypeOp ||
        proxy->proxyType == inspectorProxyEventTypeStep) {
      JSON_CHK(jsonKey(jfo, "proxy_direction")); JSON_CHK(jsonStr(jfo, inspectorProxyDirectionToString(proxy->isSend)));
      JSON_CHK(jsonKey(jfo, "proxy_channel_id")); JSON_CHK(jsonInt(jfo, proxy->channelId));
      JSON_CHK(jsonKey(jfo, "proxy_peer")); JSON_CHK(jsonInt(jfo, proxy->peer));
      JSON_CHK(jsonKey(jfo, "proxy_n_steps")); JSON_CHK(jsonInt(jfo, proxy->nSteps));
      JSON_CHK(jsonKey(jfo, "proxy_chunk_size")); JSON_CHK(jsonInt(jfo, proxy->chunkSize));
      JSON_CHK(jsonKey(jfo, "proxy_trans_size")); JSON_CHK(jsonUint64(jfo, proxy->transSize));
    }

    if (proxy->proxyType == inspectorProxyEventTypeStep) {
      JSON_CHK(jsonKey(jfo, "proxy_step")); JSON_CHK(jsonInt(jfo, proxy->step));
    }

    JSON_CHK(jsonKey(jfo, "proxy_states"));
    INS_CHK(inspectorProxyStateList(jfo, &proxy->states));

    if (proxy->proxyType == inspectorProxyEventTypeOp) {
      JSON_CHK(jsonKey(jfo, "proxy_step_count"));
      JSON_CHK(jsonUint64(jfo, proxy->steps.size()));
      JSON_CHK(jsonKey(jfo, "proxy_step_records"));
      INS_CHK(inspectorProxyStepList(jfo, &proxy->steps));
    }
  }
  JSON_CHK(jsonFinishObject(jfo));

  return inspectorSuccess;
}

static inline inspectorResult_t inspectorCompletedKernelCh(jsonFileOutput* jfo,
                                                           const struct inspectorCompletedOpInfo* op,
                                                           uint32_t ch) {
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "event_type")); JSON_CHK(jsonStr(jfo, "KernelCh"));
    JSON_CHK(jsonKey(jfo, "channel_id")); JSON_CHK(jsonInt(jfo, ch));

    JSON_CHK(jsonKey(jfo, "event_trace_sn"));
    JSON_CHK(jsonStartObject(jfo));
    {
      JSON_CHK(jsonKey(jfo, "kernel_start_sn")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_START].sn));
      JSON_CHK(jsonKey(jfo, "kernel_stop_sn")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_STOP].sn));
      JSON_CHK(jsonKey(jfo, "kernel_record_sn")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_RECORD].sn));
    }
    JSON_CHK(jsonFinishObject(jfo));

    JSON_CHK(jsonKey(jfo, "event_trace_ts"));
    JSON_CHK(jsonStartObject(jfo));
    {
      JSON_CHK(jsonKey(jfo, "kernel_start_ts")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_START].ts));
      JSON_CHK(jsonKey(jfo, "kernel_stop_ts")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_STOP].ts));
      JSON_CHK(jsonKey(jfo, "kernel_record_ts")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_RECORD].ts));
    }
    JSON_CHK(jsonFinishObject(jfo));
  }
  JSON_CHK(jsonFinishObject(jfo));
  return inspectorSuccess;
}

static inline inspectorResult_t inspectorCompletedOpChildEventList(
    jsonFileOutput* jfo,
    const struct inspectorCompletedOpInfo* op) {
  JSON_CHK(jsonStartList(jfo));
  for (uint32_t ch = 0; ch < op->evtTrk.nChannels; ch++) {
    INS_CHK(inspectorCompletedKernelCh(jfo, op, ch));
  }

  uint32_t count = static_cast<uint32_t>(op->proxyOps.size());
  for (uint32_t i = 0; i < count; i++) {
    const struct inspectorCompletedProxyEventInfo* proxy = &op->proxyOps[i];
    if (proxy == nullptr) return inspectorMemoryError;
    INS_CHK(inspectorCompletedProxy(jfo, proxy));
  }
  JSON_CHK(jsonFinishList(jfo));
  return inspectorSuccess;
}


/*
 * Description:
 *
 *   Dumps the state of a communicator to the JSON output if needed.
 *
 * Thread Safety:
 *   Not thread-safe (should be called with proper locking).
 *
 * Input:
 *   jsonFileOutput* jfo - JSON output handle.
 *   inspectorCommInfo* commInfo - communicator info.
 *   bool* needs_writing - set to true if output was written.
 *
 * Output:
 *   State is dumped to JSON output if needed.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
static inspectorResult_t inspectorCommInfoDumpColl(jsonFileOutput* jfo,
                                                   inspectorCommInfo* commInfo,
                                                   bool* needs_writing) {
  if (commInfo == nullptr) {
    return inspectorSuccess;
  }

  thread_local std::vector<inspectorCompletedOpInfo> drainedColl;
  inspectorCompletedOpVectorCleanup(drainedColl);

  inspectorLockWr(&commInfo->guard);
  if (commInfo->dump_coll) {
    drainedColl = commInfo->completedCollRing.drain();
    commInfo->dump_coll = commInfo->completedCollRing.nonEmpty();
  }
  inspectorUnlockRWLock(&commInfo->guard);

  if (!drainedColl.empty()) {
    *needs_writing = true;
    JSON_CHK(jsonLockOutput(jfo));
    for (size_t i = 0; i < drainedColl.size(); i++) {
      JSON_CHK(jsonStartObject(jfo));
      {
        JSON_CHK(jsonKey(jfo, "header"));
        inspectorCommInfoHeader(jfo, commInfo);

        JSON_CHK(jsonKey(jfo, "metadata"));
        inspectorCommInfoMetaHeader(jfo);

        JSON_CHK(jsonKey(jfo, "coll_perf"));
        INS_CHK(inspectorCompletedColl(jfo, &drainedColl[i]));
      }
      JSON_CHK(jsonFinishObject(jfo));
      JSON_CHK(jsonNewline(jfo));
    }
    JSON_CHK(jsonUnlockOutput(jfo));
    inspectorCompletedOpVectorCleanup(drainedColl);
  }
  return inspectorSuccess;
}

static inspectorResult_t inspectorCommInfoDumpP2p(jsonFileOutput* jfo,
                                                  inspectorCommInfo* commInfo,
                                                  bool* needs_writing) {
  if (commInfo == nullptr) {
    return inspectorSuccess;
  }

  thread_local std::vector<inspectorCompletedOpInfo> drainedP2p;
  inspectorCompletedOpVectorCleanup(drainedP2p);

  inspectorLockWr(&commInfo->guard);
  if (commInfo->dump_p2p) {
    drainedP2p = commInfo->completedP2pRing.drain();
    commInfo->dump_p2p = commInfo->completedP2pRing.nonEmpty();
  }
  inspectorUnlockRWLock(&commInfo->guard);

  if (!drainedP2p.empty()) {
    *needs_writing = true;
    JSON_CHK(jsonLockOutput(jfo));
    for (size_t i = 0; i < drainedP2p.size(); i++) {
      JSON_CHK(jsonStartObject(jfo));
      {
        JSON_CHK(jsonKey(jfo, "header"));
        inspectorCommInfoHeader(jfo, commInfo);

        JSON_CHK(jsonKey(jfo, "metadata"));
        inspectorCommInfoMetaHeader(jfo);

        JSON_CHK(jsonKey(jfo, "p2p_perf"));
        INS_CHK(inspectorCompletedP2p(jfo, &drainedP2p[i]));
      }
      JSON_CHK(jsonFinishObject(jfo));
      JSON_CHK(jsonNewline(jfo));
    }
    JSON_CHK(jsonUnlockOutput(jfo));
    inspectorCompletedOpVectorCleanup(drainedP2p);
  }
  return inspectorSuccess;
}

static inspectorResult_t inspectorCommInfoDumpProxy(jsonFileOutput* jfo,
                                                    inspectorCommInfo* commInfo,
                                                    bool* needs_writing) {
  if (commInfo == nullptr) {
    return inspectorSuccess;
  }
  if (!inspectorIsDumpVerboseEnabled()) {
    return inspectorSuccess;
  }

  std::vector<inspectorCompletedProxyEventInfo> drainedProxy;
  inspectorResult_t res = inspectorSuccess;
  bool outputLocked = false;

  inspectorLockWr(&commInfo->guard);
  if (commInfo->dump_proxy) {
    drainedProxy = commInfo->completedProxyRing.drain();
    commInfo->dump_proxy = commInfo->completedProxyRing.nonEmpty();
  }
  inspectorUnlockRWLock(&commInfo->guard);
  if (res != inspectorSuccess) goto cleanup;

  if (!drainedProxy.empty()) {
    *needs_writing = true;
    JSON_CHK_GOTO(jsonLockOutput(jfo), res, cleanup);
    outputLocked = true;
    for (size_t i = 0; i < drainedProxy.size(); i++) {
      JSON_CHK_GOTO(jsonStartObject(jfo), res, cleanup);
      {
        JSON_CHK_GOTO(jsonKey(jfo, "header"), res, cleanup);
        INS_CHK_GOTO(inspectorCommInfoHeader(jfo, commInfo), res, cleanup);

        JSON_CHK_GOTO(jsonKey(jfo, "metadata"), res, cleanup);
        INS_CHK_GOTO(inspectorCommInfoMetaHeader(jfo), res, cleanup);

        JSON_CHK_GOTO(jsonKey(jfo, inspectorProxyEventTypeToKeyJson(drainedProxy[i].proxyType)), res, cleanup);
        INS_CHK_GOTO(inspectorCompletedProxy(jfo, &drainedProxy[i]), res, cleanup);
      }
      JSON_CHK_GOTO(jsonFinishObject(jfo), res, cleanup);
      JSON_CHK_GOTO(jsonNewline(jfo), res, cleanup);
    }
    JSON_CHK_GOTO(jsonUnlockOutput(jfo), res, cleanup);
    outputLocked = false;
  }

cleanup:
  if (outputLocked) {
    jsonUnlockOutput(jfo);
  }
  // std::vector destructor handles cleanup
  return res;
}

static inspectorResult_t inspectorCommInfoDumpSimple(jsonFileOutput* jfo,
                                                      inspectorCommInfo* commInfo,
                                                      bool* needs_writing) {
  if (commInfo == nullptr) {
    return inspectorSuccess;
  }

  std::vector<inspectorCompletedSimpleEventInfo> drainedSimple;

  inspectorLockWr(&commInfo->guard);
  if (commInfo->dump_simple) {
    drainedSimple = commInfo->completedSimpleRing.drain();
    commInfo->dump_simple = commInfo->completedSimpleRing.nonEmpty();
  }
  inspectorUnlockRWLock(&commInfo->guard);

  if (!drainedSimple.empty()) {
    *needs_writing = true;
    JSON_CHK(jsonLockOutput(jfo));
    for (size_t i = 0; i < drainedSimple.size(); i++) {
      const struct inspectorCompletedSimpleEventInfo* evt = &drainedSimple[i];
      JSON_CHK(jsonStartObject(jfo));
      {
        JSON_CHK(jsonKey(jfo, "header"));
        inspectorCommInfoHeader(jfo, commInfo);

        JSON_CHK(jsonKey(jfo, "metadata"));
        inspectorCommInfoMetaHeader(jfo);

        JSON_CHK(jsonKey(jfo, inspectorSimpleEventTypeToKeyJson(evt->type)));
        JSON_CHK(jsonStartObject(jfo));
        {
          JSON_CHK(jsonKey(jfo, "start_ts"));
          JSON_CHK(jsonUint64(jfo, evt->tsStartUsec));

          JSON_CHK(jsonKey(jfo, "stop_ts"));
          JSON_CHK(jsonUint64(jfo, evt->tsStopUsec));

          uint64_t durationUsec =
            (evt->tsStopUsec >= evt->tsStartUsec) ?
            (evt->tsStopUsec - evt->tsStartUsec) : 0;
          JSON_CHK(jsonKey(jfo, "duration_us"));
          JSON_CHK(jsonUint64(jfo, durationUsec));

          JSON_CHK(jsonKey(jfo, "rank"));
          JSON_CHK(jsonInt(jfo, evt->rank));

          // Type-specific fields
          if (evt->type == ncclProfileCollApi || evt->type == ncclProfileP2pApi) {
            if (evt->funcName[0] != '\0') {
              JSON_CHK(jsonKey(jfo, "func")); JSON_CHK(jsonStr(jfo, evt->funcName));
            }
            JSON_CHK(jsonKey(jfo, "count")); JSON_CHK(jsonUint64(jfo, evt->count));
            if (evt->datatype[0] != '\0') {
              JSON_CHK(jsonKey(jfo, "datatype")); JSON_CHK(jsonStr(jfo, evt->datatype));
            }
            JSON_CHK(jsonKey(jfo, "graph_captured")); JSON_CHK(jsonBool(jfo, evt->graphCaptured));
          }
          if (evt->type == ncclProfileCollApi) {
            JSON_CHK(jsonKey(jfo, "root")); JSON_CHK(jsonInt(jfo, evt->root));
          }
          if (evt->type == ncclProfileGroupApi) {
            JSON_CHK(jsonKey(jfo, "group_depth")); JSON_CHK(jsonInt(jfo, evt->groupDepth));
            JSON_CHK(jsonKey(jfo, "graph_captured")); JSON_CHK(jsonBool(jfo, evt->graphCaptured));
          }
          if (evt->type == ncclProfileNetPlugin) {
            JSON_CHK(jsonKey(jfo, "net_plugin_id")); JSON_CHK(jsonUint64(jfo, (uint64_t)evt->netPluginId));
          }
        }
        JSON_CHK(jsonFinishObject(jfo));
      }
      JSON_CHK(jsonFinishObject(jfo));
      JSON_CHK(jsonNewline(jfo));
    }
    JSON_CHK(jsonUnlockOutput(jfo));
  }
  return inspectorSuccess;
}

static inspectorResult_t inspectorCommInfoDump(jsonFileOutput* jfo,
                                               inspectorCommInfo* commInfo,
                                               bool* needs_writing) {
  *needs_writing = false;

  INS_CHK(inspectorCommInfoDumpColl(jfo, commInfo, needs_writing));
  INS_CHK(inspectorCommInfoDumpP2p(jfo, commInfo, needs_writing));
  INS_CHK(inspectorCommInfoDumpProxy(jfo, commInfo, needs_writing));
  INS_CHK(inspectorCommInfoDumpSimple(jfo, commInfo, needs_writing));
  return inspectorSuccess;
}


/*
 * Description:
 *
 *   Dumps the state of all communicators in a commList to the JSON
 *   output.
 *
 * Thread Safety:
 *   Thread-safe - assumes no locks are taken and acquires all
 *   necessary locks to iterate through all communicator objects and
 *   dump their state.
 *
 * Input:
 *   jsonFileOutput* jfo - JSON output handle (must not be NULL).
 *   struct inspectorCommInfoList* commList - list of communicators
 *   (must not be NULL).
 *
 * Output:
 *   State of all communicators is dumped to JSON output.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
inspectorResult_t inspectorCommInfoListDump(jsonFileOutput* jfo,
                                            struct inspectorCommInfoList* commList) {
  bool flush = false;
  INS_CHK(inspectorLockRd(&commList->guard));
  inspectorResult_t res = inspectorSuccess;
  if (!commList->comms.empty()) {
    for (auto* itr : commList->comms) {
      bool needs_writing;
      INS_CHK_GOTO(inspectorCommInfoDump(jfo, itr, &needs_writing),
                   res, exit);
      if (needs_writing) {
        flush = true;
      }
    }
    if (flush) {
      JSON_CHK_GOTO(jsonLockOutput(jfo), res, exit);
      JSON_CHK_GOTO(jsonFlushOutput(jfo), res, exit);
      JSON_CHK_GOTO(jsonUnlockOutput(jfo), res, exit);
    }
  }
exit:
  INS_CHK(inspectorUnlockRWLock(&commList->guard));
  return res;
}
