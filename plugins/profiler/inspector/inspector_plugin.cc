/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <linux/limits.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "profiler.h"
#include "inspector.h"
#include "inspector_ring.h"
#include "inspector_event_pool.h"

#define __hidden __attribute__ ((visibility("hidden")))

static int gInitialized;

static pthread_mutex_t gLock = PTHREAD_MUTEX_INITIALIZER;


/*
 * Description:
 *   Records an event trace with timestamp and sequence number
 *
 * Thread Safety:
 *   Not thread-safe - must be called with proper locking. This function
 *   is designed to be called from within locked sections where the
 *   collective info structure is already protected.
 *
 * Input:
 *   struct inspectorEventTraceInfo* evtTrace - event trace array
 *   int eventIndex - index in the event trace array (must be valid)
 *   struct inspectorCollInfo* collInfo - collective info structure (must not be NULL)
 *
 * Output:
 *   Event trace is updated with current timestamp and next sequence
 *   number from collective
 *
 * Return:
 *   uint64_t - the sequence number assigned to this event
 *
 * Preconditions:
 *   - collInfo must not be NULL
 *   - eventIndex must be within valid bounds for evtTrace array
 *   - Function must be called from within a locked section
 */
static uint64_t inspectorRecordEventTrace(struct inspectorEventTraceInfo* evtTrace,
                                          int eventIndex,
                                          struct inspectorCollInfo* collInfo) {
  evtTrace[eventIndex].ts = inspectorGetTime();
  evtTrace[eventIndex].sn = ++collInfo->collEvtTrk.sn; // Increment coll sequence counter

  return evtTrace[eventIndex].sn;
}

/*
 * Description:
 *   Records an event trace with timestamp and sequence number for P2P operations
 */
static uint64_t inspectorRecordP2pEventTrace(struct inspectorEventTraceInfo* evtTrace,
                                             int eventIndex,
                                             struct inspectorP2pInfo* p2pInfo) {
  evtTrace[eventIndex].ts = inspectorGetTime();
  evtTrace[eventIndex].sn = ++p2pInfo->p2pEvtTrk.sn;
  return evtTrace[eventIndex].sn;
}

/*
 * Description:
 *
 *   Initializes the NCCL Inspector plugin and global state for a
 *   communicator.
 *
 * Thread Safety:
 *   Thread-safe (uses mutex for initialization).
 *
 * Input:
 *   void** context - pointer to plugin context.
 *   int* eActivationMask - pointer to activation mask output.
 *   const char* commName - communicator name.
 *   uint64_t commHash - communicator hash.
 *   int nNodes - number of nodes.
 *   int nranks - number of ranks.
 *   int rank - rank.
 *   ncclDebugLogger_t logfn - logger function pointer.
 *
 * Output:
 *   context is set to plugin context; eActivationMask is set.
 *
 * Return:
 *   ncclResult_t - success or error code.
 *
 */
__hidden ncclResult_t inspectorPluginInit(void** context, uint64_t commHash,
                                          int* eActivationMask,
                                          const char* commName,
                                          int nNodes, int nranks, int rank,
                                          ncclDebugLogger_t logfn) {
  inspectorResult_t res = inspectorSuccess;
  *context = nullptr;
  logFn = logfn;

  pthread_mutex_lock(&gLock);
  if (++gInitialized == 1) {
    res = inspectorGlobalInit(rank);
    if (res != inspectorSuccess) {
      INFO_INSPECTOR("Inspector Init Failed %s:%d -> error %d: %s",
                     __FILE__, __LINE__, res,
                     inspectorErrorString(res));
      gInitialized = 0;
      pthread_mutex_unlock(&gLock);
      return ncclSuccess;
    }
  }
  pthread_mutex_unlock(&gLock);

  res = inspectorAddComm((struct inspectorCommInfo **)context,
                         commName, commHash,
                         nNodes, nranks, rank);
  if (res != inspectorSuccess) {
    INFO_INSPECTOR("%s:%d -> error %d: %s", __FILE__, __LINE__, res,
                   inspectorErrorString(res));
    return ncclSuccess;
  }
  *eActivationMask = ncclProfileColl | ncclProfileKernelCh;
  if (enableNcclInspectorP2p) {
    *eActivationMask |= ncclProfileP2p;
  }
  if (enableNcclInspectorProxy && inspectorIsDumpVerboseEnabled()) {
    *eActivationMask |= ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileProxyCtrl;
  }

  INFO(NCCL_INIT, "PROFILER/Plugin: init commName: %s commHash: %lu nranks: %d rank: %d",
       commName ? commName : "", commHash, nranks, rank);
  return ncclSuccess;
}

/*
 * Description:
 *
 *   Finalizes the NCCL Inspector plugin and global state for a
 *   communicator.
 *
 * Thread Safety:
 *   Thread-safe (uses mutex for finalization).
 *
 * Input:
 *   void* context - plugin context.
 *
 * Output:
 *   Plugin context is finalized and cleaned up.
 *
 * Return:
 *   ncclResult_t - success or error code.
 *
 */
__hidden ncclResult_t inspectorPluginFinalize(void* context) {
  inspectorDelComm((struct inspectorCommInfo *)context);
  pthread_mutex_lock(&gLock);
  if (--gInitialized == 0) {
    inspectorGlobalFinalize();
  }
  pthread_mutex_unlock(&gLock);
  return ncclSuccess;
}

inspectorResult_t inspectorPluginCollInfoRef(struct inspectorCollInfo *collInfo) {
  collInfo->refCount += 1;
  return inspectorSuccess;
}

inspectorResult_t inspectorPluginCollInfoDeRef(struct inspectorCollInfo *collInfo) {
  collInfo->refCount -= 1;
  if (collInfo->refCount == 0) {
    return inspectorReturn;
  }
  return inspectorSuccess;
}

static void inspectorPluginCollInfoCleanup(struct inspectorCollInfo *collInfo) {
  inspectorLockDestroy(&collInfo->guard);
  inspectorProxyOpRecordListCleanup(&collInfo->proxyOps);
  inspectorEventPoolReleaseColl(collInfo);
}

static void inspectorUpdateCommOpInfo(struct inspectorCommInfo *commInfo,
                                      struct inspectorCompletedOpInfo *completedOp) {
  struct inspectorCompletedRing *ring =
    completedOp->isP2p ? &commInfo->completedP2pRing : &commInfo->completedCollRing;
  inspectorLockWr(&commInfo->guard);
  inspectorComputeOpBw(commInfo, completedOp);
  inspectorRingEnqueue(ring, completedOp);
  if (completedOp->isP2p) {
    commInfo->dump_p2p = inspectorRingNonEmpty(&commInfo->completedP2pRing);
  } else {
    commInfo->dump_coll = inspectorRingNonEmpty(&commInfo->completedCollRing);
  }
  inspectorUnlockRWLock(&commInfo->guard);
}

static inspectorResult_t inspectorUpdateCommProxyInfo(struct inspectorCommInfo *commInfo,
                                                      struct inspectorCompletedProxyEventInfo *completedProxy) {
  inspectorLockWr(&commInfo->guard);
  inspectorResult_t res = inspectorRingEnqueue(&commInfo->completedProxyRing, completedProxy);
  commInfo->dump_proxy = inspectorRingNonEmpty(&commInfo->completedProxyRing);
  inspectorUnlockRWLock(&commInfo->guard);
  return res;
}

inspectorResult_t inspectorPluginP2pInfoRef(struct inspectorP2pInfo *p2pInfo) {
  p2pInfo->refCount += 1;
  return inspectorSuccess;
}

inspectorResult_t inspectorPluginP2pInfoDeRef(struct inspectorP2pInfo *p2pInfo) {
  p2pInfo->refCount -= 1;
  if (p2pInfo->refCount == 0) {
    return inspectorReturn;
  }
  return inspectorSuccess;
}

static void inspectorPluginP2pInfoCleanup(struct inspectorP2pInfo *p2pInfo) {
  inspectorLockDestroy(&p2pInfo->guard);
  inspectorProxyOpRecordListCleanup(&p2pInfo->proxyOps);
  inspectorEventPoolReleaseP2p(p2pInfo);
}

static inspectorResult_t inspectorProxyEventStateListCopy(
    inspectorProxyEventStateList* dst,
    const inspectorProxyEventStateList* src) {
  return inspectorInlineListCopy(dst, src) ? inspectorSuccess : inspectorMemoryError;
}

static void inspectorProxyStepRecordInfoCleanup(struct inspectorProxyStepRecordInfo* step) {
  if (step == nullptr) return;
  inspectorInlineListFree(&step->states);
}

static inspectorResult_t inspectorProxyStepRecordInfoCopy(
    struct inspectorProxyStepRecordInfo* dst,
    const struct inspectorProxyStepRecordInfo* src) {
  if (dst == nullptr || src == nullptr) return inspectorMemoryError;

  memset(dst, 0, sizeof(*dst));
  dst->rank = src->rank;
  dst->pid = src->pid;
  dst->channelId = src->channelId;
  dst->peer = src->peer;
  dst->nSteps = src->nSteps;
  dst->chunkSize = src->chunkSize;
  dst->isSend = src->isSend;
  dst->step = src->step;
  dst->transSize = src->transSize;
  dst->tsStartUsec = src->tsStartUsec;
  dst->tsCompletedUsec = src->tsCompletedUsec;
  inspectorResult_t res = inspectorProxyEventStateListCopy(&dst->states, &src->states);
  if (res != inspectorSuccess) {
    inspectorProxyStepRecordInfoCleanup(dst);
    return res;
  }
  return inspectorSuccess;
}

static inspectorResult_t inspectorProxyStepRecordListAppend(
    inspectorProxyStepRecordList* list,
    const struct inspectorProxyStepRecordInfo* record) {
  if (list == nullptr || record == nullptr) return inspectorMemoryError;

  struct inspectorProxyStepRecordInfo tmp;
  inspectorResult_t res = inspectorProxyStepRecordInfoCopy(&tmp, record);
  if (res != inspectorSuccess) return res;

  struct inspectorProxyStepRecordInfo* dst = inspectorInlineListAppend(list);
  if (dst == nullptr) {
    inspectorProxyStepRecordInfoCleanup(&tmp);
    return inspectorMemoryError;
  }
  *dst = tmp;
  return inspectorSuccess;
}

static void inspectorProxyStepRecordListCleanup(inspectorProxyStepRecordList* steps) {
  if (steps == nullptr) return;
  for (uint32_t i = 0; i < steps->count; i++) {
    struct inspectorProxyStepRecordInfo* step =
      inspectorInlineListGetMutable(steps, i);
    inspectorProxyStepRecordInfoCleanup(step);
  }
  inspectorInlineListFree(steps);
}

static inspectorResult_t inspectorProxyStepRecordListCopy(
    inspectorProxyStepRecordList* dst,
    const inspectorProxyStepRecordList* src) {
  if (dst == nullptr || src == nullptr) return inspectorMemoryError;

  inspectorInlineListInit(dst);
  for (uint32_t i = 0; i < src->count; i++) {
    const struct inspectorProxyStepRecordInfo* srcStep =
      inspectorInlineListGet(src, i);
    if (srcStep == nullptr) {
      inspectorProxyStepRecordListCleanup(dst);
      return inspectorMemoryError;
    }
    inspectorResult_t res = inspectorProxyStepRecordListAppend(dst, srcStep);
    if (res != inspectorSuccess) {
      inspectorProxyStepRecordListCleanup(dst);
      return res;
    }
  }
  return inspectorSuccess;
}

void inspectorCompletedProxyEventInfoCleanup(void* entry) {
  struct inspectorCompletedProxyEventInfo* proxy =
    (struct inspectorCompletedProxyEventInfo*)entry;
  if (proxy == nullptr) return;

  inspectorInlineListFree(&proxy->states);
  inspectorProxyStepRecordListCleanup(&proxy->steps);
  memset(proxy, 0, sizeof(*proxy));
}

inspectorResult_t inspectorCompletedProxyEventInfoCopy(void* dst, const void* src) {
  if (dst == nullptr || src == nullptr) return inspectorMemoryError;

  struct inspectorCompletedProxyEventInfo* dstProxy =
    (struct inspectorCompletedProxyEventInfo*)dst;
  const struct inspectorCompletedProxyEventInfo* srcProxy =
    (const struct inspectorCompletedProxyEventInfo*)src;

  memset(dstProxy, 0, sizeof(*dstProxy));
  dstProxy->proxyType = srcProxy->proxyType;
  dstProxy->proxyId = srcProxy->proxyId;
  dstProxy->rank = srcProxy->rank;
  dstProxy->pid = srcProxy->pid;
  dstProxy->channelId = srcProxy->channelId;
  dstProxy->peer = srcProxy->peer;
  dstProxy->nSteps = srcProxy->nSteps;
  dstProxy->chunkSize = srcProxy->chunkSize;
  dstProxy->isSend = srcProxy->isSend;
  dstProxy->step = srcProxy->step;
  dstProxy->transSize = srcProxy->transSize;
  dstProxy->tsStartUsec = srcProxy->tsStartUsec;
  dstProxy->tsCompletedUsec = srcProxy->tsCompletedUsec;

  inspectorResult_t res = inspectorProxyEventStateListCopy(&dstProxy->states,
                                                           &srcProxy->states);
  if (res != inspectorSuccess) {
    inspectorCompletedProxyEventInfoCleanup(dstProxy);
    return res;
  }

  res = inspectorProxyStepRecordListCopy(&dstProxy->steps, &srcProxy->steps);
  if (res != inspectorSuccess) {
    inspectorCompletedProxyEventInfoCleanup(dstProxy);
    return res;
  }
  return inspectorSuccess;
}

inspectorResult_t inspectorProxyOpRecordListAppend(
    inspectorProxyOpRecordList* list,
    const struct inspectorCompletedProxyEventInfo* record) {
  if (list == nullptr || record == nullptr) return inspectorMemoryError;

  struct inspectorCompletedProxyEventInfo tmp;
  inspectorResult_t res = inspectorCompletedProxyEventInfoCopy(&tmp, record);
  if (res != inspectorSuccess) return res;

  struct inspectorCompletedProxyEventInfo* dst = inspectorInlineListAppend(list);
  if (dst == nullptr) {
    inspectorCompletedProxyEventInfoCleanup(&tmp);
    return inspectorMemoryError;
  }
  *dst = tmp;
  return inspectorSuccess;
}

void inspectorProxyOpRecordListCleanup(inspectorProxyOpRecordList* proxyOps) {
  if (proxyOps == nullptr) return;
  for (uint32_t i = 0; i < proxyOps->count; i++) {
    struct inspectorCompletedProxyEventInfo* proxy =
      inspectorInlineListGetMutable(proxyOps, i);
    inspectorCompletedProxyEventInfoCleanup(proxy);
  }
  inspectorInlineListFree(proxyOps);
}

inspectorResult_t inspectorProxyOpRecordListCopy(
    inspectorProxyOpRecordList* dst,
    const inspectorProxyOpRecordList* src) {
  if (dst == nullptr || src == nullptr) return inspectorMemoryError;

  inspectorInlineListInit(dst);
  for (uint32_t i = 0; i < src->count; i++) {
    const struct inspectorCompletedProxyEventInfo* srcProxy =
      inspectorInlineListGet(src, i);
    if (srcProxy == nullptr) {
      inspectorProxyOpRecordListCleanup(dst);
      return inspectorMemoryError;
    }
    inspectorResult_t res = inspectorProxyOpRecordListAppend(dst, srcProxy);
    if (res != inspectorSuccess) {
      inspectorProxyOpRecordListCleanup(dst);
      return res;
    }
  }
  return inspectorSuccess;
}

void inspectorCompletedOpInfoCleanup(void* entry) {
  struct inspectorCompletedOpInfo* op = (struct inspectorCompletedOpInfo*)entry;
  if (op == nullptr) return;

  inspectorProxyOpRecordListCleanup(&op->proxyOps);
  memset(op, 0, sizeof(*op));
}

inspectorResult_t inspectorCompletedOpInfoCopy(void* dst, const void* src) {
  if (dst == nullptr || src == nullptr) return inspectorMemoryError;

  struct inspectorCompletedOpInfo* dstOp = (struct inspectorCompletedOpInfo*)dst;
  const struct inspectorCompletedOpInfo* srcOp =
    (const struct inspectorCompletedOpInfo*)src;

  memset(dstOp, 0, sizeof(*dstOp));
  dstOp->isP2p = srcOp->isP2p;
  dstOp->func = srcOp->func;
  dstOp->sn = srcOp->sn;
  dstOp->msgSizeBytes = srcOp->msgSizeBytes;
  dstOp->execTimeUsecs = srcOp->execTimeUsecs;
  dstOp->timingSource = srcOp->timingSource;
  dstOp->algoBwGbs = srcOp->algoBwGbs;
  dstOp->busBwGbs = srcOp->busBwGbs;
  dstOp->algo = srcOp->algo;
  dstOp->proto = srcOp->proto;
  dstOp->peer = srcOp->peer;
  dstOp->evtTrk = srcOp->evtTrk;

  inspectorResult_t res = inspectorProxyOpRecordListCopy(&dstOp->proxyOps,
                                                         &srcOp->proxyOps);
  if (res != inspectorSuccess) {
    inspectorCompletedOpInfoCleanup(dstOp);
    return res;
  }
  return inspectorSuccess;
}

static void inspectorCompletedProxyEventInfoInit(
    struct inspectorCompletedProxyEventInfo* proxy) {
  if (proxy == nullptr) return;
  memset(proxy, 0, sizeof(*proxy));
  inspectorInlineListInit(&proxy->states);
  inspectorInlineListInit(&proxy->steps);
}

static inspectorResult_t inspectorRecordProxyEventState(inspectorProxyEventStateList* states,
                                           ncclProfilerEventState_t eState,
                                           ncclProfilerEventStateArgs_t* eStateArgs) {
  if (states == nullptr) return inspectorMemoryError;

  struct inspectorProxyEventStateInfo* state = inspectorInlineListAppend(states);
  if (state == nullptr) return inspectorMemoryError;
  state->state = (int)eState;
  state->tsUsec = inspectorGetTime();
  state->transSize = 0;
  state->appendedProxyOps = 0;

  switch (eState) {
  case ncclProfilerProxyStepSendWait:
  case ncclProfilerProxyStepRecvFlushWait:
    if (eStateArgs != nullptr) {
      state->transSize = eStateArgs->proxyStep.transSize;
    }
    break;
  case ncclProfilerProxyStepSendGPUWait:
  case ncclProfilerProxyStepSendPeerWait_v4:
  case ncclProfilerProxyStepRecvWait:
  case ncclProfilerProxyStepRecvGPUWait:
    break;
  case ncclProfilerProxyCtrlIdle:
  case ncclProfilerProxyCtrlActive:
  case ncclProfilerProxyCtrlSleep:
  case ncclProfilerProxyCtrlWakeup:
  case ncclProfilerProxyCtrlAppend:
  case ncclProfilerProxyCtrlAppendEnd:
    if (eStateArgs != nullptr) {
      state->appendedProxyOps = eStateArgs->proxyCtrl.appendedProxyOps;
    }
    break;
  default:
    break;
  }
  return inspectorSuccess;
}

static inspectorResult_t inspectorCompletedProxyFromOpLocked(
    struct inspectorCompletedProxyEventInfo *completedProxy,
    struct inspectorProxyOpInfo *opInfo) {
  inspectorCompletedProxyEventInfoInit(completedProxy);
  completedProxy->proxyType = inspectorProxyEventTypeOp;
  completedProxy->proxyId = opInfo->proxyId;
  completedProxy->rank = opInfo->rank;
  completedProxy->pid = opInfo->pid;
  completedProxy->channelId = opInfo->channelId;
  completedProxy->peer = opInfo->peer;
  completedProxy->nSteps = opInfo->nSteps;
  completedProxy->chunkSize = opInfo->chunkSize;
  completedProxy->isSend = opInfo->isSend;
  completedProxy->step = -1;
  completedProxy->transSize = opInfo->transSize;
  completedProxy->tsStartUsec = opInfo->tsStartUsec;
  completedProxy->tsCompletedUsec = opInfo->tsCompletedUsec;
  inspectorResult_t res = inspectorProxyEventStateListCopy(&completedProxy->states,
                                                           &opInfo->states);
  if (res != inspectorSuccess) return res;
  return inspectorProxyStepRecordListCopy(&completedProxy->steps, &opInfo->steps);
}

static inspectorResult_t inspectorCompletedProxyFromCtrlLocked(
    struct inspectorCompletedProxyEventInfo *completedProxy,
    struct inspectorProxyCtrlInfo *ctrlInfo) {
  inspectorCompletedProxyEventInfoInit(completedProxy);
  completedProxy->proxyType = inspectorProxyEventTypeCtrl;
  completedProxy->proxyId = ctrlInfo->proxyId;
  completedProxy->rank = ctrlInfo->commInfo ? ctrlInfo->commInfo->rank : -1;
  completedProxy->pid = getpid();
  completedProxy->channelId = 0;
  completedProxy->peer = -1;
  completedProxy->nSteps = 0;
  completedProxy->chunkSize = 0;
  completedProxy->isSend = -1;
  completedProxy->step = -1;
  completedProxy->transSize = 0;
  completedProxy->tsStartUsec = ctrlInfo->tsStartUsec;
  completedProxy->tsCompletedUsec = ctrlInfo->tsCompletedUsec;
  return inspectorProxyEventStateListCopy(&completedProxy->states,
                                          &ctrlInfo->states);
}

static void inspectorPluginProxyOpInfoCleanup(struct inspectorProxyOpInfo *opInfo) {
  inspectorLockDestroy(&opInfo->guard);
  inspectorInlineListFree(&opInfo->states);
  inspectorProxyStepRecordListCleanup(&opInfo->steps);
  free(opInfo);
}

static void inspectorPluginProxyStepInfoCleanup(struct inspectorProxyStepInfo *stepInfo) {
  inspectorLockDestroy(&stepInfo->guard);
  inspectorInlineListFree(&stepInfo->states);
  free(stepInfo);
}

static void inspectorPluginProxyCtrlInfoCleanup(struct inspectorProxyCtrlInfo *ctrlInfo) {
  inspectorLockDestroy(&ctrlInfo->guard);
  inspectorInlineListFree(&ctrlInfo->states);
  free(ctrlInfo);
}

static inspectorResult_t inspectorPluginProxyOpMaybeCompleteLocked(struct inspectorProxyOpInfo *opInfo,
                                                      struct inspectorCompletedProxyEventInfo *completedProxy,
                                                      struct inspectorCommInfo **commInfo,
                                                      bool *doCommUpdate,
                                                      bool *needsCleanup) {
  if (opInfo->stopped && opInfo->refCount == 0 && !opInfo->enqueued) {
    inspectorResult_t res = inspectorCompletedProxyFromOpLocked(completedProxy, opInfo);
    if (res != inspectorSuccess) return res;
    opInfo->enqueued = true;
    *commInfo = opInfo->commInfo;
    *doCommUpdate = (*commInfo != nullptr);
    *needsCleanup = true;
  }
  return inspectorSuccess;
}

static void inspectorPluginProxyOpInfoInit(struct inspectorProxyOpInfo **opInfo,
                                           ncclProfilerEventDescr_t *eDescr,
                                           struct inspectorCommInfo *commInfo) {
  struct inspectorProxyOpInfo *opInfoPtr =
    (struct inspectorProxyOpInfo*)calloc(1, sizeof(struct inspectorProxyOpInfo));
  if (opInfoPtr == nullptr) {
    INFO_INSPECTOR("Inspector: Failed to allocate memory for proxy op info structure");
    *opInfo = nullptr;
    return;
  }

  opInfoPtr->type = ncclProfileProxyOp;
  opInfoPtr->refCount = 1;
  opInfoPtr->commInfo = commInfo;
  opInfoPtr->parentType = 0;
  opInfoPtr->parentObj = nullptr;
  opInfoPtr->proxyId = __atomic_add_fetch(&commInfo->proxySeqNum, 1, __ATOMIC_RELAXED);
  opInfoPtr->rank = eDescr->rank;
  opInfoPtr->pid = eDescr->proxyOp.pid;
  opInfoPtr->channelId = eDescr->proxyOp.channelId;
  opInfoPtr->peer = eDescr->proxyOp.peer;
  opInfoPtr->nSteps = eDescr->proxyOp.nSteps;
  opInfoPtr->chunkSize = eDescr->proxyOp.chunkSize;
  opInfoPtr->isSend = eDescr->proxyOp.isSend;
  opInfoPtr->tsStartUsec = inspectorGetTime();
  inspectorInlineListInit(&opInfoPtr->states);
  inspectorInlineListInit(&opInfoPtr->steps);
  inspectorLockInit(&opInfoPtr->guard);

  if (eDescr->parentObj != nullptr && eDescr->proxyOp.pid == getpid()) {
    uint64_t parentType = *(uint64_t*)eDescr->parentObj;
    if (parentType == ncclProfileColl) {
      struct inspectorCollInfo* parent = (struct inspectorCollInfo*)eDescr->parentObj;
      inspectorLockWr(&parent->guard);
      parent->nProxyOpsStarted += 1;
      inspectorPluginCollInfoRef(parent);
      opInfoPtr->parentType = ncclProfileColl;
      opInfoPtr->parentObj = parent;
      opInfoPtr->commInfo = parent->commInfo;
      inspectorUnlockRWLock(&parent->guard);
    } else if (parentType == ncclProfileP2p) {
      struct inspectorP2pInfo* parent = (struct inspectorP2pInfo*)eDescr->parentObj;
      inspectorLockWr(&parent->guard);
      parent->nProxyOpsStarted += 1;
      inspectorPluginP2pInfoRef(parent);
      opInfoPtr->parentType = ncclProfileP2p;
      opInfoPtr->parentObj = parent;
      opInfoPtr->commInfo = parent->commInfo;
      inspectorUnlockRWLock(&parent->guard);
    }
  }

  *opInfo = opInfoPtr;
}

static void inspectorPluginProxyStepInfoInit(struct inspectorProxyStepInfo **stepInfo,
                                             ncclProfilerEventDescr_t *eDescr,
                                             struct inspectorCommInfo *commInfo) {
  struct inspectorProxyStepInfo *stepInfoPtr =
    (struct inspectorProxyStepInfo*)calloc(1, sizeof(struct inspectorProxyStepInfo));
  if (stepInfoPtr == nullptr) {
    INFO_INSPECTOR("Inspector: Failed to allocate memory for proxy step info structure");
    *stepInfo = nullptr;
    return;
  }

  struct inspectorProxyOpInfo *parent = nullptr;
  if (eDescr->parentObj != nullptr && *(uint64_t*)eDescr->parentObj == ncclProfileProxyOp) {
    parent = (struct inspectorProxyOpInfo*)eDescr->parentObj;
    inspectorLockWr(&parent->guard);
    parent->refCount += 1;
    stepInfoPtr->isSend = parent->isSend;
    stepInfoPtr->commInfo = parent->commInfo;
    inspectorUnlockRWLock(&parent->guard);
  } else {
    stepInfoPtr->isSend = -1;
    stepInfoPtr->commInfo = commInfo;
  }

  stepInfoPtr->type = ncclProfileProxyStep;
  stepInfoPtr->parent = parent;
  stepInfoPtr->proxyId = __atomic_add_fetch(&commInfo->proxySeqNum, 1, __ATOMIC_RELAXED);
  stepInfoPtr->rank = eDescr->rank;
  stepInfoPtr->step = eDescr->proxyStep.step;
  stepInfoPtr->tsStartUsec = inspectorGetTime();
  inspectorInlineListInit(&stepInfoPtr->states);
  inspectorLockInit(&stepInfoPtr->guard);
  *stepInfo = stepInfoPtr;
}

static void inspectorPluginProxyCtrlInfoInit(struct inspectorProxyCtrlInfo **ctrlInfo,
                                             struct inspectorCommInfo *commInfo) {
  struct inspectorProxyCtrlInfo *ctrlInfoPtr =
    (struct inspectorProxyCtrlInfo*)calloc(1, sizeof(struct inspectorProxyCtrlInfo));
  if (ctrlInfoPtr == nullptr) {
    INFO_INSPECTOR("Inspector: Failed to allocate memory for proxy ctrl info structure");
    *ctrlInfo = nullptr;
    return;
  }

  ctrlInfoPtr->type = ncclProfileProxyCtrl;
  ctrlInfoPtr->commInfo = commInfo;
  ctrlInfoPtr->proxyId = __atomic_add_fetch(&commInfo->proxySeqNum, 1, __ATOMIC_RELAXED);
  ctrlInfoPtr->tsStartUsec = inspectorGetTime();
  inspectorInlineListInit(&ctrlInfoPtr->states);
  inspectorLockInit(&ctrlInfoPtr->guard);
  *ctrlInfo = ctrlInfoPtr;
}

/*
 * Description:
 *   Initializes a new inspectorCollInfo structure for a collective
 *   event.
 *
 * Thread Safety:
 *   Not thread-safe (allocates and initializes a new collective info
 *   structure).
 *
 * Input:
 *
 *   struct inspectorCollInfo **collInfo - pointer to output
 *   collective info struct.
 *   ncclProfilerEventDescr_t *eDescr - event descriptor.
 *
 * Output:
 *   collInfo is set to the new collective info struct.
 *
 * Return:
 *   None.
 */
static void inspectorPluginCollInfoInit(struct inspectorCollInfo **collInfo,
                                        ncclProfilerEventDescr_t *eDescr,
                                        struct inspectorCommInfo *commInfo) {
  struct inspectorCollInfo *collInfoPtr = inspectorEventPoolAllocColl();
  if (collInfoPtr == nullptr) {
    INFO_INSPECTOR("Inspector: Failed to allocate memory for collective info structure");
    *collInfo = nullptr;
    return;
  }
  collInfoPtr->type = ncclProfileColl;
  collInfoPtr->refCount = 0;
  collInfoPtr->stopped = false;
  collInfoPtr->completedEnqueued = false;
  inspectorPluginCollInfoRef(collInfoPtr); //self ref; no locks needed
  collInfoPtr->func = eDescr->coll.func;
  collInfoPtr->algo = eDescr->coll.algo;
  collInfoPtr->proto = eDescr->coll.proto;
  collInfoPtr->sn = eDescr->coll.seqNumber;
  collInfoPtr->nChannels = eDescr->coll.nChannels;
  if (collInfoPtr->nChannels > 0) {
    inspectorPluginCollInfoRef(collInfoPtr); //extra ref for kernel completion
  }
  collInfoPtr->tsStartUsec = inspectorGetTime();
  collInfoPtr->msgSizeBytes =
    ncclTypeSize(inspectorStringToDatatype(eDescr->coll.datatype)) * eDescr->coll.count;


  collInfoPtr->commInfo = commInfo;
  collInfoPtr->collEvtTrk.sn = 0;
  collInfoPtr->collEvtTrk.nChannels = collInfoPtr->nChannels;
  inspectorInlineListInit(&collInfoPtr->proxyOps);
  inspectorRecordEventTrace(collInfoPtr->collEvtTrk.evntTrace,
                            NCCL_INSP_EVT_TRK_OP_START, collInfoPtr);

  inspectorLockInit(&collInfoPtr->guard);
  *collInfo = collInfoPtr;
}

static void inspectorPluginP2pInfoInit(struct inspectorP2pInfo **p2pInfo,
                                       ncclProfilerEventDescr_t *eDescr,
                                       struct inspectorCommInfo *commInfo) {
  struct inspectorP2pInfo *p2pInfoPtr = inspectorEventPoolAllocP2p();
  if (p2pInfoPtr == nullptr) {
    INFO_INSPECTOR("Inspector: Failed to allocate memory for P2P info structure");
    *p2pInfo = nullptr;
    return;
  }
  p2pInfoPtr->type = ncclProfileP2p;
  p2pInfoPtr->refCount = 0;
  p2pInfoPtr->stopped = false;
  p2pInfoPtr->completedEnqueued = false;
  inspectorPluginP2pInfoRef(p2pInfoPtr); // self ref
  p2pInfoPtr->func = eDescr->p2p.func;
  p2pInfoPtr->nChannels = eDescr->p2p.nChannels;
  p2pInfoPtr->peer = eDescr->p2p.peer;
  if (p2pInfoPtr->nChannels > 0) {
    inspectorPluginP2pInfoRef(p2pInfoPtr); // extra ref for kernel completion
  }
  p2pInfoPtr->tsStartUsec = inspectorGetTime();
  p2pInfoPtr->msgSizeBytes =
    ncclTypeSize(inspectorStringToDatatype(eDescr->p2p.datatype)) * eDescr->p2p.count;

  p2pInfoPtr->commInfo = commInfo;
  p2pInfoPtr->sn = __atomic_add_fetch(&commInfo->p2pSeqNum, 1, __ATOMIC_RELAXED);
  p2pInfoPtr->p2pEvtTrk.nChannels = p2pInfoPtr->nChannels;
  p2pInfoPtr->p2pEvtTrk.sn = p2pInfoPtr->sn;
  inspectorInlineListInit(&p2pInfoPtr->proxyOps);
  inspectorRecordP2pEventTrace(p2pInfoPtr->p2pEvtTrk.evntTrace,
                               NCCL_INSP_EVT_TRK_OP_START, p2pInfoPtr);

  inspectorLockInit(&p2pInfoPtr->guard);
  *p2pInfo = p2pInfoPtr;
}

/*
 * Description:
 *
 *   Initializes a new inspectorKernelChInfo structure for a kernel
 *   channel event.
 *
 * Thread Safety:
 *   Not thread-safe (initializes kernel channel info within a
 *   collective info structure).
 *
 * Input:
 *   struct inspectorKernelChInfo **kernelChInfo - pointer to output
 *   kernel channel info struct.
 *   ncclProfilerEventDescr_t *eDescr - event descriptor.
 *
 * Output:
 *
 *   kernelChInfo is set to the new kernel channel info struct.
 *
 * Return:
 *   None.
 */
static struct inspectorCollInfo* getKernelChCollInfo(struct inspectorKernelChInfo *kernelChInfo) {
  if (kernelChInfo && kernelChInfo->parentType == ncclProfileColl) {
    return (struct inspectorCollInfo*)kernelChInfo->parentObj;
  }
  return nullptr;
}

static struct inspectorP2pInfo* getKernelChP2pInfo(struct inspectorKernelChInfo *kernelChInfo) {
  if (kernelChInfo && kernelChInfo->parentType == ncclProfileP2p) {
    return (struct inspectorP2pInfo*)kernelChInfo->parentObj;
  }
  return nullptr;
}

static void inspectorPluginKernelChInfoInitColl(struct inspectorKernelChInfo **kernelChInfo,
                                                ncclProfilerEventDescr_t *eDescr,
                                                struct inspectorCollInfo *collInfo) {
  inspectorLockWr(&collInfo->guard);
  struct inspectorEventTraceInfo *krnlEvtTrk =
    collInfo->collEvtTrk.kernelCh[eDescr->kernelCh.channelId].evntTrace;
  inspectorRecordEventTrace(krnlEvtTrk,
                            NCCL_INSP_EVT_TRK_KERNEL_START,
                            collInfo);
  struct inspectorKernelChInfo *kernelChInfoPtr
    = &collInfo->kernelCh[eDescr->kernelCh.channelId];
  kernelChInfoPtr->type = ncclProfileKernelCh;
  kernelChInfoPtr->channelId = eDescr->kernelCh.channelId;
  kernelChInfoPtr->startGpuClk = eDescr->kernelCh.pTimer;
  kernelChInfoPtr->parentType = ncclProfileColl;
  kernelChInfoPtr->parentObj = collInfo;
  if (kernelChInfoPtr->stopGpuClk == 0) {
    inspectorPluginCollInfoRef(collInfo); //Pairs with Record Kernel Stop event
  }
  kernelChInfoPtr->tsStartUsec = inspectorGetTime();
  if (collInfo->nKernelChStarted == 0) {
    collInfo->tsStartUsec = kernelChInfoPtr->tsStartUsec;
  }
  collInfo->nKernelChStarted += 1;
  inspectorPluginCollInfoRef(collInfo); //Pairs with Stop Kernel Event

  *kernelChInfo = kernelChInfoPtr;
  inspectorUnlockRWLock(&collInfo->guard);
}

static void inspectorPluginKernelChInfoInitP2p(struct inspectorKernelChInfo **kernelChInfo,
                                               ncclProfilerEventDescr_t *eDescr,
                                               struct inspectorP2pInfo *p2pInfo) {
  inspectorLockWr(&p2pInfo->guard);
  struct inspectorEventTraceInfo *krnlEvtTrk =
    p2pInfo->p2pEvtTrk.kernelCh[eDescr->kernelCh.channelId].evntTrace;
  inspectorRecordP2pEventTrace(krnlEvtTrk,
                               NCCL_INSP_EVT_TRK_KERNEL_START,
                               p2pInfo);
  struct inspectorKernelChInfo *kernelChInfoPtr
    = &p2pInfo->kernelCh[eDescr->kernelCh.channelId];
  kernelChInfoPtr->type = ncclProfileKernelCh;
  kernelChInfoPtr->channelId = eDescr->kernelCh.channelId;
  kernelChInfoPtr->startGpuClk = eDescr->kernelCh.pTimer;
  kernelChInfoPtr->parentType = ncclProfileP2p;
  kernelChInfoPtr->parentObj = p2pInfo;
  if (kernelChInfoPtr->stopGpuClk == 0) {
    inspectorPluginP2pInfoRef(p2pInfo); //Pairs with Record Kernel Stop event
  }
  kernelChInfoPtr->tsStartUsec = inspectorGetTime();
  if (p2pInfo->nKernelChStarted == 0) {
    p2pInfo->tsStartUsec = kernelChInfoPtr->tsStartUsec;
  }
  p2pInfo->nKernelChStarted += 1;
  inspectorPluginP2pInfoRef(p2pInfo); //Pairs with Stop Kernel Event

  *kernelChInfo = kernelChInfoPtr;
  inspectorUnlockRWLock(&p2pInfo->guard);
}

static void inspectorPluginKernelChInfoInit(struct inspectorKernelChInfo **kernelChInfo,
                                            ncclProfilerEventDescr_t *eDescr) {
  if (eDescr->parentObj) {
    uint64_t parentType = *(uint64_t*)eDescr->parentObj;
    if (parentType == ncclProfileColl) {
      struct inspectorCollInfo *collInfo = (struct inspectorCollInfo*)eDescr->parentObj;
      if (collInfo && collInfo->type == ncclProfileColl) {
        inspectorPluginKernelChInfoInitColl(kernelChInfo, eDescr, collInfo);
      }
    } else if (parentType == ncclProfileP2p) {
      struct inspectorP2pInfo *p2pInfo = (struct inspectorP2pInfo*)eDescr->parentObj;
      if (p2pInfo && p2pInfo->type == ncclProfileP2p) {
        inspectorPluginKernelChInfoInitP2p(kernelChInfo, eDescr, p2pInfo);
      }
    }
  }
}

static bool inspectorShouldTrackColl(const ncclProfilerEventDescr_t* eDescr) {
  if (!eDescr) {
    return false;
  }
  int typeSize = ncclTypeSize(inspectorStringToDatatype(eDescr->coll.datatype));
  if (typeSize <= 0) {
    return true;
  }
  if (eDescr->coll.count == 0) {
    return false;
  }
  if (eDescr->coll.count > (SIZE_MAX / (size_t)typeSize)) {
    return true;
  }
  size_t msgSizeBytes = (size_t)typeSize * eDescr->coll.count;
  return msgSizeBytes >= ncclInspectorDumpMinSizeBytes;
}

static bool inspectorShouldTrackP2p(const ncclProfilerEventDescr_t* eDescr) {
  if (!eDescr) {
    return false;
  }
  int typeSize = ncclTypeSize(inspectorStringToDatatype(eDescr->p2p.datatype));
  if (typeSize <= 0) {
    return true;
  }
  if (eDescr->p2p.count == 0) {
    return false;
  }
  if (eDescr->p2p.count > (SIZE_MAX / (size_t)typeSize)) {
    return true;
  }
  size_t msgSizeBytes = (size_t)typeSize * eDescr->p2p.count;
  return msgSizeBytes >= ncclInspectorDumpMinSizeBytes;
}

static bool inspectorCollKernelsCompleteLocked(const struct inspectorCollInfo* collInfo) {
  return collInfo->nChannels == 0 ||
         ((collInfo->nKernelChCompleted == collInfo->nKernelChStarted) &&
          (collInfo->nKernelChCompleted == collInfo->nChannels));
}

static bool inspectorP2pKernelsCompleteLocked(const struct inspectorP2pInfo* p2pInfo) {
  return p2pInfo->nChannels == 0 ||
         ((p2pInfo->nKernelChCompleted == p2pInfo->nKernelChStarted) &&
          (p2pInfo->nKernelChCompleted == p2pInfo->nChannels));
}

static inspectorResult_t inspectorPluginCollMaybeCompleteLocked(
    struct inspectorCollInfo* collInfo,
    struct inspectorCompletedOpInfo* completedOp,
    bool* doCommUpdate) {
  *doCommUpdate = false;
  if (collInfo->completedEnqueued || !collInfo->stopped ||
      !inspectorCollKernelsCompleteLocked(collInfo) ||
      collInfo->nProxyOpsCompleted != collInfo->nProxyOpsStarted) {
    return inspectorSuccess;
  }

  if (collInfo->tsCompletedUsec == 0) {
    collInfo->tsCompletedUsec =
      collInfo->collEvtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_STOP].ts;
  }

  inspectorUpdateCollPerf(completedOp, collInfo);
  collInfo->completedEnqueued = true;

  if (requireKernelTiming &&
      completedOp->timingSource != inspectorTimingSourceKernelGpu) {
    inspectorCompletedOpInfoCleanup(completedOp);
    return inspectorSuccess;
  }

  *doCommUpdate = (collInfo->commInfo != nullptr);
  return inspectorSuccess;
}

static inspectorResult_t inspectorPluginP2pMaybeCompleteLocked(
    struct inspectorP2pInfo* p2pInfo,
    struct inspectorCompletedOpInfo* completedOp,
    bool* doCommUpdate) {
  *doCommUpdate = false;
  if (p2pInfo->completedEnqueued || !p2pInfo->stopped ||
      !inspectorP2pKernelsCompleteLocked(p2pInfo) ||
      p2pInfo->nProxyOpsCompleted != p2pInfo->nProxyOpsStarted) {
    return inspectorSuccess;
  }

  if (p2pInfo->tsCompletedUsec == 0) {
    p2pInfo->tsCompletedUsec =
      p2pInfo->p2pEvtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_STOP].ts;
  }

  inspectorUpdateP2pPerf(completedOp, p2pInfo);
  p2pInfo->completedEnqueued = true;

  if (requireKernelTiming &&
      completedOp->timingSource != inspectorTimingSourceKernelGpu) {
    inspectorCompletedOpInfoCleanup(completedOp);
    return inspectorSuccess;
  }

  *doCommUpdate = (p2pInfo->commInfo != nullptr);
  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Starts a profiling event for the NCCL Inspector plugin.
 *
 * Thread Safety:
 *   Thread-safe (allocates and initializes event structures).
 *
 * Input:
 *   void* context - plugin context.
 *   void** eHandle - pointer to event handle output.
 *   ncclProfilerEventDescr_t* eDescr - event descriptor.
 *
 * Output:
 *   eHandle is set to the new event structure.
 *
 * Return:
 *   ncclResult_t - success or error code.
 *
 */
__hidden ncclResult_t inspectorPluginStartEvent(void* context,
                                                void** eHandle,
                                                ncclProfilerEventDescr_t* eDescr) {
  if (context == nullptr || eDescr == nullptr) {
    INFO(NCCL_INIT, "Profiler/Plugin: context/eDescr NULL for start event %s", __func__);
    return ncclSuccess;
  }
  *eHandle = nullptr;
  if (eDescr->type == ncclProfileColl) {
    if (!inspectorShouldTrackColl(eDescr)) return ncclSuccess;
    struct inspectorCollInfo *collEvent = nullptr;
    struct inspectorCommInfo *commInfoCtx = (struct inspectorCommInfo*)context;
    inspectorPluginCollInfoInit(&collEvent, eDescr, commInfoCtx);
    *eHandle = collEvent;
  } else if (eDescr->type == ncclProfileP2p) {
    if (!enableNcclInspectorP2p) return ncclSuccess;
    if (!inspectorShouldTrackP2p(eDescr)) return ncclSuccess;
    struct inspectorP2pInfo *p2pEvent = nullptr;
    struct inspectorCommInfo *commInfoCtx = (struct inspectorCommInfo*)context;
    inspectorPluginP2pInfoInit(&p2pEvent, eDescr, commInfoCtx);
    *eHandle = p2pEvent;
  } else if (eDescr->type == ncclProfileKernelCh) {
    struct inspectorKernelChInfo *kernelChEvent = nullptr;
    inspectorPluginKernelChInfoInit(&kernelChEvent, eDescr);
    *eHandle = kernelChEvent;
  } else if (eDescr->type == ncclProfileProxyOp) {
    if (!enableNcclInspectorProxy || !inspectorIsDumpVerboseEnabled()) return ncclSuccess;
    struct inspectorProxyOpInfo *proxyOpEvent = nullptr;
    struct inspectorCommInfo *commInfoCtx = (struct inspectorCommInfo*)context;
    inspectorPluginProxyOpInfoInit(&proxyOpEvent, eDescr, commInfoCtx);
    *eHandle = proxyOpEvent;
  } else if (eDescr->type == ncclProfileProxyStep) {
    if (!enableNcclInspectorProxy || !inspectorIsDumpVerboseEnabled()) return ncclSuccess;
    struct inspectorProxyStepInfo *proxyStepEvent = nullptr;
    struct inspectorCommInfo *commInfoCtx = (struct inspectorCommInfo*)context;
    inspectorPluginProxyStepInfoInit(&proxyStepEvent, eDescr, commInfoCtx);
    *eHandle = proxyStepEvent;
  } else if (eDescr->type == ncclProfileProxyCtrl) {
    if (!enableNcclInspectorProxy || !inspectorIsDumpVerboseEnabled()) return ncclSuccess;
    struct inspectorProxyCtrlInfo *proxyCtrlEvent = nullptr;
    struct inspectorCommInfo *commInfoCtx = (struct inspectorCommInfo*)context;
    inspectorPluginProxyCtrlInfoInit(&proxyCtrlEvent, commInfoCtx);
    *eHandle = proxyCtrlEvent;
  } else {
    return ncclSuccess;
  }
  return ncclSuccess;
}

static ncclResult_t inspectorPluginStopEventColl(struct inspectorCollInfo *collInfo) {
  struct inspectorCompletedOpInfo completedOp;
  struct inspectorCommInfo *commInfo = nullptr;
  bool doCommUpdate = false;
  memset(&completedOp, 0, sizeof(completedOp));

  inspectorLockWr(&collInfo->guard);
  inspectorRecordEventTrace(collInfo->collEvtTrk.evntTrace,
                            NCCL_INSP_EVT_TRK_OP_STOP,
                            collInfo);
  collInfo->stopped = true;
  commInfo = collInfo->commInfo;
  inspectorResult_t completeRes =
    inspectorPluginCollMaybeCompleteLocked(collInfo, &completedOp, &doCommUpdate);
  inspectorResult_t res = inspectorPluginCollInfoDeRef(collInfo);
  inspectorUnlockRWLock(&collInfo->guard);
  if (completeRes != inspectorSuccess) {
    INFO_INSPECTOR("Inspector: Failed to complete collective: %s",
                   inspectorErrorString(completeRes));
  }
  if (doCommUpdate) {
    inspectorUpdateCommOpInfo(commInfo, &completedOp);
    inspectorCompletedOpInfoCleanup(&completedOp);
  }
  if (res == inspectorReturn) {
    inspectorPluginCollInfoCleanup(collInfo);
  }
  return ncclSuccess;
}

static ncclResult_t inspectorPluginStopEventP2p(struct inspectorP2pInfo *p2pInfo) {
  struct inspectorCompletedOpInfo completedOp;
  struct inspectorCommInfo *commInfo = nullptr;
  bool doCommUpdate = false;
  memset(&completedOp, 0, sizeof(completedOp));

  inspectorLockWr(&p2pInfo->guard);
  inspectorRecordP2pEventTrace(p2pInfo->p2pEvtTrk.evntTrace,
                               NCCL_INSP_EVT_TRK_OP_STOP,
                               p2pInfo);
  p2pInfo->stopped = true;
  commInfo = p2pInfo->commInfo;
  inspectorResult_t completeRes =
    inspectorPluginP2pMaybeCompleteLocked(p2pInfo, &completedOp, &doCommUpdate);
  inspectorResult_t res = inspectorPluginP2pInfoDeRef(p2pInfo);
  inspectorUnlockRWLock(&p2pInfo->guard);
  if (completeRes != inspectorSuccess) {
    INFO_INSPECTOR("Inspector: Failed to complete P2P: %s",
                   inspectorErrorString(completeRes));
  }
  if (doCommUpdate) {
    inspectorUpdateCommOpInfo(commInfo, &completedOp);
    inspectorCompletedOpInfoCleanup(&completedOp);
  }
  if (res == inspectorReturn) {
    inspectorPluginP2pInfoCleanup(p2pInfo);
  }
  return ncclSuccess;
}

static ncclResult_t inspectorPluginStopEventKernelChColl(struct inspectorKernelChInfo *kernelChInfo,
                                                         struct inspectorCollInfo *collInfo) {
  struct inspectorCompletedOpInfo completedOp;
  bool needsCleanup = false;
  bool doCommUpdate = false;
  memset(&completedOp, 0, sizeof(completedOp));

  inspectorLockWr(&collInfo->guard);
  struct inspectorCommInfo *commInfo = collInfo->commInfo;
  struct inspectorEventTraceInfo *krnlEvtTrk =
    collInfo->collEvtTrk.kernelCh[kernelChInfo->channelId].evntTrace;
  inspectorRecordEventTrace(krnlEvtTrk,
                            NCCL_INSP_EVT_TRK_KERNEL_STOP,
                            collInfo);
  kernelChInfo->tsCompletedUsec = inspectorGetTime();
  collInfo->nKernelChCompleted += 1;

  inspectorResult_t res = inspectorPluginCollInfoDeRef(collInfo);
  if (res == inspectorReturn) {
    needsCleanup = true;
    goto done;
  }

  if ((collInfo->nKernelChCompleted == collInfo->nKernelChStarted)
      && (collInfo->nKernelChCompleted == collInfo->nChannels)) {

    collInfo->tsCompletedUsec = kernelChInfo->tsCompletedUsec;
    res = inspectorPluginCollMaybeCompleteLocked(collInfo, &completedOp,
                                                &doCommUpdate);
    if (res != inspectorSuccess) {
      INFO_INSPECTOR("Inspector: Failed to complete collective: %s",
                     inspectorErrorString(res));
      doCommUpdate = false;
    }

    res = inspectorPluginCollInfoDeRef(collInfo);
    if (res == inspectorReturn) {
      needsCleanup = true;
    }
  }

done:
  inspectorUnlockRWLock(&collInfo->guard);
  if (needsCleanup) {
    inspectorPluginCollInfoCleanup(collInfo);
  }
  if (doCommUpdate) {
    inspectorUpdateCommOpInfo(commInfo, &completedOp);
    inspectorCompletedOpInfoCleanup(&completedOp);
  }
  return ncclSuccess;
}

static ncclResult_t inspectorPluginStopEventKernelChP2p(struct inspectorKernelChInfo *kernelChInfo,
                                                        struct inspectorP2pInfo *p2pInfo) {
  struct inspectorCompletedOpInfo completedOp;
  bool needsCleanup = false;
  bool doCommUpdate = false;
  memset(&completedOp, 0, sizeof(completedOp));

  inspectorLockWr(&p2pInfo->guard);
  struct inspectorCommInfo *commInfo = p2pInfo->commInfo;
  struct inspectorEventTraceInfo *krnlEvtTrk =
    p2pInfo->p2pEvtTrk.kernelCh[kernelChInfo->channelId].evntTrace;
  inspectorRecordP2pEventTrace(krnlEvtTrk,
                               NCCL_INSP_EVT_TRK_KERNEL_STOP,
                               p2pInfo);
  kernelChInfo->tsCompletedUsec = inspectorGetTime();
  p2pInfo->nKernelChCompleted += 1;

  inspectorResult_t res = inspectorPluginP2pInfoDeRef(p2pInfo);
  if (res == inspectorReturn) {
    needsCleanup = true;
    goto done;
  }

  if ((p2pInfo->nKernelChCompleted == p2pInfo->nKernelChStarted)
      && (p2pInfo->nKernelChCompleted == p2pInfo->nChannels)) {

    p2pInfo->tsCompletedUsec = kernelChInfo->tsCompletedUsec;
    res = inspectorPluginP2pMaybeCompleteLocked(p2pInfo, &completedOp,
                                               &doCommUpdate);
    if (res != inspectorSuccess) {
      INFO_INSPECTOR("Inspector: Failed to complete P2P: %s",
                     inspectorErrorString(res));
      doCommUpdate = false;
    }

    res = inspectorPluginP2pInfoDeRef(p2pInfo);
    if (res == inspectorReturn) {
      needsCleanup = true;
    }
  }

done:
  inspectorUnlockRWLock(&p2pInfo->guard);
  if (needsCleanup) {
    inspectorPluginP2pInfoCleanup(p2pInfo);
  }
  if (doCommUpdate) {
    inspectorUpdateCommOpInfo(commInfo, &completedOp);
    inspectorCompletedOpInfoCleanup(&completedOp);
  }
  return ncclSuccess;
}

static ncclResult_t inspectorPluginStopEventKernelCh(struct inspectorKernelChInfo *kernelChInfo) {
  if (kernelChInfo->parentType == ncclProfileColl) {
    struct inspectorCollInfo *collInfo = getKernelChCollInfo(kernelChInfo);
    if (collInfo) return inspectorPluginStopEventKernelChColl(kernelChInfo, collInfo);
  } else if (kernelChInfo->parentType == ncclProfileP2p) {
    struct inspectorP2pInfo *p2pInfo = getKernelChP2pInfo(kernelChInfo);
    if (p2pInfo) return inspectorPluginStopEventKernelChP2p(kernelChInfo, p2pInfo);
  }
  return ncclSuccess;
}

static inspectorResult_t inspectorProxyStepRecordFromStepLocked(
    struct inspectorProxyStepRecordInfo* record,
    struct inspectorProxyStepInfo* stepInfo) {
  if (record == nullptr || stepInfo == nullptr) return inspectorMemoryError;

  memset(record, 0, sizeof(*record));
  record->rank = stepInfo->rank;
  record->pid = stepInfo->parent ? stepInfo->parent->pid : getpid();
  record->channelId = stepInfo->parent ? stepInfo->parent->channelId : 0;
  record->peer = stepInfo->parent ? stepInfo->parent->peer : -1;
  record->nSteps = stepInfo->parent ? stepInfo->parent->nSteps : 0;
  record->chunkSize = stepInfo->parent ? stepInfo->parent->chunkSize : 0;
  record->isSend = stepInfo->isSend;
  record->step = stepInfo->step;
  record->transSize = stepInfo->transSize;
  record->tsStartUsec = stepInfo->tsStartUsec;
  record->tsCompletedUsec = stepInfo->tsCompletedUsec;
  return inspectorProxyEventStateListCopy(&record->states, &stepInfo->states);
}

static inspectorResult_t inspectorProxyOpAppendStepRecordLocked(
    struct inspectorProxyOpInfo* opInfo,
    const struct inspectorProxyStepRecordInfo* record) {
  if (opInfo == nullptr || record == nullptr) return inspectorMemoryError;

  return inspectorProxyStepRecordListAppend(&opInfo->steps, record);
}

static inspectorResult_t inspectorPluginAppendCompletedProxyToParent(
    struct inspectorProxyOpInfo* opInfo,
    const struct inspectorCompletedProxyEventInfo* completedProxy,
    struct inspectorCompletedOpInfo* completedOp,
    struct inspectorCommInfo** commInfo,
    bool* doCommUpdate,
    bool* needsParentCleanup) {
  if (opInfo == nullptr || completedProxy == nullptr || completedOp == nullptr ||
      commInfo == nullptr || doCommUpdate == nullptr || needsParentCleanup == nullptr) {
    return inspectorMemoryError;
  }

  *commInfo = nullptr;
  *doCommUpdate = false;
  *needsParentCleanup = false;
  inspectorResult_t res = inspectorSuccess;

  if (opInfo->parentType == ncclProfileColl) {
    struct inspectorCollInfo* parent = (struct inspectorCollInfo*)opInfo->parentObj;
    if (parent == nullptr) return inspectorSuccess;

    inspectorLockWr(&parent->guard);
    inspectorResult_t appendRes =
      inspectorProxyOpRecordListAppend(&parent->proxyOps, completedProxy);
    if (appendRes != inspectorSuccess) {
      INFO_INSPECTOR("Inspector: Failed to append proxy op to collective: %s",
                     inspectorErrorString(appendRes));
      res = appendRes;
    }
    parent->nProxyOpsCompleted += 1;
    *commInfo = parent->commInfo;

    inspectorResult_t completeRes =
      inspectorPluginCollMaybeCompleteLocked(parent, completedOp, doCommUpdate);
    if (completeRes != inspectorSuccess) {
      res = completeRes;
    }

    inspectorResult_t derefRes = inspectorPluginCollInfoDeRef(parent);
    if (derefRes == inspectorReturn) {
      *needsParentCleanup = true;
    }
    inspectorUnlockRWLock(&parent->guard);
  } else if (opInfo->parentType == ncclProfileP2p) {
    struct inspectorP2pInfo* parent = (struct inspectorP2pInfo*)opInfo->parentObj;
    if (parent == nullptr) return inspectorSuccess;

    inspectorLockWr(&parent->guard);
    inspectorResult_t appendRes =
      inspectorProxyOpRecordListAppend(&parent->proxyOps, completedProxy);
    if (appendRes != inspectorSuccess) {
      INFO_INSPECTOR("Inspector: Failed to append proxy op to P2P: %s",
                     inspectorErrorString(appendRes));
      res = appendRes;
    }
    parent->nProxyOpsCompleted += 1;
    *commInfo = parent->commInfo;

    inspectorResult_t completeRes =
      inspectorPluginP2pMaybeCompleteLocked(parent, completedOp, doCommUpdate);
    if (completeRes != inspectorSuccess) {
      res = completeRes;
    }

    inspectorResult_t derefRes = inspectorPluginP2pInfoDeRef(parent);
    if (derefRes == inspectorReturn) {
      *needsParentCleanup = true;
    }
    inspectorUnlockRWLock(&parent->guard);
  }

  return res;
}

static ncclResult_t inspectorPluginStopEventProxyOp(struct inspectorProxyOpInfo *opInfo) {
  struct inspectorCompletedProxyEventInfo completedProxy;
  struct inspectorCompletedOpInfo completedParentOp;
  struct inspectorCommInfo *commInfo = nullptr;
  struct inspectorCommInfo *parentCommInfo = nullptr;
  bool doCommUpdate = false;
  bool doParentCommUpdate = false;
  bool needsCleanup = false;
  bool needsParentCleanup = false;
  inspectorCompletedProxyEventInfoInit(&completedProxy);
  memset(&completedParentOp, 0, sizeof(completedParentOp));

  inspectorLockWr(&opInfo->guard);
  opInfo->tsCompletedUsec = inspectorGetTime();
  opInfo->stopped = true;
  opInfo->refCount -= 1;
  inspectorResult_t res =
    inspectorPluginProxyOpMaybeCompleteLocked(opInfo, &completedProxy, &commInfo,
                                              &doCommUpdate, &needsCleanup);
  inspectorUnlockRWLock(&opInfo->guard);
  if (res != inspectorSuccess) {
    INFO_INSPECTOR("Inspector: Failed to complete proxy op: %s",
                   inspectorErrorString(res));
  }

  if (needsCleanup &&
      (opInfo->parentType == ncclProfileColl ||
       opInfo->parentType == ncclProfileP2p)) {
    res = inspectorPluginAppendCompletedProxyToParent(opInfo, &completedProxy,
                                                      &completedParentOp,
                                                      &parentCommInfo,
                                                      &doParentCommUpdate,
                                                      &needsParentCleanup);
    if (res != inspectorSuccess) {
      INFO_INSPECTOR("Inspector: Failed to attach proxy op to parent: %s",
                     inspectorErrorString(res));
    }
  } else if (doCommUpdate) {
    res = inspectorUpdateCommProxyInfo(commInfo, &completedProxy);
    if (res != inspectorSuccess) {
      INFO_INSPECTOR("Inspector: Failed to enqueue proxy op: %s",
                     inspectorErrorString(res));
    }
  }
  if (doParentCommUpdate) {
    inspectorUpdateCommOpInfo(parentCommInfo, &completedParentOp);
    inspectorCompletedOpInfoCleanup(&completedParentOp);
  }
  inspectorCompletedProxyEventInfoCleanup(&completedProxy);
  if (needsParentCleanup) {
    if (opInfo->parentType == ncclProfileColl) {
      inspectorPluginCollInfoCleanup((struct inspectorCollInfo*)opInfo->parentObj);
    } else if (opInfo->parentType == ncclProfileP2p) {
      inspectorPluginP2pInfoCleanup((struct inspectorP2pInfo*)opInfo->parentObj);
    }
  }
  if (needsCleanup) {
    inspectorPluginProxyOpInfoCleanup(opInfo);
  }
  return ncclSuccess;
}

static ncclResult_t inspectorPluginStopEventProxyStep(struct inspectorProxyStepInfo *stepInfo) {
  struct inspectorCompletedProxyEventInfo completedProxyOp;
  struct inspectorCompletedOpInfo completedParentOp;
  struct inspectorProxyStepRecordInfo stepRecord;
  struct inspectorCommInfo *opCommInfo = nullptr;
  struct inspectorCommInfo *parentCommInfo = nullptr;
  bool doOpCommUpdate = false;
  bool doParentCommUpdate = false;
  bool needsOpCleanup = false;
  bool needsParentCleanup = false;
  inspectorCompletedProxyEventInfoInit(&completedProxyOp);
  memset(&completedParentOp, 0, sizeof(completedParentOp));
  memset(&stepRecord, 0, sizeof(stepRecord));

  inspectorLockWr(&stepInfo->guard);
  stepInfo->tsCompletedUsec = inspectorGetTime();
  inspectorResult_t recordRes = inspectorProxyStepRecordFromStepLocked(&stepRecord, stepInfo);
  inspectorUnlockRWLock(&stepInfo->guard);
  if (recordRes != inspectorSuccess) {
    INFO_INSPECTOR("Inspector: Failed to copy proxy step record: %s",
                   inspectorErrorString(recordRes));
  }

  if (stepInfo->parent != nullptr) {
    struct inspectorProxyOpInfo *parent = stepInfo->parent;
    inspectorLockWr(&parent->guard);
    if (recordRes == inspectorSuccess) {
      inspectorResult_t appendRes =
        inspectorProxyOpAppendStepRecordLocked(parent, &stepRecord);
      if (appendRes != inspectorSuccess) {
        INFO_INSPECTOR("Inspector: Failed to append proxy step to proxy op: %s",
                       inspectorErrorString(appendRes));
      }
    }
    parent->refCount -= 1;
    inspectorResult_t res =
      inspectorPluginProxyOpMaybeCompleteLocked(parent, &completedProxyOp,
                                                &opCommInfo, &doOpCommUpdate,
                                                &needsOpCleanup);
    inspectorUnlockRWLock(&parent->guard);
    if (res != inspectorSuccess) {
      INFO_INSPECTOR("Inspector: Failed to complete parent proxy op: %s",
                     inspectorErrorString(res));
    }

    if (needsOpCleanup &&
        (parent->parentType == ncclProfileColl ||
         parent->parentType == ncclProfileP2p)) {
      res = inspectorPluginAppendCompletedProxyToParent(parent, &completedProxyOp,
                                                        &completedParentOp,
                                                        &parentCommInfo,
                                                        &doParentCommUpdate,
                                                        &needsParentCleanup);
      if (res != inspectorSuccess) {
        INFO_INSPECTOR("Inspector: Failed to attach parent proxy op: %s",
                       inspectorErrorString(res));
      }
    } else if (doOpCommUpdate) {
      res = inspectorUpdateCommProxyInfo(opCommInfo, &completedProxyOp);
      if (res != inspectorSuccess) {
        INFO_INSPECTOR("Inspector: Failed to enqueue parent proxy op: %s",
                       inspectorErrorString(res));
      }
    }
    if (doParentCommUpdate) {
      inspectorUpdateCommOpInfo(parentCommInfo, &completedParentOp);
      inspectorCompletedOpInfoCleanup(&completedParentOp);
    }
    if (needsParentCleanup) {
      if (parent->parentType == ncclProfileColl) {
        inspectorPluginCollInfoCleanup((struct inspectorCollInfo*)parent->parentObj);
      } else if (parent->parentType == ncclProfileP2p) {
        inspectorPluginP2pInfoCleanup((struct inspectorP2pInfo*)parent->parentObj);
      }
    }
    if (needsOpCleanup) {
      inspectorPluginProxyOpInfoCleanup(parent);
    }
  }

  inspectorProxyStepRecordInfoCleanup(&stepRecord);
  inspectorCompletedProxyEventInfoCleanup(&completedProxyOp);
  inspectorPluginProxyStepInfoCleanup(stepInfo);
  return ncclSuccess;
}

static ncclResult_t inspectorPluginStopEventProxyCtrl(struct inspectorProxyCtrlInfo *ctrlInfo) {
  struct inspectorCompletedProxyEventInfo completedProxy;
  struct inspectorCommInfo *commInfo = nullptr;
  inspectorCompletedProxyEventInfoInit(&completedProxy);

  inspectorLockWr(&ctrlInfo->guard);
  ctrlInfo->tsCompletedUsec = inspectorGetTime();
  commInfo = ctrlInfo->commInfo;
  inspectorResult_t res = inspectorCompletedProxyFromCtrlLocked(&completedProxy, ctrlInfo);
  inspectorUnlockRWLock(&ctrlInfo->guard);
  if (res != inspectorSuccess) {
    INFO_INSPECTOR("Inspector: Failed to complete proxy ctrl: %s",
                   inspectorErrorString(res));
  }

  if (commInfo != nullptr && res == inspectorSuccess) {
    res = inspectorUpdateCommProxyInfo(commInfo, &completedProxy);
    if (res != inspectorSuccess) {
      INFO_INSPECTOR("Inspector: Failed to enqueue proxy ctrl: %s",
                     inspectorErrorString(res));
    }
  }
  inspectorCompletedProxyEventInfoCleanup(&completedProxy);
  inspectorPluginProxyCtrlInfoCleanup(ctrlInfo);
  return ncclSuccess;
}

static ncclResult_t inspectorPluginRecordEventStateKernelChColl(struct inspectorKernelChInfo *kernelChInfo,
                                                                struct inspectorCollInfo *collInfo,
                                                                ncclProfilerEventStateArgs_t* eStateArgs) {
  bool needsCleanup = false;
  inspectorLockWr(&collInfo->guard);
  struct inspectorEventTraceInfo *krnlEvtTrk
    = collInfo->collEvtTrk.kernelCh[kernelChInfo->channelId].evntTrace;
  inspectorRecordEventTrace(krnlEvtTrk,
                            NCCL_INSP_EVT_TRK_KERNEL_RECORD,
                            collInfo);
  kernelChInfo->stopGpuClk = eStateArgs->kernelCh.pTimer;
  if (kernelChInfo->startGpuClk != 0) {
    inspectorResult_t res = inspectorPluginCollInfoDeRef(collInfo);
    if (res == inspectorReturn) {
      needsCleanup = true;
    }
  }
  inspectorUnlockRWLock(&collInfo->guard);
  if (needsCleanup) {
    inspectorPluginCollInfoCleanup(collInfo);
  }
  return ncclSuccess;
}

static ncclResult_t inspectorPluginRecordEventStateKernelChP2p(struct inspectorKernelChInfo *kernelChInfo,
                                                               struct inspectorP2pInfo *p2pInfo,
                                                               ncclProfilerEventStateArgs_t* eStateArgs) {
  bool needsCleanup = false;
  inspectorLockWr(&p2pInfo->guard);
  struct inspectorEventTraceInfo *krnlEvtTrk
    = p2pInfo->p2pEvtTrk.kernelCh[kernelChInfo->channelId].evntTrace;
  inspectorRecordP2pEventTrace(krnlEvtTrk,
                               NCCL_INSP_EVT_TRK_KERNEL_RECORD,
                               p2pInfo);
  kernelChInfo->stopGpuClk = eStateArgs->kernelCh.pTimer;
  if (kernelChInfo->startGpuClk != 0) {
    inspectorResult_t res = inspectorPluginP2pInfoDeRef(p2pInfo);
    if (res == inspectorReturn) {
      needsCleanup = true;
    }
  }
  inspectorUnlockRWLock(&p2pInfo->guard);
  if (needsCleanup) {
    inspectorPluginP2pInfoCleanup(p2pInfo);
  }
  return ncclSuccess;
}

static ncclResult_t inspectorPluginRecordEventStateKernelCh(struct inspectorKernelChInfo *kernelChInfo,
                                                            ncclProfilerEventStateArgs_t* eStateArgs) {
  if (kernelChInfo->parentType == ncclProfileColl) {
    struct inspectorCollInfo *collInfo = getKernelChCollInfo(kernelChInfo);
    if (collInfo) return inspectorPluginRecordEventStateKernelChColl(kernelChInfo, collInfo, eStateArgs);
  } else if (kernelChInfo->parentType == ncclProfileP2p) {
    struct inspectorP2pInfo *p2pInfo = getKernelChP2pInfo(kernelChInfo);
    if (p2pInfo) return inspectorPluginRecordEventStateKernelChP2p(kernelChInfo, p2pInfo, eStateArgs);
  }
  return ncclSuccess;
}

static ncclResult_t inspectorPluginRecordEventStateProxyOp(struct inspectorProxyOpInfo *opInfo,
                                                           ncclProfilerEventState_t eState,
                                                           ncclProfilerEventStateArgs_t* eStateArgs) {
  inspectorLockWr(&opInfo->guard);
  inspectorResult_t res = inspectorRecordProxyEventState(&opInfo->states, eState, eStateArgs);
  inspectorUnlockRWLock(&opInfo->guard);
  if (res != inspectorSuccess) {
    INFO_INSPECTOR("Inspector: Failed to record proxy op state: %s",
                   inspectorErrorString(res));
  }
  return ncclSuccess;
}

static bool inspectorProxyStepStateCarriesTransSize(ncclProfilerEventState_t eState) {
  return eState == ncclProfilerProxyStepSendWait ||
         eState == ncclProfilerProxyStepRecvFlushWait;
}

static ncclResult_t inspectorPluginRecordEventStateProxyStep(struct inspectorProxyStepInfo *stepInfo,
                                                             ncclProfilerEventState_t eState,
                                                             ncclProfilerEventStateArgs_t* eStateArgs) {
  if (eState == ncclProfilerProxyStepSendPeerWait_v4) return ncclSuccess;

  size_t transSize = 0;
  if (eStateArgs != nullptr && inspectorProxyStepStateCarriesTransSize(eState)) {
    transSize = eStateArgs->proxyStep.transSize;
  }

  inspectorLockWr(&stepInfo->guard);
  inspectorResult_t res = inspectorRecordProxyEventState(&stepInfo->states, eState, eStateArgs);
  stepInfo->transSize += transSize;
  inspectorUnlockRWLock(&stepInfo->guard);
  if (res != inspectorSuccess) {
    INFO_INSPECTOR("Inspector: Failed to record proxy step state: %s",
                   inspectorErrorString(res));
  }

  if (transSize != 0 && stepInfo->parent != nullptr) {
    inspectorLockWr(&stepInfo->parent->guard);
    stepInfo->parent->transSize += transSize;
    inspectorUnlockRWLock(&stepInfo->parent->guard);
  }
  return ncclSuccess;
}

static ncclResult_t inspectorPluginRecordEventStateProxyCtrl(struct inspectorProxyCtrlInfo *ctrlInfo,
                                                             ncclProfilerEventState_t eState,
                                                             ncclProfilerEventStateArgs_t* eStateArgs) {
  inspectorLockWr(&ctrlInfo->guard);
  inspectorResult_t res = inspectorRecordProxyEventState(&ctrlInfo->states, eState, eStateArgs);
  inspectorUnlockRWLock(&ctrlInfo->guard);
  if (res != inspectorSuccess) {
    INFO_INSPECTOR("Inspector: Failed to record proxy ctrl state: %s",
                   inspectorErrorString(res));
  }
  return ncclSuccess;
}

/*
 * Description:
 *
 *   Stops a profiling event for the NCCL Inspector plugin.
 *
 * Thread Safety:
 *
 *   Thread-safe (updates event state and performance info).
 *
 * Input:
 *
 *   void *eHandle - event handle.
 *
 * Output:
 *
 *   Event is stopped and performance info may be updated.
 *
 * Return:
 *   ncclResult_t - success or error code.
 *
 */
__hidden ncclResult_t inspectorPluginStopEvent(void *eHandle) {
  if (eHandle == nullptr) {
    INFO(NCCL_INIT,
         "Profiler/Plugin: Event Handle NULL for start event %s", __func__);
    return ncclSuccess;
  }

  uint64_t type = *(uint64_t *)eHandle;
  if (type == ncclProfileColl) {
    struct inspectorCollInfo *collInfo = (struct inspectorCollInfo *)eHandle;
    return inspectorPluginStopEventColl(collInfo);
  } else if (type == ncclProfileP2p) {
    struct inspectorP2pInfo *p2pInfo = (struct inspectorP2pInfo *)eHandle;
    return inspectorPluginStopEventP2p(p2pInfo);
  } else if (type == ncclProfileKernelCh) {
    struct inspectorKernelChInfo *kernelChInfo
      = (struct inspectorKernelChInfo *)eHandle;
    return inspectorPluginStopEventKernelCh(kernelChInfo);
  } else if (type == ncclProfileProxyOp) {
    struct inspectorProxyOpInfo *proxyOpInfo =
      (struct inspectorProxyOpInfo *)eHandle;
    return inspectorPluginStopEventProxyOp(proxyOpInfo);
  } else if (type == ncclProfileProxyStep) {
    struct inspectorProxyStepInfo *proxyStepInfo =
      (struct inspectorProxyStepInfo *)eHandle;
    return inspectorPluginStopEventProxyStep(proxyStepInfo);
  } else if (type == ncclProfileProxyCtrl) {
    struct inspectorProxyCtrlInfo *proxyCtrlInfo =
      (struct inspectorProxyCtrlInfo *)eHandle;
    return inspectorPluginStopEventProxyCtrl(proxyCtrlInfo);
  }
  return ncclSuccess;
}

/*
 * Description:
 *
 *   Records the state of a profiling event for the NCCL Inspector
 *   plugin.
 *
 * Thread Safety:
 *
 *   Thread-safe (updates event state as needed).
 *
 * Input:
 *   void* eHandle - event handle.
 *   ncclProfilerEventState_t eState - event state.
 *   ncclProfilerEventStateArgs_t* eStateArgs - event state arguments.
 *
 * Output:
 *   Event state is updated as needed.
 *
 * Return:
 *   ncclResult_t - success or error code.
 *
 */
__hidden ncclResult_t inspectorPluginRecordEventState(void* eHandle,
                                                      ncclProfilerEventState_t eState,
                                                      ncclProfilerEventStateArgs_t* eStateArgs) {
  if (eHandle == nullptr)
    return ncclSuccess;

  uint64_t type = *(uint64_t *)eHandle;

  if (type == ncclProfileKernelCh && eState == ncclProfilerKernelChStop) {
    if (eStateArgs == nullptr) {
      return ncclSuccess;
    }

    struct inspectorKernelChInfo *kernelChInfo
      = (struct inspectorKernelChInfo *)eHandle;

    return inspectorPluginRecordEventStateKernelCh(kernelChInfo,
                                                   eStateArgs);

  } else if (type == ncclProfileProxyOp) {
    struct inspectorProxyOpInfo *proxyOpInfo =
      (struct inspectorProxyOpInfo *)eHandle;
    return inspectorPluginRecordEventStateProxyOp(proxyOpInfo, eState, eStateArgs);

  } else if (type == ncclProfileProxyStep) {
    struct inspectorProxyStepInfo *proxyStepInfo =
      (struct inspectorProxyStepInfo *)eHandle;
    return inspectorPluginRecordEventStateProxyStep(proxyStepInfo, eState, eStateArgs);

  } else if (type == ncclProfileProxyCtrl) {
    struct inspectorProxyCtrlInfo *proxyCtrlInfo =
      (struct inspectorProxyCtrlInfo *)eHandle;
    return inspectorPluginRecordEventStateProxyCtrl(proxyCtrlInfo, eState, eStateArgs);

  }
  return ncclSuccess;
}

ncclProfiler_t ncclProfiler_v5 = {
  "Inspector",
  inspectorPluginInit,
  inspectorPluginStartEvent,
  inspectorPluginStopEvent,
  inspectorPluginRecordEventState,
  inspectorPluginFinalize,
};
