/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <atomic>
#include <new>
#include <linux/limits.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "profiler.h"
#include "inspector.h"
#include "inspector_ring.h"
#include "inspector_object_pool.h"

#define __hidden __attribute__ ((visibility("hidden")))

// Spin-wait instrumentation counters
static std::atomic<uint64_t> gAsyncEnqueueCalls{0};
static std::atomic<uint64_t> gAsyncEnqueueSpinEvents{0};
static std::atomic<uint64_t> gAsyncEnqueueTotalSpins{0};
static std::atomic<uint64_t> gAsyncEnqueueUsleeps{0};
static std::atomic<uint64_t> gAsyncHandleAllocFails{0};
static std::atomic<uint64_t> gAsyncHandleReleaseSpins{0};

// Minimum message size for proxy event tracking (default 16MB)
static uint64_t minProxyMsgSizeBytes = 16ULL * 1024 * 1024;

// Generic event info for lightweight tracking of API/Group/KernelLaunch/CE events
struct inspectorGenericEventInfo {
  uint64_t type;           // First field - used for type dispatch in stop/record
  uint64_t tsStartUsec;   // Start timestamp
  uint64_t tsStopUsec;    // Stop timestamp
  struct inspectorCommInfo *commInfo; // Communicator for ring enqueue
  int rank;               // Originating rank from descriptor
  // Descriptor-specific fields captured at startEvent
  char funcName[64];      // CollApi/P2pApi: func name
  size_t count;           // CollApi/P2pApi: element count
  char datatype[32];      // CollApi/P2pApi: datatype
  int root;              // CollApi: root rank
  int groupDepth;        // GroupApi: group depth
  bool graphCaptured;    // GroupApi/CollApi/P2pApi: CUDA graph captured
  int64_t netPluginId;   // NetPlugin: id
};

// Counters for generic events (for diagnostics)
static std::atomic<uint64_t> gGenericEventStartCount{0};
static std::atomic<uint64_t> gGenericEventStopCount{0};

static int gInitialized;

static pthread_mutex_t gLock = PTHREAD_MUTEX_INITIALIZER;

// Typed object pools (replacing inspector_event_pool)
static ObjectPool<inspectorCollInfo>* g_collPool = nullptr;
static ObjectPool<inspectorP2pInfo>* g_p2pPool = nullptr;
static ObjectPool<inspectorProxyOpInfo>* g_proxyOpPool = nullptr;
static ObjectPool<inspectorProxyStepInfo>* g_proxyStepPool = nullptr;
static ObjectPool<inspectorProxyCtrlInfo>* g_proxyCtrlPool = nullptr;

static ncclResult_t inspectorPluginStartEventSync(void* context,
                                                  void** eHandle,
                                                  ncclProfilerEventDescr_t* eDescr);
static ncclResult_t inspectorPluginStopEventSync(void *eHandle, uint64_t tsOverride = 0);
static ncclResult_t inspectorPluginRecordEventStateSync(void* eHandle,
                                                        ncclProfilerEventState_t eState,
                                                        ncclProfilerEventStateArgs_t* eStateArgs,
                                                        uint64_t tsOverride = 0);

enum inspectorAsyncEventType {
  inspectorAsyncEventStart = 0,
  inspectorAsyncEventStop = 1,
  inspectorAsyncEventRecord = 2,
};

struct inspectorAsyncHandle {
  uint64_t type;
  std::atomic<void*> realHandle;
};

struct inspectorAsyncEvent {
  inspectorAsyncEventType type;
  void* eHandle;
  void* context;
  inspectorAsyncHandle* asyncHandle;
  ncclProfilerEventDescr_t eDescr;
  ncclProfilerEventState_t eState;
  bool hasArgs;
  ncclProfilerEventStateArgs_t eStateArgs;
  uint64_t tsProducerUsec;
  struct inspectorCommInfo* commInfo;
};

struct inspectorAsyncSlot {
  inspectorAsyncEvent event;
  std::atomic<bool> ready;
};

struct inspectorAsyncHandlePool {
  inspectorAsyncHandle* handles;
  inspectorAsyncHandle** freeHandles;
  size_t size;
  std::atomic<size_t> head;
  std::atomic<size_t> tail;
};

struct inspectorAsyncQueue {
  bool initialized;
  std::atomic<bool> stop;
  std::atomic<bool> processing;
  size_t size;
  std::atomic<size_t> head;
  std::atomic<size_t> tail;
  inspectorAsyncSlot* slots;
  pthread_t pthread;
};

static inspectorAsyncQueue gAsyncQueue;
static inspectorAsyncHandlePool gAsyncHandlePool;

static void inspectorPluginAsyncHandlePoolReset() {
  gAsyncHandlePool.handles = nullptr;
  gAsyncHandlePool.freeHandles = nullptr;
  gAsyncHandlePool.size = 0;
  gAsyncHandlePool.head.store(0, std::memory_order_relaxed);
  gAsyncHandlePool.tail.store(0, std::memory_order_relaxed);
}

static void inspectorPluginAsyncReset() {
  gAsyncQueue.initialized = false;
  gAsyncQueue.stop.store(false, std::memory_order_relaxed);
  gAsyncQueue.processing.store(false, std::memory_order_relaxed);
  gAsyncQueue.size = 0;
  gAsyncQueue.head.store(0, std::memory_order_relaxed);
  gAsyncQueue.tail.store(0, std::memory_order_relaxed);
  gAsyncQueue.slots = nullptr;
}

static bool inspectorPluginIsAsyncHandle(void* eHandle) {
  if (eHandle == nullptr || gAsyncHandlePool.handles == nullptr) return false;
  inspectorAsyncHandle* handle = (inspectorAsyncHandle*)eHandle;
  return handle >= gAsyncHandlePool.handles &&
         handle < gAsyncHandlePool.handles + gAsyncHandlePool.size;
}

static void* inspectorPluginResolveAsyncHandle(void* eHandle) {
  if (!inspectorPluginIsAsyncHandle(eHandle)) return eHandle;
  inspectorAsyncHandle* handle = (inspectorAsyncHandle*)eHandle;
  return handle->realHandle.load(std::memory_order_acquire);
}

static void inspectorPluginResolveDescrParent(ncclProfilerEventDescr_t* eDescr) {
  if (eDescr == nullptr || eDescr->parentObj == nullptr) return;
  if (!inspectorPluginIsAsyncHandle(eDescr->parentObj)) return;  // Already a real pointer
  inspectorAsyncHandle* handle = (inspectorAsyncHandle*)eDescr->parentObj;
  eDescr->parentObj = handle->realHandle.load(std::memory_order_acquire);  // nullptr if parent was filtered
}

static inspectorResult_t inspectorPluginAsyncHandlePoolInit(size_t size) {
  inspectorPluginAsyncHandlePoolReset();
  gAsyncHandlePool.size = size;
  gAsyncHandlePool.handles = new (std::nothrow) inspectorAsyncHandle[size]();
  gAsyncHandlePool.freeHandles = new (std::nothrow) inspectorAsyncHandle*[size]();
  if (gAsyncHandlePool.handles == nullptr || gAsyncHandlePool.freeHandles == nullptr) {
    delete[] gAsyncHandlePool.handles;
    delete[] gAsyncHandlePool.freeHandles;
    inspectorPluginAsyncHandlePoolReset();
    return inspectorMemoryError;
  }

  for (size_t i = 0; i < size; i++) {
    gAsyncHandlePool.handles[i].realHandle.store(nullptr, std::memory_order_relaxed);
    gAsyncHandlePool.freeHandles[i] = &gAsyncHandlePool.handles[i];
  }
  gAsyncHandlePool.head.store(0, std::memory_order_relaxed);
  gAsyncHandlePool.tail.store(size, std::memory_order_relaxed);
  return inspectorSuccess;
}

static void inspectorPluginAsyncHandlePoolFinalize() {
  delete[] gAsyncHandlePool.handles;
  delete[] gAsyncHandlePool.freeHandles;
  inspectorPluginAsyncHandlePoolReset();
}

static inspectorAsyncHandle* inspectorPluginAsyncHandleAlloc(uint64_t type) {
  if (gAsyncHandlePool.handles == nullptr) return nullptr;

  size_t head;
  while (true) {
    head = gAsyncHandlePool.head.load(std::memory_order_acquire);
    size_t tail = gAsyncHandlePool.tail.load(std::memory_order_acquire);
    if (head == tail) {
      gAsyncHandleAllocFails.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    if (gAsyncHandlePool.head.compare_exchange_weak(head, head + 1,
                                                    std::memory_order_acq_rel,
                                                    std::memory_order_acquire)) {
      break;
    }
  }

  inspectorAsyncHandle* handle = gAsyncHandlePool.freeHandles[head % gAsyncHandlePool.size];
  handle->type = type;
  handle->realHandle.store(nullptr, std::memory_order_release);
  return handle;
}

static void inspectorPluginAsyncHandleRelease(inspectorAsyncHandle* handle) {
  if (handle == nullptr || gAsyncHandlePool.freeHandles == nullptr) return;

  handle->realHandle.store(nullptr, std::memory_order_release);
  handle->type = 0;
  size_t tail = gAsyncHandlePool.tail.load(std::memory_order_relaxed);
  while (tail - gAsyncHandlePool.head.load(std::memory_order_acquire) >=
         gAsyncHandlePool.size) {
    gAsyncHandleReleaseSpins.fetch_add(1, std::memory_order_relaxed);
    sched_yield();
  }
  gAsyncHandlePool.freeHandles[tail % gAsyncHandlePool.size] = handle;
  gAsyncHandlePool.tail.store(tail + 1, std::memory_order_release);
}



static size_t inspectorPluginAsyncQueueSizeFromEnv() {
  const char* str = getenv("NCCL_INSPECTOR_ASYNC_QUEUE_SIZE");
  uint64_t size = str ? strtoull(str, nullptr, 0) : 1048576;
  if (size < 1024) size = 1024;
  if (size > SIZE_MAX) size = SIZE_MAX;
  return (size_t)size;
}

// Forward declarations for inspectorPluginStartEventConsumer
static bool inspectorShouldTrackColl(const ncclProfilerEventDescr_t* eDescr);
static bool inspectorShouldTrackP2p(const ncclProfilerEventDescr_t* eDescr);
static void inspectorPluginCollInfoInit(struct inspectorCollInfo **collInfo,
                                        ncclProfilerEventDescr_t *eDescr,
                                        struct inspectorCommInfo *commInfo,
                                        uint64_t tsOverride);
static void inspectorPluginP2pInfoInit(struct inspectorP2pInfo **p2pInfo,
                                       ncclProfilerEventDescr_t *eDescr,
                                       struct inspectorCommInfo *commInfo,
                                       uint64_t tsOverride);
static void inspectorPluginKernelChInfoInit(struct inspectorKernelChInfo **kernelChInfo,
                                            ncclProfilerEventDescr_t *eDescr);
static void inspectorPluginProxyOpInfoInit(struct inspectorProxyOpInfo **opInfo,
                                           ncclProfilerEventDescr_t *eDescr,
                                           struct inspectorCommInfo *commInfo,
                                           uint64_t tsOverride);
static void inspectorPluginProxyStepInfoInit(struct inspectorProxyStepInfo **stepInfo,
                                             ncclProfilerEventDescr_t *eDescr,
                                             struct inspectorCommInfo *commInfo,
                                             uint64_t tsOverride);
static void inspectorPluginProxyCtrlInfoInit(struct inspectorProxyCtrlInfo **ctrlInfo,
                                             struct inspectorCommInfo *commInfo,
                                             uint64_t tsOverride);

static bool inspectorIsGenericEventType(uint64_t type);
static void inspectorGenericEventPopulateDescr(struct inspectorGenericEventInfo *genEvent,
                                               ncclProfilerEventDescr_t *eDescr);
static void inspectorGenericEventEnqueueCompleted(struct inspectorGenericEventInfo *genInfo);

static bool inspectorShouldTrackProxyForParent(void* parentObj) {
  (void)parentObj;
  return true;  // No filtering — track all proxy events
}

static void inspectorPluginStartEventConsumer(inspectorAsyncEvent* event) {
  void* realHandle = nullptr;
  inspectorPluginResolveDescrParent(&event->eDescr);

  ncclProfilerEventDescr_t* eDescr = &event->eDescr;
  struct inspectorCommInfo* commInfo = event->commInfo;
  uint64_t ts = event->tsProducerUsec;

  if (eDescr->type == ncclProfileColl) {
    if (!inspectorShouldTrackColl(eDescr)) return;
    struct inspectorCollInfo *collEvent = nullptr;
    inspectorPluginCollInfoInit(&collEvent, eDescr, commInfo, ts);
    realHandle = collEvent;
  } else if (eDescr->type == ncclProfileP2p) {
    if (!enableNcclInspectorP2p) return;
    if (!inspectorShouldTrackP2p(eDescr)) return;
    struct inspectorP2pInfo *p2pEvent = nullptr;
    inspectorPluginP2pInfoInit(&p2pEvent, eDescr, commInfo, ts);
    realHandle = p2pEvent;
  } else if (eDescr->type == ncclProfileKernelCh) {
    struct inspectorKernelChInfo *kernelChEvent = nullptr;
    inspectorPluginKernelChInfoInit(&kernelChEvent, eDescr);
    if (kernelChEvent != nullptr && ts > 0) {
      kernelChEvent->tsStartUsec = ts;
    }
    realHandle = kernelChEvent;
  } else if (eDescr->type == ncclProfileProxyOp) {
    if (!enableNcclInspectorProxy || !inspectorIsDumpVerboseEnabled()) return;
    struct inspectorProxyOpInfo *proxyOpEvent = nullptr;
    inspectorPluginProxyOpInfoInit(&proxyOpEvent, eDescr, commInfo, ts);
    realHandle = proxyOpEvent;
  } else if (eDescr->type == ncclProfileProxyStep) {
    if (!enableNcclInspectorProxy || !inspectorIsDumpVerboseEnabled()) return;
    struct inspectorProxyStepInfo *proxyStepEvent = nullptr;
    inspectorPluginProxyStepInfoInit(&proxyStepEvent, eDescr, commInfo, ts);
    realHandle = proxyStepEvent;
  } else if (eDescr->type == ncclProfileProxyCtrl) {
    if (!enableNcclInspectorProxy || !inspectorIsDumpVerboseEnabled()) return;
    struct inspectorProxyCtrlInfo *proxyCtrlEvent = nullptr;
    inspectorPluginProxyCtrlInfoInit(&proxyCtrlEvent, commInfo, ts);
    realHandle = proxyCtrlEvent;
  } else if (inspectorIsGenericEventType(eDescr->type)) {
    struct inspectorGenericEventInfo *genEvent = new (std::nothrow) inspectorGenericEventInfo();
    if (genEvent != nullptr) {
      memset(genEvent, 0, sizeof(*genEvent));
      genEvent->type = eDescr->type;
      genEvent->tsStartUsec = ts;
      genEvent->tsStopUsec = 0;
      genEvent->commInfo = commInfo;
      inspectorGenericEventPopulateDescr(genEvent, eDescr);
      gGenericEventStartCount.fetch_add(1, std::memory_order_relaxed);
    }
    realHandle = genEvent;
  }

  if (event->asyncHandle != nullptr) {
    event->asyncHandle->realHandle.store(realHandle, std::memory_order_release);
  }
}

static void* inspectorPluginAsyncMain(void* arg) {
  inspectorAsyncQueue* queue = (inspectorAsyncQueue*)arg;
  uint32_t idleSpins = 0;

  while (true) {
    size_t head = queue->head.load(std::memory_order_relaxed);
    size_t tail = queue->tail.load(std::memory_order_acquire);
    if (head == tail) {
      if (queue->stop.load(std::memory_order_acquire)) {
        break;
      }
      if (++idleSpins < 1024) {
        sched_yield();
      } else {
        usleep(50);
      }
      continue;
    }

    idleSpins = 0;
    inspectorAsyncSlot* slot = &queue->slots[head % queue->size];
    if (!slot->ready.load(std::memory_order_acquire)) {
      sched_yield();
      continue;
    }
    inspectorAsyncEvent event = slot->event;
    slot->ready.store(false, std::memory_order_release);
    queue->processing.store(true, std::memory_order_release);
    queue->head.store(head + 1, std::memory_order_release);

    if (event.type == inspectorAsyncEventStart) {
      inspectorPluginStartEventConsumer(&event);
    } else if (event.type == inspectorAsyncEventStop) {
      void* realHandle = inspectorPluginResolveAsyncHandle(event.eHandle);
      if (realHandle != nullptr) {
        inspectorPluginStopEventSync(realHandle, event.tsProducerUsec);
      }
      if (inspectorPluginIsAsyncHandle(event.eHandle)) {
        inspectorAsyncHandle* handle = (inspectorAsyncHandle*)event.eHandle;
        if (handle->type != ncclProfileColl && handle->type != ncclProfileP2p) {
          inspectorPluginAsyncHandleRelease(handle);
        }
      }
    } else if (event.type == inspectorAsyncEventRecord) {
      void* realHandle = inspectorPluginResolveAsyncHandle(event.eHandle);
      if (realHandle != nullptr) {
        inspectorPluginRecordEventStateSync(realHandle, event.eState,
                                            event.hasArgs ? &event.eStateArgs : nullptr,
                                            event.tsProducerUsec);
      }
    }

    queue->processing.store(false, std::memory_order_release);
  }

  return nullptr;
}

static inspectorResult_t inspectorPluginAsyncInit() {
  inspectorPluginAsyncReset();

  // Read proxy message size filter threshold
  const char* minProxyEnv = getenv("NCCL_INSPECTOR_PROXY_MIN_MSG_SIZE");
  if (minProxyEnv) {
    minProxyMsgSizeBytes = strtoull(minProxyEnv, nullptr, 0);
  }

  gAsyncQueue.size = inspectorPluginAsyncQueueSizeFromEnv();
  gAsyncQueue.slots = new (std::nothrow) inspectorAsyncSlot[gAsyncQueue.size]();
  if (gAsyncQueue.slots == nullptr) {
    INFO_INSPECTOR("NCCL Inspector: failed to allocate async event queue");
    return inspectorMemoryError;
  }
  for (size_t i = 0; i < gAsyncQueue.size; i++) {
    gAsyncQueue.slots[i].ready.store(false, std::memory_order_relaxed);
  }

  if (inspectorPluginAsyncHandlePoolInit(gAsyncQueue.size) != inspectorSuccess) {
    delete[] gAsyncQueue.slots;
    inspectorPluginAsyncReset();
    INFO_INSPECTOR("NCCL Inspector: failed to allocate async handle pool");
    return inspectorMemoryError;
  }

  if (pthread_create(&gAsyncQueue.pthread, nullptr,
                     inspectorPluginAsyncMain, &gAsyncQueue) != 0) {
    inspectorPluginAsyncHandlePoolFinalize();
    delete[] gAsyncQueue.slots;
    inspectorPluginAsyncReset();
    INFO_INSPECTOR("NCCL Inspector: failed to start async event consumer thread");
    return inspectorPthreadError;
  }

  gAsyncQueue.initialized = true;
  INFO_INSPECTOR("NCCL Inspector: async event consumer started with queue size %lu",
                 (unsigned long)gAsyncQueue.size);
  return inspectorSuccess;
}

static void inspectorPluginAsyncFlush() {
  if (!gAsyncQueue.initialized) return;

  while (gAsyncQueue.head.load(std::memory_order_acquire) !=
             gAsyncQueue.tail.load(std::memory_order_acquire) ||
         gAsyncQueue.processing.load(std::memory_order_acquire)) {
    sched_yield();
  }
}

static void inspectorPluginAsyncFinalize() {
  if (!gAsyncQueue.initialized) return;

  inspectorPluginAsyncFlush();

  uint64_t enqCalls = gAsyncEnqueueCalls.load(std::memory_order_relaxed);
  uint64_t spinEvents = gAsyncEnqueueSpinEvents.load(std::memory_order_relaxed);
  uint64_t totalSpins = gAsyncEnqueueTotalSpins.load(std::memory_order_relaxed);
  uint64_t usleeps = gAsyncEnqueueUsleeps.load(std::memory_order_relaxed);
  uint64_t handleFails = gAsyncHandleAllocFails.load(std::memory_order_relaxed);
  uint64_t releaseSpins = gAsyncHandleReleaseSpins.load(std::memory_order_relaxed);
  fprintf(stderr,
    "[Inspector AsyncQueue Stats] "
    "enqueue_calls=%lu spin_events=%lu total_spins=%lu usleeps=%lu "
    "handle_alloc_fails=%lu handle_release_spins=%lu "
    "spin_rate=%.4f%% "
    "generic_starts=%lu generic_stops=%lu\n",
    (unsigned long)enqCalls, (unsigned long)spinEvents,
    (unsigned long)totalSpins, (unsigned long)usleeps,
    (unsigned long)handleFails, (unsigned long)releaseSpins,
    enqCalls > 0 ? 100.0 * spinEvents / enqCalls : 0.0,
    (unsigned long)gGenericEventStartCount.load(std::memory_order_relaxed),
    (unsigned long)gGenericEventStopCount.load(std::memory_order_relaxed));

  gAsyncQueue.stop.store(true, std::memory_order_release);

  pthread_join(gAsyncQueue.pthread, nullptr);
  delete[] gAsyncQueue.slots;
  inspectorPluginAsyncHandlePoolFinalize();
  inspectorPluginAsyncReset();
}

static bool inspectorPluginAsyncEnqueue(const inspectorAsyncEvent* event) {
  if (!gAsyncQueue.initialized || event == nullptr) {
    return false;
  }

  gAsyncEnqueueCalls.fetch_add(1, std::memory_order_relaxed);

  size_t tail;
  int spinCount = 0;
  bool didSpin = false;
  while (true) {
    if (gAsyncQueue.stop.load(std::memory_order_acquire)) {
      return false;
    }
    tail = gAsyncQueue.tail.load(std::memory_order_acquire);
    size_t head = gAsyncQueue.head.load(std::memory_order_acquire);
    if (tail - head >= gAsyncQueue.size) {
      // Queue full — spin with backoff
      didSpin = true;
      gAsyncEnqueueTotalSpins.fetch_add(1, std::memory_order_relaxed);
      if (++spinCount > 1024) {
        gAsyncEnqueueUsleeps.fetch_add(1, std::memory_order_relaxed);
        usleep(1);
        spinCount = 0;
      } else {
        sched_yield();
      }
      continue;
    }
    if (gAsyncQueue.tail.compare_exchange_weak(tail, tail + 1,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire)) {
      break;
    }
    // CAS contention — retry immediately
  }
  if (didSpin) {
    gAsyncEnqueueSpinEvents.fetch_add(1, std::memory_order_relaxed);
  }

  inspectorAsyncSlot* slot = &gAsyncQueue.slots[tail % gAsyncQueue.size];
  slot->event = *event;
  slot->ready.store(true, std::memory_order_release);
  return true;
}

static bool inspectorPluginAsyncSerializesProxyEvents() {
  return gAsyncQueue.initialized;
}


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
    // Initialize object pools
    g_collPool = new ObjectPool<inspectorCollInfo>();
    g_p2pPool = new ObjectPool<inspectorP2pInfo>();
    g_proxyOpPool = new ObjectPool<inspectorProxyOpInfo>();
    g_proxyStepPool = new ObjectPool<inspectorProxyStepInfo>();
    g_proxyCtrlPool = new ObjectPool<inspectorProxyCtrlInfo>();
    g_collPool->init(256);
    g_p2pPool->init(256);
    g_proxyOpPool->init(1024);
    g_proxyStepPool->init(8192);
    g_proxyCtrlPool->init(1024);

    res = inspectorPluginAsyncInit();
    if (res != inspectorSuccess) {
      INFO_INSPECTOR("Inspector Async Init Failed %s:%d -> error %d: %s",
                     __FILE__, __LINE__, res,
                     inspectorErrorString(res));
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
  // Enable ALL event types for comprehensive profiling overhead measurement
  *eActivationMask = ncclProfileColl | ncclProfileKernelCh
    | ncclProfileP2p
    | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileProxyCtrl
    | ncclProfileGroup | ncclProfileGroupApi | ncclProfileCollApi
    | ncclProfileP2pApi | ncclProfileKernelLaunch | ncclProfileNetPlugin;

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
  inspectorPluginAsyncFlush();
  inspectorDelComm((struct inspectorCommInfo *)context);
  pthread_mutex_lock(&gLock);
  if (--gInitialized == 0) {
    inspectorPluginAsyncFinalize();
    inspectorGlobalFinalize();
    // Finalize object pools
    delete g_collPool; g_collPool = nullptr;
    delete g_p2pPool; g_p2pPool = nullptr;
    delete g_proxyOpPool; g_proxyOpPool = nullptr;
    delete g_proxyStepPool; g_proxyStepPool = nullptr;
    delete g_proxyCtrlPool; g_proxyCtrlPool = nullptr;
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
  // proxyOps (std::vector) is cleaned up automatically by pool release destructor
  g_collPool->release(collInfo);
}

static void inspectorUpdateCommOpInfo(struct inspectorCommInfo *commInfo,
                                      struct inspectorCompletedOpInfo *completedOp) {
  inspectorLockWr(&commInfo->guard);
  inspectorComputeOpBw(commInfo, completedOp);
  if (completedOp->isP2p) {
    commInfo->completedP2pRing.enqueue(std::move(*completedOp));
    commInfo->dump_p2p = commInfo->completedP2pRing.nonEmpty();
  } else {
    commInfo->completedCollRing.enqueue(std::move(*completedOp));
    commInfo->dump_coll = commInfo->completedCollRing.nonEmpty();
  }
  inspectorUnlockRWLock(&commInfo->guard);
}

static inspectorResult_t inspectorUpdateCommProxyInfo(struct inspectorCommInfo *commInfo,
                                                      struct inspectorCompletedProxyEventInfo *completedProxy) {
  inspectorLockWr(&commInfo->guard);
  commInfo->completedProxyRing.enqueue(std::move(*completedProxy));
  commInfo->dump_proxy = commInfo->completedProxyRing.nonEmpty();
  inspectorUnlockRWLock(&commInfo->guard);
  return inspectorSuccess;
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
  // proxyOps (std::vector) is cleaned up automatically by pool release destructor
  g_p2pPool->release(p2pInfo);
}

// With STL vectors, copy/move/cleanup are handled by value semantics.
// These thin wrappers remain for code structure clarity.

static void inspectorProxyOpRecordListAppendMove(
    inspectorProxyOpRecordList* list,
    struct inspectorCompletedProxyEventInfo* record) {
  if (list == nullptr || record == nullptr) return;
  list->push_back(std::move(*record));
}

void inspectorCompletedOpInfoCleanup(void* entry) {
  // No-op: std::vector members self-destruct. Retained for API compatibility.
  (void)entry;
}

inspectorResult_t inspectorCompletedOpInfoCopy(void* dst, const void* src) {
  if (dst == nullptr || src == nullptr) return inspectorMemoryError;
  *static_cast<inspectorCompletedOpInfo*>(dst) =
    *static_cast<const inspectorCompletedOpInfo*>(src);
  return inspectorSuccess;
}

inspectorResult_t inspectorCompletedOpInfoMove(void* dst, const void* src) {
  if (dst == nullptr || src == nullptr) return inspectorMemoryError;
  *static_cast<inspectorCompletedOpInfo*>(dst) =
    std::move(*const_cast<inspectorCompletedOpInfo*>(static_cast<const inspectorCompletedOpInfo*>(src)));
  return inspectorSuccess;
}

void inspectorCompletedProxyEventInfoCleanup(void* entry) {
  // No-op: std::vector members self-destruct.
  (void)entry;
}

inspectorResult_t inspectorCompletedProxyEventInfoCopy(void* dst, const void* src) {
  if (dst == nullptr || src == nullptr) return inspectorMemoryError;
  *static_cast<inspectorCompletedProxyEventInfo*>(dst) =
    *static_cast<const inspectorCompletedProxyEventInfo*>(src);
  return inspectorSuccess;
}

inspectorResult_t inspectorCompletedProxyEventInfoMove(void* dst, const void* src) {
  if (dst == nullptr || src == nullptr) return inspectorMemoryError;
  *static_cast<inspectorCompletedProxyEventInfo*>(dst) =
    std::move(*const_cast<inspectorCompletedProxyEventInfo*>(static_cast<const inspectorCompletedProxyEventInfo*>(src)));
  return inspectorSuccess;
}

void inspectorProxyOpRecordListCleanup(inspectorProxyOpRecordList* proxyOps) {
  // No-op: std::vector destructor handles element cleanup.
  if (proxyOps) proxyOps->clear();
}

static void inspectorCompletedProxyEventInfoInit(
    struct inspectorCompletedProxyEventInfo* proxy) {
  if (proxy == nullptr) return;
  *proxy = inspectorCompletedProxyEventInfo{};
}

static inspectorResult_t inspectorRecordProxyEventState(inspectorProxyEventStateList* states,
                                           ncclProfilerEventState_t eState,
                                           ncclProfilerEventStateArgs_t* eStateArgs,
                                           uint64_t tsOverride = 0) {
  if (states == nullptr) return inspectorMemoryError;

  // State compression: if last recorded state has the same enum value,
  // update its data fields instead of appending a new entry.
  if (!states->empty()) {
    struct inspectorProxyEventStateInfo* last = &states->back();
    if (last != nullptr && last->state == (int)eState) {
      // Same state repeating — update data fields, skip allocation
      switch (eState) {
      case ncclProfilerProxyStepSendWait:
      case ncclProfilerProxyStepRecvFlushWait:
        if (eStateArgs != nullptr) {
          last->transSize = eStateArgs->proxyStep.transSize;
        }
        break;
      case ncclProfilerProxyCtrlIdle:
      case ncclProfilerProxyCtrlActive:
      case ncclProfilerProxyCtrlSleep:
      case ncclProfilerProxyCtrlWakeup:
      case ncclProfilerProxyCtrlAppend:
      case ncclProfilerProxyCtrlAppendEnd:
        if (eStateArgs != nullptr) {
          last->appendedProxyOps = eStateArgs->proxyCtrl.appendedProxyOps;
        }
        break;
      default:
        break;
      }
      return inspectorSuccess;
    }
  }

  states->emplace_back();
  struct inspectorProxyEventStateInfo* state = &states->back();
  state->state = (int)eState;
  state->tsUsec = tsOverride > 0 ? tsOverride : inspectorGetTime();
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
  // Move ownership: opInfo is cleaned up immediately after this call
  completedProxy->states = std::move(opInfo->states);
  completedProxy->steps = std::move(opInfo->steps);
  return inspectorSuccess;
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
  // Move ownership: ctrlInfo is cleaned up immediately after this call
  completedProxy->states = std::move(ctrlInfo->states);
  return inspectorSuccess;
}

static void inspectorPluginProxyOpInfoCleanup(struct inspectorProxyOpInfo *opInfo) {
  // states and steps are std::vectors - cleaned up by pool release destructor
  g_proxyOpPool->release(opInfo);
}

static void inspectorPluginProxyStepInfoCleanup(struct inspectorProxyStepInfo *stepInfo) {
  // states is a std::vector - cleaned up by pool release destructor
  g_proxyStepPool->release(stepInfo);
}

static void inspectorPluginProxyCtrlInfoCleanup(struct inspectorProxyCtrlInfo *ctrlInfo) {
  // states is a std::vector - cleaned up by pool release destructor
  g_proxyCtrlPool->release(ctrlInfo);
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
                                           struct inspectorCommInfo *commInfo,
                                           uint64_t tsOverride = 0) {
  struct inspectorProxyOpInfo *opInfoPtr =
    g_proxyOpPool->alloc();
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
  opInfoPtr->tsStartUsec = tsOverride > 0 ? tsOverride : inspectorGetTime();

  if (eDescr->parentObj != nullptr && eDescr->proxyOp.pid == getpid()) {
    uint64_t parentType = *(uint64_t*)eDescr->parentObj;
    if (parentType == ncclProfileColl) {
      struct inspectorCollInfo* parent = (struct inspectorCollInfo*)eDescr->parentObj;
      parent->nProxyOpsStarted += 1;
      inspectorPluginCollInfoRef(parent);
      opInfoPtr->parentType = ncclProfileColl;
      opInfoPtr->parentObj = parent;
      opInfoPtr->commInfo = parent->commInfo;
    } else if (parentType == ncclProfileP2p) {
      struct inspectorP2pInfo* parent = (struct inspectorP2pInfo*)eDescr->parentObj;
      parent->nProxyOpsStarted += 1;
      inspectorPluginP2pInfoRef(parent);
      opInfoPtr->parentType = ncclProfileP2p;
      opInfoPtr->parentObj = parent;
      opInfoPtr->commInfo = parent->commInfo;
    }
  }

  *opInfo = opInfoPtr;
}

static void inspectorPluginProxyStepInfoInit(struct inspectorProxyStepInfo **stepInfo,
                                             ncclProfilerEventDescr_t *eDescr,
                                             struct inspectorCommInfo *commInfo,
                                             uint64_t tsOverride = 0) {
  struct inspectorProxyStepInfo *stepInfoPtr =
    g_proxyStepPool->alloc();
  if (stepInfoPtr == nullptr) {
    INFO_INSPECTOR("Inspector: Failed to allocate memory for proxy step info structure");
    *stepInfo = nullptr;
    return;
  }

  struct inspectorProxyOpInfo *parent = nullptr;
  if (eDescr->parentObj != nullptr && *(uint64_t*)eDescr->parentObj == ncclProfileProxyOp) {
    parent = (struct inspectorProxyOpInfo*)eDescr->parentObj;
    parent->refCount += 1;
    stepInfoPtr->isSend = parent->isSend;
    stepInfoPtr->commInfo = parent->commInfo;
  } else {
    stepInfoPtr->isSend = -1;
    stepInfoPtr->commInfo = commInfo;
  }

  stepInfoPtr->type = ncclProfileProxyStep;
  stepInfoPtr->parent = parent;
  stepInfoPtr->proxyId = __atomic_add_fetch(&commInfo->proxySeqNum, 1, __ATOMIC_RELAXED);
  stepInfoPtr->rank = eDescr->rank;
  stepInfoPtr->step = eDescr->proxyStep.step;
  stepInfoPtr->tsStartUsec = tsOverride > 0 ? tsOverride : inspectorGetTime();
  *stepInfo = stepInfoPtr;
}

static void inspectorPluginProxyCtrlInfoInit(struct inspectorProxyCtrlInfo **ctrlInfo,
                                             struct inspectorCommInfo *commInfo,
                                             uint64_t tsOverride = 0) {
  struct inspectorProxyCtrlInfo *ctrlInfoPtr =
    g_proxyCtrlPool->alloc();
  if (ctrlInfoPtr == nullptr) {
    INFO_INSPECTOR("Inspector: Failed to allocate memory for proxy ctrl info structure");
    *ctrlInfo = nullptr;
    return;
  }

  ctrlInfoPtr->type = ncclProfileProxyCtrl;
  ctrlInfoPtr->commInfo = commInfo;
  ctrlInfoPtr->proxyId = __atomic_add_fetch(&commInfo->proxySeqNum, 1, __ATOMIC_RELAXED);
  ctrlInfoPtr->tsStartUsec = tsOverride > 0 ? tsOverride : inspectorGetTime();
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
                                        struct inspectorCommInfo *commInfo,
                                        uint64_t tsOverride = 0) {
  struct inspectorCollInfo *collInfoPtr = g_collPool->alloc();
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
  collInfoPtr->tsStartUsec = tsOverride > 0 ? tsOverride : inspectorGetTime();
  collInfoPtr->msgSizeBytes =
    ncclTypeSize(inspectorStringToDatatype(eDescr->coll.datatype)) * eDescr->coll.count;


  collInfoPtr->commInfo = commInfo;
  collInfoPtr->collEvtTrk.sn = 0;
  collInfoPtr->collEvtTrk.nChannels = collInfoPtr->nChannels;
  inspectorRecordEventTrace(collInfoPtr->collEvtTrk.evntTrace,
                            NCCL_INSP_EVT_TRK_OP_START, collInfoPtr);

  *collInfo = collInfoPtr;
}

static void inspectorPluginP2pInfoInit(struct inspectorP2pInfo **p2pInfo,
                                       ncclProfilerEventDescr_t *eDescr,
                                       struct inspectorCommInfo *commInfo,
                                       uint64_t tsOverride = 0) {
  struct inspectorP2pInfo *p2pInfoPtr = g_p2pPool->alloc();
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
  p2pInfoPtr->tsStartUsec = tsOverride > 0 ? tsOverride : inspectorGetTime();
  p2pInfoPtr->msgSizeBytes =
    ncclTypeSize(inspectorStringToDatatype(eDescr->p2p.datatype)) * eDescr->p2p.count;

  p2pInfoPtr->commInfo = commInfo;
  p2pInfoPtr->sn = __atomic_add_fetch(&commInfo->p2pSeqNum, 1, __ATOMIC_RELAXED);
  p2pInfoPtr->p2pEvtTrk.nChannels = p2pInfoPtr->nChannels;
  p2pInfoPtr->p2pEvtTrk.sn = p2pInfoPtr->sn;
  inspectorRecordP2pEventTrace(p2pInfoPtr->p2pEvtTrk.evntTrace,
                               NCCL_INSP_EVT_TRK_OP_START, p2pInfoPtr);

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
}

static void inspectorPluginKernelChInfoInitP2p(struct inspectorKernelChInfo **kernelChInfo,
                                               ncclProfilerEventDescr_t *eDescr,
                                               struct inspectorP2pInfo *p2pInfo) {
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
static ncclResult_t inspectorPluginStartEventSync(void* context,
                                                  void** eHandle,
                                                  ncclProfilerEventDescr_t* eDescr) {
  if (context == nullptr || eDescr == nullptr) {
    INFO(NCCL_INIT, "Profiler/Plugin: context/eDescr NULL for start event %s", __func__);
    return ncclSuccess;
  }
  *eHandle = nullptr;
  inspectorPluginResolveDescrParent(eDescr);
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
    if (!inspectorShouldTrackProxyForParent(eDescr->parentObj)) return ncclSuccess;
    struct inspectorProxyOpInfo *proxyOpEvent = nullptr;
    struct inspectorCommInfo *commInfoCtx = (struct inspectorCommInfo*)context;
    inspectorPluginProxyOpInfoInit(&proxyOpEvent, eDescr, commInfoCtx);
    *eHandle = proxyOpEvent;
  } else if (eDescr->type == ncclProfileProxyStep) {
    if (!enableNcclInspectorProxy || !inspectorIsDumpVerboseEnabled()) return ncclSuccess;
    if (eDescr->parentObj == nullptr) return ncclSuccess;  // Parent ProxyOp was filtered
    struct inspectorProxyStepInfo *proxyStepEvent = nullptr;
    struct inspectorCommInfo *commInfoCtx = (struct inspectorCommInfo*)context;
    inspectorPluginProxyStepInfoInit(&proxyStepEvent, eDescr, commInfoCtx);
    *eHandle = proxyStepEvent;
  } else if (eDescr->type == ncclProfileProxyCtrl) {
    if (!enableNcclInspectorProxy || !inspectorIsDumpVerboseEnabled()) return ncclSuccess;
    if (eDescr->parentObj == nullptr) return ncclSuccess;  // Parent was filtered
    struct inspectorProxyCtrlInfo *proxyCtrlEvent = nullptr;
    struct inspectorCommInfo *commInfoCtx = (struct inspectorCommInfo*)context;
    inspectorPluginProxyCtrlInfoInit(&proxyCtrlEvent, commInfoCtx);
    *eHandle = proxyCtrlEvent;
  } else if (inspectorIsGenericEventType(eDescr->type)) {
    struct inspectorGenericEventInfo *genEvent = new (std::nothrow) inspectorGenericEventInfo();
    if (genEvent != nullptr) {
      memset(genEvent, 0, sizeof(*genEvent));
      genEvent->type = eDescr->type;
      genEvent->tsStartUsec = inspectorGetTime();
      genEvent->tsStopUsec = 0;
      genEvent->commInfo = (struct inspectorCommInfo*)context;
      inspectorGenericEventPopulateDescr(genEvent, eDescr);
      gGenericEventStartCount.fetch_add(1, std::memory_order_relaxed);
    }
    *eHandle = genEvent;
  } else {
    return ncclSuccess;
  }
  return ncclSuccess;
}

static bool inspectorIsGenericEventType(uint64_t type) {
  return type == ncclProfileGroup || type == ncclProfileGroupApi ||
         type == ncclProfileCollApi || type == ncclProfileP2pApi ||
         type == ncclProfileKernelLaunch || type == ncclProfileNetPlugin;
}

static void inspectorGenericEventPopulateDescr(struct inspectorGenericEventInfo *genEvent,
                                               ncclProfilerEventDescr_t *eDescr) {
  genEvent->rank = eDescr->rank;
  genEvent->funcName[0] = '\0';
  genEvent->datatype[0] = '\0';
  genEvent->count = 0;
  genEvent->root = -1;
  genEvent->groupDepth = 0;
  genEvent->graphCaptured = false;
  genEvent->netPluginId = 0;

  if (eDescr->type == ncclProfileCollApi) {
    if (eDescr->collApi.func) {
      strncpy(genEvent->funcName, eDescr->collApi.func, sizeof(genEvent->funcName) - 1);
      genEvent->funcName[sizeof(genEvent->funcName) - 1] = '\0';
    }
    genEvent->count = eDescr->collApi.count;
    if (eDescr->collApi.datatype) {
      strncpy(genEvent->datatype, eDescr->collApi.datatype, sizeof(genEvent->datatype) - 1);
      genEvent->datatype[sizeof(genEvent->datatype) - 1] = '\0';
    }
    genEvent->root = eDescr->collApi.root;
    genEvent->graphCaptured = eDescr->collApi.graphCaptured;
  } else if (eDescr->type == ncclProfileP2pApi) {
    if (eDescr->p2pApi.func) {
      strncpy(genEvent->funcName, eDescr->p2pApi.func, sizeof(genEvent->funcName) - 1);
      genEvent->funcName[sizeof(genEvent->funcName) - 1] = '\0';
    }
    genEvent->count = eDescr->p2pApi.count;
    if (eDescr->p2pApi.datatype) {
      strncpy(genEvent->datatype, eDescr->p2pApi.datatype, sizeof(genEvent->datatype) - 1);
      genEvent->datatype[sizeof(genEvent->datatype) - 1] = '\0';
    }
    genEvent->graphCaptured = eDescr->p2pApi.graphCaptured;
  } else if (eDescr->type == ncclProfileGroupApi) {
    genEvent->groupDepth = eDescr->groupApi.groupDepth;
    genEvent->graphCaptured = eDescr->groupApi.graphCaptured;
  } else if (eDescr->type == ncclProfileNetPlugin) {
    genEvent->netPluginId = eDescr->netPlugin.id;
  }
}

static void inspectorGenericEventEnqueueCompleted(struct inspectorGenericEventInfo *genInfo) {
  struct inspectorCommInfo *commInfo = genInfo->commInfo;
  if (commInfo == nullptr) return;

  struct inspectorCompletedGenericEventInfo completed;
  memset(&completed, 0, sizeof(completed));
  completed.type = genInfo->type;
  completed.tsStartUsec = genInfo->tsStartUsec;
  completed.tsStopUsec = genInfo->tsStopUsec;
  completed.rank = genInfo->rank;
  memcpy(completed.funcName, genInfo->funcName, sizeof(completed.funcName));
  completed.count = genInfo->count;
  memcpy(completed.datatype, genInfo->datatype, sizeof(completed.datatype));
  completed.root = genInfo->root;
  completed.groupDepth = genInfo->groupDepth;
  completed.graphCaptured = genInfo->graphCaptured;
  completed.netPluginId = genInfo->netPluginId;

  inspectorLockWr(&commInfo->guard);
  commInfo->completedGenericRing.enqueue(std::move(completed));
  commInfo->dump_generic = commInfo->completedGenericRing.nonEmpty();
  inspectorUnlockRWLock(&commInfo->guard);
}

static bool inspectorPluginShouldTrackStartEvent(ncclProfilerEventDescr_t* eDescr) {
  if (eDescr == nullptr) return false;
  if (eDescr->type == ncclProfileColl) {
    return inspectorShouldTrackColl(eDescr);
  } else if (eDescr->type == ncclProfileP2p) {
    return inspectorShouldTrackP2p(eDescr);
  } else if (eDescr->type == ncclProfileKernelCh) {
    return eDescr->parentObj != nullptr;
  } else if (eDescr->type == ncclProfileProxyOp ||
             eDescr->type == ncclProfileProxyStep ||
             eDescr->type == ncclProfileProxyCtrl) {
    return true;
  } else if (inspectorIsGenericEventType(eDescr->type)) {
    return true;
  }
  return false;
}

__hidden ncclResult_t inspectorPluginStartEvent(void* context,
                                                void** eHandle,
                                                ncclProfilerEventDescr_t* eDescr) {
  if (eHandle == nullptr) return ncclSuccess;
  *eHandle = nullptr;
  if (context == nullptr || eDescr == nullptr) {
    INFO(NCCL_INIT, "Profiler/Plugin: context/eDescr NULL for start event %s", __func__);
    return ncclSuccess;
  }

  if (!inspectorPluginShouldTrackStartEvent(eDescr)) {
    return ncclSuccess;
  }

  inspectorAsyncHandle* handle = inspectorPluginAsyncHandleAlloc(eDescr->type);
  if (handle != nullptr) {
    inspectorAsyncEvent event;
    memset(&event, 0, sizeof(event));
    event.type = inspectorAsyncEventStart;
    event.context = context;
    event.asyncHandle = handle;
    event.eDescr = *eDescr;
    event.tsProducerUsec = inspectorGetTime();
    event.commInfo = (struct inspectorCommInfo*)context;
    if (inspectorPluginAsyncEnqueue(&event)) {
      *eHandle = handle;
      return ncclSuccess;
    }
    inspectorPluginAsyncHandleRelease(handle);
  }
  // Event dropped (handle pool or queue exhausted)
  return ncclSuccess;
}

static ncclResult_t inspectorPluginStopEventColl(struct inspectorCollInfo *collInfo, uint64_t tsOverride = 0) {
  (void)tsOverride;
  struct inspectorCompletedOpInfo completedOp{};
  struct inspectorCommInfo *commInfo = nullptr;
  bool doCommUpdate = false;

  inspectorRecordEventTrace(collInfo->collEvtTrk.evntTrace,
                            NCCL_INSP_EVT_TRK_OP_STOP,
                            collInfo);
  collInfo->stopped = true;
  commInfo = collInfo->commInfo;
  inspectorResult_t completeRes =
    inspectorPluginCollMaybeCompleteLocked(collInfo, &completedOp, &doCommUpdate);
  inspectorResult_t res = inspectorPluginCollInfoDeRef(collInfo);
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

static ncclResult_t inspectorPluginStopEventP2p(struct inspectorP2pInfo *p2pInfo, uint64_t tsOverride = 0) {
  (void)tsOverride;
  struct inspectorCompletedOpInfo completedOp{};
  struct inspectorCommInfo *commInfo = nullptr;
  bool doCommUpdate = false;

  inspectorRecordP2pEventTrace(p2pInfo->p2pEvtTrk.evntTrace,
                               NCCL_INSP_EVT_TRK_OP_STOP,
                               p2pInfo);
  p2pInfo->stopped = true;
  commInfo = p2pInfo->commInfo;
  inspectorResult_t completeRes =
    inspectorPluginP2pMaybeCompleteLocked(p2pInfo, &completedOp, &doCommUpdate);
  inspectorResult_t res = inspectorPluginP2pInfoDeRef(p2pInfo);
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
  struct inspectorCompletedOpInfo completedOp{};
  bool needsCleanup = false;
  bool doCommUpdate = false;

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
  struct inspectorCompletedOpInfo completedOp{};
  bool needsCleanup = false;
  bool doCommUpdate = false;

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

  *record = inspectorProxyStepRecordInfo{};
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
  // Move states from stepInfo (stepInfo will be cleaned up after this)
  record->states = std::move(stepInfo->states);
  return inspectorSuccess;
}

static inspectorResult_t inspectorProxyOpAppendStepRecordLocked(
    struct inspectorProxyOpInfo* opInfo,
    struct inspectorProxyStepRecordInfo* record) {
  if (opInfo == nullptr || record == nullptr) return inspectorMemoryError;

  opInfo->steps.push_back(std::move(*record));
  return inspectorSuccess;
}

static inspectorResult_t inspectorPluginAppendCompletedProxyToParent(
    struct inspectorProxyOpInfo* opInfo,
    struct inspectorCompletedProxyEventInfo* completedProxy,
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

    inspectorProxyOpRecordListAppendMove(&parent->proxyOps, completedProxy);
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
  } else if (opInfo->parentType == ncclProfileP2p) {
    struct inspectorP2pInfo* parent = (struct inspectorP2pInfo*)opInfo->parentObj;
    if (parent == nullptr) return inspectorSuccess;

    inspectorProxyOpRecordListAppendMove(&parent->proxyOps, completedProxy);
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
  }

  return res;
}

static ncclResult_t inspectorPluginStopEventProxyOp(struct inspectorProxyOpInfo *opInfo, uint64_t tsOverride = 0) {
  struct inspectorCompletedProxyEventInfo completedProxy;
  struct inspectorCompletedOpInfo completedParentOp{};
  struct inspectorCommInfo *commInfo = nullptr;
  struct inspectorCommInfo *parentCommInfo = nullptr;
  bool doCommUpdate = false;
  bool doParentCommUpdate = false;
  bool needsCleanup = false;
  bool needsParentCleanup = false;
  inspectorCompletedProxyEventInfoInit(&completedProxy);
  completedParentOp = inspectorCompletedOpInfo{};

  opInfo->tsCompletedUsec = tsOverride > 0 ? tsOverride : inspectorGetTime();
  opInfo->stopped = true;
  opInfo->refCount -= 1;
  inspectorResult_t res =
    inspectorPluginProxyOpMaybeCompleteLocked(opInfo, &completedProxy, &commInfo,
                                              &doCommUpdate, &needsCleanup);
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

static ncclResult_t inspectorPluginStopEventProxyStep(struct inspectorProxyStepInfo *stepInfo, uint64_t tsOverride = 0) {
  struct inspectorCompletedProxyEventInfo completedProxyOp{};
  struct inspectorCompletedOpInfo completedParentOp{};
  struct inspectorProxyStepRecordInfo stepRecord{};
  struct inspectorCommInfo *opCommInfo = nullptr;
  struct inspectorCommInfo *parentCommInfo = nullptr;
  bool doOpCommUpdate = false;
  bool doParentCommUpdate = false;
  bool needsOpCleanup = false;
  bool needsParentCleanup = false;

  stepInfo->tsCompletedUsec = tsOverride > 0 ? tsOverride : inspectorGetTime();
  inspectorResult_t recordRes = inspectorProxyStepRecordFromStepLocked(&stepRecord, stepInfo);
  if (recordRes != inspectorSuccess) {
    INFO_INSPECTOR("Inspector: Failed to copy proxy step record: %s",
                   inspectorErrorString(recordRes));
  }

  if (stepInfo->parent != nullptr) {
    struct inspectorProxyOpInfo *parent = stepInfo->parent;
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

  inspectorCompletedProxyEventInfoCleanup(&completedProxyOp);
  inspectorPluginProxyStepInfoCleanup(stepInfo);
  return ncclSuccess;
}

static ncclResult_t inspectorPluginStopEventProxyCtrl(struct inspectorProxyCtrlInfo *ctrlInfo, uint64_t tsOverride = 0) {
  struct inspectorCompletedProxyEventInfo completedProxy;
  struct inspectorCommInfo *commInfo = nullptr;
  inspectorCompletedProxyEventInfoInit(&completedProxy);

  ctrlInfo->tsCompletedUsec = tsOverride > 0 ? tsOverride : inspectorGetTime();
  commInfo = ctrlInfo->commInfo;
  inspectorResult_t res = inspectorCompletedProxyFromCtrlLocked(&completedProxy, ctrlInfo);
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
  if (needsCleanup) {
    inspectorPluginCollInfoCleanup(collInfo);
  }
  return ncclSuccess;
}

static ncclResult_t inspectorPluginRecordEventStateKernelChP2p(struct inspectorKernelChInfo *kernelChInfo,
                                                               struct inspectorP2pInfo *p2pInfo,
                                                               ncclProfilerEventStateArgs_t* eStateArgs) {
  bool needsCleanup = false;
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
                                                           ncclProfilerEventStateArgs_t* eStateArgs,
                                                           uint64_t tsOverride = 0) {
  inspectorResult_t res = inspectorRecordProxyEventState(&opInfo->states, eState, eStateArgs, tsOverride);
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
                                                             ncclProfilerEventStateArgs_t* eStateArgs,
                                                             uint64_t tsOverride = 0) {
  if (eState == ncclProfilerProxyStepSendPeerWait_v4) return ncclSuccess;

  size_t transSize = 0;
  if (eStateArgs != nullptr && inspectorProxyStepStateCarriesTransSize(eState)) {
    transSize = eStateArgs->proxyStep.transSize;
  }

  inspectorResult_t res = inspectorRecordProxyEventState(&stepInfo->states, eState, eStateArgs, tsOverride);
  stepInfo->transSize += transSize;
  if (res != inspectorSuccess) {
    INFO_INSPECTOR("Inspector: Failed to record proxy step state: %s",
                   inspectorErrorString(res));
  }

  if (transSize != 0 && stepInfo->parent != nullptr) {
    stepInfo->parent->transSize += transSize;
  }
  return ncclSuccess;
}

static ncclResult_t inspectorPluginRecordEventStateProxyCtrl(struct inspectorProxyCtrlInfo *ctrlInfo,
                                                             ncclProfilerEventState_t eState,
                                                             ncclProfilerEventStateArgs_t* eStateArgs,
                                                             uint64_t tsOverride = 0) {
  inspectorResult_t res = inspectorRecordProxyEventState(&ctrlInfo->states, eState, eStateArgs, tsOverride);
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
static ncclResult_t inspectorPluginStopEventSync(void *eHandle, uint64_t tsOverride) {
  if (eHandle == nullptr) {
    INFO(NCCL_INIT,
         "Profiler/Plugin: Event Handle NULL for start event %s", __func__);
    return ncclSuccess;
  }

  uint64_t type = *(uint64_t *)eHandle;
  if (type == ncclProfileColl) {
    struct inspectorCollInfo *collInfo = (struct inspectorCollInfo *)eHandle;
    return inspectorPluginStopEventColl(collInfo, tsOverride);
  } else if (type == ncclProfileP2p) {
    struct inspectorP2pInfo *p2pInfo = (struct inspectorP2pInfo *)eHandle;
    return inspectorPluginStopEventP2p(p2pInfo, tsOverride);
  } else if (type == ncclProfileKernelCh) {
    struct inspectorKernelChInfo *kernelChInfo
      = (struct inspectorKernelChInfo *)eHandle;
    return inspectorPluginStopEventKernelCh(kernelChInfo);
  } else if (type == ncclProfileProxyOp) {
    struct inspectorProxyOpInfo *proxyOpInfo =
      (struct inspectorProxyOpInfo *)eHandle;
    return inspectorPluginStopEventProxyOp(proxyOpInfo, tsOverride);
  } else if (type == ncclProfileProxyStep) {
    struct inspectorProxyStepInfo *proxyStepInfo =
      (struct inspectorProxyStepInfo *)eHandle;
    return inspectorPluginStopEventProxyStep(proxyStepInfo, tsOverride);
  } else if (type == ncclProfileProxyCtrl) {
    struct inspectorProxyCtrlInfo *proxyCtrlInfo =
      (struct inspectorProxyCtrlInfo *)eHandle;
    return inspectorPluginStopEventProxyCtrl(proxyCtrlInfo, tsOverride);
  } else if (inspectorIsGenericEventType(type)) {
    struct inspectorGenericEventInfo *genInfo =
      (struct inspectorGenericEventInfo *)eHandle;
    genInfo->tsStopUsec = tsOverride > 0 ? tsOverride : inspectorGetTime();
    gGenericEventStopCount.fetch_add(1, std::memory_order_relaxed);
    inspectorGenericEventEnqueueCompleted(genInfo);
    delete genInfo;
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
static ncclResult_t inspectorPluginRecordEventStateSync(void* eHandle,
                                                        ncclProfilerEventState_t eState,
                                                        ncclProfilerEventStateArgs_t* eStateArgs,
                                                        uint64_t tsOverride) {
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
    return inspectorPluginRecordEventStateProxyOp(proxyOpInfo, eState, eStateArgs, tsOverride);

  } else if (type == ncclProfileProxyStep) {
    struct inspectorProxyStepInfo *proxyStepInfo =
      (struct inspectorProxyStepInfo *)eHandle;
    return inspectorPluginRecordEventStateProxyStep(proxyStepInfo, eState, eStateArgs, tsOverride);

  } else if (type == ncclProfileProxyCtrl) {
    struct inspectorProxyCtrlInfo *proxyCtrlInfo =
      (struct inspectorProxyCtrlInfo *)eHandle;
    return inspectorPluginRecordEventStateProxyCtrl(proxyCtrlInfo, eState, eStateArgs, tsOverride);

  } else if (inspectorIsGenericEventType(type)) {
    // Generic events: accept state transitions with no-op (overhead measurement only)
    (void)eState;
    (void)eStateArgs;
    (void)tsOverride;
  }
  return ncclSuccess;
}

__hidden ncclResult_t inspectorPluginStopEvent(void *eHandle) {
  if (eHandle == nullptr) {
    INFO(NCCL_INIT,
         "Profiler/Plugin: Event Handle NULL for stop event %s", __func__);
    return ncclSuccess;
  }

  inspectorAsyncEvent event;
  memset(&event, 0, sizeof(event));
  event.type = inspectorAsyncEventStop;
  event.eHandle = eHandle;
  event.tsProducerUsec = inspectorGetTime();
  inspectorPluginAsyncEnqueue(&event);
  return ncclSuccess;
}

__hidden ncclResult_t inspectorPluginRecordEventState(void* eHandle,
                                                      ncclProfilerEventState_t eState,
                                                      ncclProfilerEventStateArgs_t* eStateArgs) {
  if (eHandle == nullptr) {
    return ncclSuccess;
  }

  inspectorAsyncEvent event;
  memset(&event, 0, sizeof(event));
  event.type = inspectorAsyncEventRecord;
  event.eHandle = eHandle;
  event.eState = eState;
  event.hasArgs = eStateArgs != nullptr;
  if (eStateArgs != nullptr) {
    event.eStateArgs = *eStateArgs;
  }
  event.tsProducerUsec = inspectorGetTime();
  inspectorPluginAsyncEnqueue(&event);
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
