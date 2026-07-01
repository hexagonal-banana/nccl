#ifndef NCCL_INSPECTOR_EVENT_POOL_H_
#define NCCL_INSPECTOR_EVENT_POOL_H_

#include <pthread.h>
#include <stdint.h>

#include "inspector.h"

// Memory pool entry structures
struct inspectorCollInfoPoolEntry {
  struct inspectorCollInfo obj;
  struct inspectorCollInfoPoolEntry* next;
  bool inUse;
};

struct inspectorP2pInfoPoolEntry {
  struct inspectorP2pInfo obj;
  struct inspectorP2pInfoPoolEntry* next;
  bool inUse;
};

struct inspectorCommInfoPoolEntry {
  struct inspectorCommInfo obj;
  struct inspectorCommInfoPoolEntry* next;
  bool inUse;
};

struct inspectorProxyOpInfoPoolEntry {
  struct inspectorProxyOpInfo obj;
  struct inspectorProxyOpInfoPoolEntry* next;
  bool inUse;
};

struct inspectorProxyStepInfoPoolEntry {
  struct inspectorProxyStepInfo obj;
  struct inspectorProxyStepInfoPoolEntry* next;
  bool inUse;
};

struct inspectorProxyCtrlInfoPoolEntry {
  struct inspectorProxyCtrlInfo obj;
  struct inspectorProxyCtrlInfoPoolEntry* next;
  bool inUse;
};

// Chunk structure for stride-based pool growth
struct inspectorPoolChunk {
  void* entries;                    // Pointer to the array of entries in this chunk
  uint32_t chunkSize;               // Number of entries in this chunk
  struct inspectorPoolChunk* next;  // Next chunk in the list
};

struct inspectorEventPool {
  // Collective info pool
  struct inspectorPoolChunk* collChunkList;
  struct inspectorCollInfoPoolEntry* collFreeList;
  uint32_t collStrideSize;
  uint32_t collTotalSize;
  uint32_t collAllocCount;
  uint32_t collChunkCount;
  pthread_mutex_t collPoolLock;

  // P2P info pool
  struct inspectorPoolChunk* p2pChunkList;
  struct inspectorP2pInfoPoolEntry* p2pFreeList;
  uint32_t p2pStrideSize;
  uint32_t p2pTotalSize;
  uint32_t p2pAllocCount;
  uint32_t p2pChunkCount;
  pthread_mutex_t p2pPoolLock;

  // Comm info pool (keeping for future extensibility)
  struct inspectorPoolChunk* commChunkList;
  struct inspectorCommInfoPoolEntry* commFreeList;
  uint32_t commStrideSize;
  uint32_t commTotalSize;
  uint32_t commAllocCount;
  uint32_t commChunkCount;
  pthread_mutex_t commPoolLock;

  // Proxy event pools
  struct inspectorPoolChunk* proxyOpChunkList;
  struct inspectorProxyOpInfoPoolEntry* proxyOpFreeList;
  uint32_t proxyOpStrideSize;
  uint32_t proxyOpTotalSize;
  uint32_t proxyOpAllocCount;
  uint32_t proxyOpChunkCount;
  pthread_mutex_t proxyOpPoolLock;

  struct inspectorPoolChunk* proxyStepChunkList;
  struct inspectorProxyStepInfoPoolEntry* proxyStepFreeList;
  uint32_t proxyStepStrideSize;
  uint32_t proxyStepTotalSize;
  uint32_t proxyStepAllocCount;
  uint32_t proxyStepChunkCount;
  pthread_mutex_t proxyStepPoolLock;

  struct inspectorPoolChunk* proxyCtrlChunkList;
  struct inspectorProxyCtrlInfoPoolEntry* proxyCtrlFreeList;
  uint32_t proxyCtrlStrideSize;
  uint32_t proxyCtrlTotalSize;
  uint32_t proxyCtrlAllocCount;
  uint32_t proxyCtrlChunkCount;
  pthread_mutex_t proxyCtrlPoolLock;

  // Controls whether pools are allowed to grow beyond their initial size.
  // Disabled via NCCL_INSPECTOR_POOL_GROW=0.
  bool growEnabled;
};

extern struct inspectorEventPool g_eventPool;

// Memory pool functions
inspectorResult_t inspectorEventPoolInit(uint32_t collPoolSize,
                                         uint32_t p2pPoolSize,
                                         uint32_t commPoolSize,
                                         uint32_t proxyOpPoolSize,
                                         uint32_t proxyStepPoolSize,
                                         uint32_t proxyCtrlPoolSize);
inspectorResult_t inspectorEventPoolFinalize();

struct inspectorCollInfo* inspectorEventPoolAllocColl();
struct inspectorP2pInfo* inspectorEventPoolAllocP2p();
struct inspectorCommInfo* inspectorEventPoolAllocComm();
struct inspectorProxyOpInfo* inspectorEventPoolAllocProxyOp();
struct inspectorProxyStepInfo* inspectorEventPoolAllocProxyStep();
struct inspectorProxyCtrlInfo* inspectorEventPoolAllocProxyCtrl();
void inspectorEventPoolReleaseColl(struct inspectorCollInfo* collInfo);
void inspectorEventPoolReleaseP2p(struct inspectorP2pInfo* p2pInfo);
void inspectorEventPoolReleaseComm(struct inspectorCommInfo* commInfo);
void inspectorEventPoolReleaseProxyOp(struct inspectorProxyOpInfo* opInfo);
void inspectorEventPoolReleaseProxyStep(struct inspectorProxyStepInfo* stepInfo);
void inspectorEventPoolReleaseProxyCtrl(struct inspectorProxyCtrlInfo* ctrlInfo);

#endif // NCCL_INSPECTOR_EVENT_POOL_H_
