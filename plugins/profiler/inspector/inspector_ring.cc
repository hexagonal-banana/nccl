#include "inspector.h"
#include "inspector_ring.h"
#include <stdlib.h>
#include <string.h>

/*
 * Description:
 *   Initializes a fixed-size ring buffer.
 *
 * Thread Safety:
 *   Not thread-safe (should be called during communicator initialization).
 *
 * Input:
 *   struct inspectorCompletedRing* ring - ring buffer to initialize.
 *   uint32_t size      - ring capacity (# of entries).
 *   size_t   entrySize - size of each entry in bytes.
 *
 * Output:
 *   Ring buffer memory is allocated and counters are reset.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
inspectorResult_t inspectorRingInit(struct inspectorCompletedRing* ring,
                                    uint32_t size,
                                    size_t entrySize) {
  return inspectorRingInitWithCallbacks(ring, size, entrySize, nullptr, nullptr);
}

inspectorResult_t inspectorRingInitWithCallbacks(struct inspectorCompletedRing* ring,
                                                 uint32_t size,
                                                 size_t entrySize,
                                                 inspectorRingEntryCopyFn copyEntry,
                                                 inspectorRingEntryDestroyFn destroyEntry) {
  if (!ring) return inspectorMemoryError;
  ring->entrySize = entrySize;
  ring->copyEntry = copyEntry;
  ring->destroyEntry = destroyEntry;
  if (size == 0) {
    ring->entries = nullptr;
    ring->size = ring->head = ring->tail = 0;
    return inspectorSuccess;
  }

  // Allocate one extra slot so empty (head==tail) and full
  // ((tail+1)%(size+1)==head) are always distinguishable without a count field.
  ring->entries = calloc(size + 1, entrySize);
  if (!ring->entries) return inspectorMemoryError;

  ring->size = size;
  ring->head = 0;
  ring->tail = 0;
  return inspectorSuccess;
}

/*
 * Description:
 *   Finalizes a ring buffer and frees its storage.
 *
 * Thread Safety:
 *   Not thread-safe (should be called during communicator teardown).
 *
 * Input:
 *   struct inspectorCompletedRing* ring - ring buffer to finalize.
 *
 * Output:
 *   Ring buffer memory is freed and counters are reset.
 *
 * Return:
 *   None.
 */
void inspectorRingFinalize(struct inspectorCompletedRing* ring) {
  if (!ring) return;
  if (ring->destroyEntry && ring->entries && ring->size > 0) {
    uint32_t bufSize = ring->size + 1;
    uint32_t count = (ring->tail + bufSize - ring->head) % bufSize;
    for (uint32_t i = 0; i < count; i++) {
      uint32_t idx = (ring->head + i) % bufSize;
      ring->destroyEntry(ringSlot(ring, idx));
    }
  }
  free(ring->entries);
  ring->entries = nullptr;
  ring->size = ring->head = ring->tail = 0;
  ring->copyEntry = nullptr;
  ring->destroyEntry = nullptr;
}

/*
 * Description:
 *   Enqueues an entry into the ring buffer.
 *   Overwrites the oldest entry if the ring is full.
 *
 * Thread Safety:
 *   Not thread-safe (caller must provide synchronization).
 *
 * Input:
 *   struct inspectorCompletedRing* ring - target ring buffer.
 *   const void* entry - entry to copy in (must be ring->entrySize bytes).
 *
 * Output:
 *   Ring buffer state is updated to include the new entry.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
inspectorResult_t inspectorRingEnqueue(struct inspectorCompletedRing* ring,
                                       const void* entry) {
  if (!ring || !entry) return inspectorMemoryError;
  if (ring->size == 0 || !ring->entries) return inspectorSuccess;

  uint32_t bufSize = ring->size + 1;
  if ((ring->tail + 1) % bufSize == ring->head) {
    // Ring is full: advance head to overwrite the oldest entry
    if (ring->destroyEntry) ring->destroyEntry(ringSlot(ring, ring->head));
    ring->head = (ring->head + 1) % bufSize;
  }

  void* dst = ringSlot(ring, ring->tail);
  if (ring->copyEntry) {
    inspectorResult_t res = ring->copyEntry(dst, entry);
    if (res != inspectorSuccess) return res;
  } else {
    memcpy(dst, entry, ring->entrySize);
  }
  ring->tail = (ring->tail + 1) % bufSize;
  return inspectorSuccess;
}
