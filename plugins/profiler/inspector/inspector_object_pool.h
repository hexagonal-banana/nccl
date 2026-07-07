#ifndef INSPECTOR_OBJECT_POOL_H_
#define INSPECTOR_OBJECT_POOL_H_

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <new>

// Forward declaration for logging macros (defined in inspector.h)
// Pool implementation uses INFO_INSPECTOR/WARN_INSPECTOR when available.

/*
 * Generic thread-safe object pool with stride-based chunk growth.
 *
 * Objects are stored in pre-allocated chunks. alloc() uses placement-new to
 * construct a T in-place; release() calls the destructor and returns the slot
 * to the free list. This is safe for types containing std::vector or other
 * non-trivial members.
 *
 * Template parameter T must be default-constructible.
 */
template<typename T>
class ObjectPool {
 public:
  ObjectPool() = default;

  bool init(uint32_t stride, bool grow = true) {
    stride_ = stride;
    growEnabled_ = grow;
    totalSize_ = 0;
    allocCount_ = 0;
    chunkCount_ = 0;
    chunks_ = nullptr;
    freeList_ = nullptr;

    if (pthread_mutex_init(&mu_, nullptr) != 0) {
      return false;
    }
    muInitialized_ = true;

    if (!growPool()) {
      pthread_mutex_destroy(&mu_);
      muInitialized_ = false;
      return false;
    }
    return true;
  }

  ~ObjectPool() {
    // Free all chunks (objects should already be released)
    Chunk* chunk = chunks_;
    while (chunk != nullptr) {
      Chunk* next = chunk->next;
      free(chunk->slots);
      free(chunk);
      chunk = next;
    }
    chunks_ = nullptr;
    freeList_ = nullptr;
    if (muInitialized_) {
      pthread_mutex_destroy(&mu_);
    }
  }

  // Non-copyable, non-movable
  ObjectPool(const ObjectPool&) = delete;
  ObjectPool& operator=(const ObjectPool&) = delete;

  T* alloc() {
    pthread_mutex_lock(&mu_);

    if (freeList_ == nullptr) {
      if (!growEnabled_) {
        pthread_mutex_unlock(&mu_);
        return nullptr;
      }
      if (!growPool()) {
        pthread_mutex_unlock(&mu_);
        return nullptr;
      }
    }

    Slot* slot = freeList_;
    freeList_ = slot->nextFree;
    slot->inUse = true;
    allocCount_++;

    pthread_mutex_unlock(&mu_);

    // Placement-new: properly constructs T (including std::vector members)
    return new (slot->storage) T();
  }

  void release(T* ptr) {
    if (ptr == nullptr) return;

    // storage is the first member of Slot, so ptr == &slot->storage[0]
    Slot* slot = reinterpret_cast<Slot*>(ptr);

    pthread_mutex_lock(&mu_);

    if (!slot->inUse) {
      pthread_mutex_unlock(&mu_);
      // Double release detected
      return;
    }

    // Call destructor (properly destroys std::vector members, etc.)
    ptr->~T();

    slot->inUse = false;
    slot->nextFree = freeList_;
    freeList_ = slot;
    allocCount_--;

    pthread_mutex_unlock(&mu_);
  }

  uint32_t allocCount() const { return allocCount_; }
  uint32_t totalSize() const { return totalSize_; }
  uint32_t chunkCount() const { return chunkCount_; }

 private:
  struct Slot {
    // storage MUST be the first member so that reinterpret_cast<Slot*>(T*) works
    alignas(T) unsigned char storage[sizeof(T)];
    Slot* nextFree;
    bool inUse;
  };

  struct Chunk {
    Slot* slots;
    uint32_t size;
    Chunk* next;
  };

  bool growPool() {
    Chunk* chunk = static_cast<Chunk*>(calloc(1, sizeof(Chunk)));
    if (chunk == nullptr) return false;

    chunk->slots = static_cast<Slot*>(calloc(stride_, sizeof(Slot)));
    if (chunk->slots == nullptr) {
      free(chunk);
      return false;
    }
    chunk->size = stride_;

    // Build free list for new chunk
    for (uint32_t i = 0; i < stride_ - 1; i++) {
      chunk->slots[i].nextFree = &chunk->slots[i + 1];
      chunk->slots[i].inUse = false;
    }
    chunk->slots[stride_ - 1].nextFree = freeList_;
    chunk->slots[stride_ - 1].inUse = false;
    freeList_ = &chunk->slots[0];

    // Prepend chunk to list
    chunk->next = chunks_;
    chunks_ = chunk;
    chunkCount_++;
    totalSize_ += stride_;

    return true;
  }

  Chunk* chunks_{nullptr};
  Slot* freeList_{nullptr};
  pthread_mutex_t mu_{};
  bool muInitialized_{false};
  uint32_t stride_{256};
  uint32_t totalSize_{0};
  uint32_t allocCount_{0};
  uint32_t chunkCount_{0};
  bool growEnabled_{true};
};

#endif  // INSPECTOR_OBJECT_POOL_H_
