#ifndef INSPECTOR_INSPECTOR_RING_H_
#define INSPECTOR_INSPECTOR_RING_H_

#include <stdint.h>
#include <vector>
#include <utility>

/*
 * Typed fixed-size ring buffer with overwrite-on-full semantics.
 *
 * Uses the sentinel approach: allocates size+1 slots so that
 * empty (head==tail) and full ((tail+1)%(size+1)==head) are always
 * distinguishable without a separate count field.
 *
 * Elements are moved in/out via std::move. No function pointers for
 * copy/destroy — the element type's move constructor/destructor handles
 * resource management automatically (RAII).
 */
template<typename T>
class FixedRingBuffer {
 public:
  FixedRingBuffer() = default;

  explicit FixedRingBuffer(uint32_t capacity)
    : buf_(capacity + 1), size_(capacity) {}

  // Deferred initialization (for default-constructed instances)
  void init(uint32_t capacity) {
    buf_.resize(capacity + 1);
    size_ = capacity;
    head_ = 0;
    tail_ = 0;
  }

  // Move-enqueue an element. Overwrites the oldest entry if the ring is full.
  void enqueue(T&& item) {
    if (size_ == 0) return;
    uint32_t bufSize = size_ + 1;
    if ((tail_ + 1) % bufSize == head_) {
      // Ring full: discard oldest by advancing head
      buf_[head_] = T{};  // reset oldest slot
      head_ = (head_ + 1) % bufSize;
    }
    buf_[tail_] = std::move(item);
    tail_ = (tail_ + 1) % bufSize;
  }

  // Drain all entries from the ring, returning them as a vector via move.
  // The ring is left empty after this call.
  std::vector<T> drain() {
    std::vector<T> result;
    if (size_ == 0 || head_ == tail_) return result;

    uint32_t bufSize = size_ + 1;
    uint32_t count = (tail_ + bufSize - head_) % bufSize;
    result.reserve(count);

    for (uint32_t i = 0; i < count; i++) {
      uint32_t idx = (head_ + i) % bufSize;
      result.push_back(std::move(buf_[idx]));
    }
    head_ = 0;
    tail_ = 0;
    return result;
  }

  bool nonEmpty() const { return head_ != tail_; }

  uint32_t capacity() const { return size_; }

  // Clear and release backing storage
  void finalize() {
    buf_.clear();
    buf_.shrink_to_fit();
    head_ = tail_ = size_ = 0;
  }

 private:
  std::vector<T> buf_;
  uint32_t head_{0};
  uint32_t tail_{0};
  uint32_t size_{0};
};

#endif  // INSPECTOR_INSPECTOR_RING_H_
