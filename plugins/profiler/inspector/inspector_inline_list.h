/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef INSPECTOR_INLINE_LIST_H_
#define INSPECTOR_INLINE_LIST_H_

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Inline-first append-only list.
 *
 * The first InlineCapacity entries live directly inside the owner object.
 * Entries beyond that are allocated in fixed-size malloc chunks chained
 * together. This keeps the common case allocation-free while still preserving
 * every appended entry for large proxy operations.
 */
template<typename T, uint32_t InlineCapacity, uint32_t BlockCapacity>
struct inspectorInlineListBlock {
  inspectorInlineListBlock* next;
  uint32_t count;
  T entries[BlockCapacity];
};

template<typename T, uint32_t InlineCapacity, uint32_t BlockCapacity>
struct inspectorInlineList {
  uint32_t count;
  inspectorInlineListBlock<T, InlineCapacity, BlockCapacity>* head;
  inspectorInlineListBlock<T, InlineCapacity, BlockCapacity>* tail;
  T inlineEntries[InlineCapacity];
};

template<typename T, uint32_t InlineCapacity, uint32_t BlockCapacity>
static inline void inspectorInlineListInit(
    inspectorInlineList<T, InlineCapacity, BlockCapacity>* list) {
  if (list) memset(list, 0, sizeof(*list));
}

template<typename T, uint32_t InlineCapacity, uint32_t BlockCapacity>
static inline T* inspectorInlineListAppend(
    inspectorInlineList<T, InlineCapacity, BlockCapacity>* list) {
  if (list == nullptr) return nullptr;

  if (list->count < InlineCapacity) {
    T* entry = &list->inlineEntries[list->count++];
    memset(entry, 0, sizeof(*entry));
    return entry;
  }

  if (list->tail == nullptr || list->tail->count == BlockCapacity) {
    typedef inspectorInlineListBlock<T, InlineCapacity, BlockCapacity> Block;
    Block* block = (Block*)calloc(1, sizeof(Block));
    if (block == nullptr) return nullptr;
    if (list->tail != nullptr) {
      list->tail->next = block;
    } else {
      list->head = block;
    }
    list->tail = block;
  }

  T* entry = &list->tail->entries[list->tail->count++];
  list->count++;
  memset(entry, 0, sizeof(*entry));
  return entry;
}

template<typename T, uint32_t InlineCapacity, uint32_t BlockCapacity>
static inline const T* inspectorInlineListGet(
    const inspectorInlineList<T, InlineCapacity, BlockCapacity>* list,
    uint32_t index) {
  if (list == nullptr || index >= list->count) return nullptr;
  if (index < InlineCapacity) return &list->inlineEntries[index];

  uint32_t extraIndex = index - InlineCapacity;
  const inspectorInlineListBlock<T, InlineCapacity, BlockCapacity>* block = list->head;
  while (block != nullptr) {
    if (extraIndex < block->count) return &block->entries[extraIndex];
    extraIndex -= block->count;
    block = block->next;
  }
  return nullptr;
}

template<typename T, uint32_t InlineCapacity, uint32_t BlockCapacity>
static inline T* inspectorInlineListGetMutable(
    inspectorInlineList<T, InlineCapacity, BlockCapacity>* list,
    uint32_t index) {
  return const_cast<T*>(inspectorInlineListGet(list, index));
}

template<typename T, uint32_t InlineCapacity, uint32_t BlockCapacity>
static inline void inspectorInlineListFree(
    inspectorInlineList<T, InlineCapacity, BlockCapacity>* list) {
  if (list == nullptr) return;

  inspectorInlineListBlock<T, InlineCapacity, BlockCapacity>* block = list->head;
  while (block != nullptr) {
    inspectorInlineListBlock<T, InlineCapacity, BlockCapacity>* next = block->next;
    free(block);
    block = next;
  }
  memset(list, 0, sizeof(*list));
}

/*
 * Move ownership of list contents from src to dst.
 * After the move, dst owns all inline entries and overflow blocks;
 * src is left in a valid empty state (no dangling pointers).
 * This is O(1) for overflow blocks (pointer transfer) plus a fixed-size
 * memcpy of the inline array.
 */
template<typename T, uint32_t InlineCapacity, uint32_t BlockCapacity>
static inline void inspectorInlineListMove(
    inspectorInlineList<T, InlineCapacity, BlockCapacity>* dst,
    inspectorInlineList<T, InlineCapacity, BlockCapacity>* src) {
  if (dst == nullptr || src == nullptr) return;
  *dst = *src;
  memset(src, 0, sizeof(*src));
}

template<typename T, uint32_t InlineCapacity, uint32_t BlockCapacity>
static inline bool inspectorInlineListCopy(
    inspectorInlineList<T, InlineCapacity, BlockCapacity>* dst,
    const inspectorInlineList<T, InlineCapacity, BlockCapacity>* src) {
  if (dst == nullptr || src == nullptr) return false;

  inspectorInlineListInit(dst);
  for (uint32_t i = 0; i < src->count; i++) {
    const T* srcEntry = inspectorInlineListGet(src, i);
    T* dstEntry = inspectorInlineListAppend(dst);
    if (srcEntry == nullptr || dstEntry == nullptr) {
      inspectorInlineListFree(dst);
      return false;
    }
    *dstEntry = *srcEntry;
  }
  return true;
}

#endif  // INSPECTOR_INLINE_LIST_H_
