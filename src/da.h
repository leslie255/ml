#pragma once

#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DA_INIT_CAP 256

#define da_push(DA, ITEM)                                                                                              \
  do {                                                                                                                 \
    __auto_type DA_ = (DA);                                                                                            \
    ++DA_->da_len;                                                                                                     \
    if (DA_->da_items == NULL) {                                                                                       \
      DA_->da_cap = DA_INIT_CAP;                                                                                       \
      DA_->da_items = malloc(sizeof(DA_->da_items[0]) * DA_INIT_CAP);                                                  \
      assert(DA_->da_items != NULL);                                                                                   \
    } else if (DA_->da_len > DA_->da_cap) {                                                                            \
      DA_->da_cap *= 2;                                                                                                \
      DA_->da_items = realloc(DA_->da_items, sizeof(DA_->da_items[0]) * DA_->da_cap);                                  \
      assert(DA_->da_items != NULL);                                                                                   \
    }                                                                                                                  \
    DA_->da_items[DA_->da_len - 1] = (ITEM);                                                                           \
  } while (0)

#define da_append(DA, ITEMS, N)                                                                                        \
  do {                                                                                                                 \
    __auto_type DA_ = (DA);                                                                                            \
    size_t N_ = (N);                                                                                                   \
    size_t I_ = DA_->da_len;                                                                                           \
    DA_->da_len += N_;                                                                                                 \
    if (DA_->da_items == NULL) {                                                                                       \
      DA_->da_cap = DA_INIT_CAP > N_ ? DA_INIT_CAP : N_;                                                               \
      DA_->da_items = malloc(sizeof(DA_->da_items[0]) * DA_->da_cap);                                                  \
      assert(DA_->da_items != NULL);                                                                                   \
    } else if (DA_->da_len > DA_->da_cap) {                                                                            \
      DA_->da_cap *= 2;                                                                                                \
      DA_->da_cap = DA_->da_cap > DA_->da_len ? DA_->da_cap : DA_->da_len;                                             \
      DA_->da_items = realloc(DA_->da_items, sizeof(DA_->da_items[0]) * DA_->da_cap);                                  \
      assert(DA_->da_items != NULL);                                                                                   \
    }                                                                                                                  \
    memcpy(&DA_->da_items[I_], (ITEMS), sizeof(DA_->da_items[0]) * N_);                                                \
  } while (0)

#define da_append_zeros(DA, N)                                                                                         \
  do {                                                                                                                 \
    __auto_type DA_ = (DA);                                                                                            \
    size_t N_ = (N);                                                                                                   \
    size_t I_ = DA_->da_len;                                                                                           \
    DA_->da_len += N_;                                                                                                 \
    if (DA_->da_items == NULL) {                                                                                       \
      DA_->da_cap = DA_INIT_CAP > N_ ? DA_INIT_CAP : N_;                                                               \
      DA_->da_items = malloc(sizeof(DA_->da_items[0]) * DA_->da_cap);                                                  \
      assert(DA_->da_items != NULL);                                                                                   \
    } else if (DA_->da_len > DA_->da_cap) {                                                                            \
      DA_->da_cap *= 2;                                                                                                \
      DA_->da_cap = DA_->da_cap > DA_->da_len ? DA_->da_cap : DA_->da_len;                                             \
      DA_->da_items = realloc(DA_->da_items, sizeof(DA_->da_items[0]) * DA_->da_cap);                                  \
      assert(DA_->da_items != NULL);                                                                                   \
    }                                                                                                                  \
    memset(&DA_->da_items[I_], 0, sizeof(DA_->da_items[0]) * N_);                                                      \
  } while (0)

#define da_append_multiple(DA, ...)                                                                                    \
  do {                                                                                                                 \
    __auto_type DA__ = (DA);                                                                                           \
    typeof((DA__)->da_items[0]) items[] = {__VA_ARGS__};                                                               \
    da_append(DA__, items, sizeof(items) / sizeof(items[0]));                                                          \
  } while (0)

#define da_reserve_exact(DA, N)                                                                                        \
  do {                                                                                                                 \
    __auto_type DA_ = (DA);                                                                                            \
    size_t N_ = (N);                                                                                                   \
    if (DA_->da_items == NULL) {                                                                                       \
      DA_->da_cap = N_;                                                                                                \
      DA_->da_items = malloc(sizeof(DA_->da_items[0]) * DA_->da_cap);                                                  \
      assert(DA_->da_items != NULL);                                                                                   \
    } else if (DA_->da_len + N_ > DA_->da_cap) {                                                                       \
      DA_->da_cap = DA_->da_len + N_;                                                                                  \
      DA_->da_items = realloc(DA_->da_items, sizeof(DA_->da_items[0]) * DA_->da_cap);                                  \
      assert(DA_->da_items != NULL);                                                                                   \
    }                                                                                                                  \
  } while (0)

#define da_reserve(DA, N)                                                                                              \
  do {                                                                                                                 \
    __auto_type DA_ = (DA);                                                                                            \
    size_t N_ = (N);                                                                                                   \
    size_t NEW_LEN = DA_->da_len + N;                                                                                  \
    if (DA_->da_items == NULL) {                                                                                       \
      DA_->da_cap = DA_INIT_CAP > N_ ? DA_INIT_CAP : N_;                                                               \
      DA_->da_items = malloc(sizeof(DA_->da_items[0]) * DA_->da_cap);                                                  \
      assert(DA_->da_items != NULL);                                                                                   \
    } else if (NEW_LEN > DA_->da_cap) {                                                                                \
      DA_->da_cap *= 2;                                                                                                \
      DA_->da_cap = DA_->da_cap > NEW_LEN ? DA_->da_cap : NEW_LEN;                                                     \
      DA_->da_items = realloc(DA_->da_items, sizeof(DA_->da_items[0]) * DA_->da_cap);                                  \
      assert(DA_->da_items != NULL);                                                                                   \
    }                                                                                                                  \
  } while (0)

#define da_remove_last(DA) (--(DA)->da_len)

#define da_insert(DA, I, ITEM)                                                                                         \
  do {                                                                                                                 \
    __auto_type DA_ = (DA);                                                                                            \
    size_t I_ = (I);                                                                                                   \
    ++DA_->da_len;                                                                                                     \
    if (DA_->da_items == NULL) {                                                                                       \
      DA_->da_cap = DA_INIT_CAP;                                                                                       \
      DA_->da_items = malloc(sizeof(DA_->da_items[0]) * DA_INIT_CAP);                                                  \
      assert(DA_->da_items != NULL);                                                                                   \
    } else if (DA_->da_len > DA_->da_cap) {                                                                            \
      DA_->da_cap *= 2;                                                                                                \
      DA_->da_items = realloc(DA_->da_items, sizeof(DA_->da_items[0]) * DA_->da_cap);                                  \
      assert(DA_->da_items != NULL);                                                                                   \
    }                                                                                                                  \
    memmove(&DA_->da_items[I_ + 1], &DA_->da_items[I_], (DA_->da_len - I_ - 1) * sizeof(DA_->da_items[0]));            \
    DA_->da_items[I_] = (ITEM);                                                                                        \
  } while (0)

#define da_remove(DA, I)                                                                                               \
  do {                                                                                                                 \
    __auto_type DA_ = (DA);                                                                                            \
    size_t I_ = (I);                                                                                                   \
    --DA_->da_len;                                                                                                     \
    memmove(&DA_->da_items[I_], &DA_->da_items[I_ + 1], (DA_->da_len - I_) * sizeof(DA_->da_items[0]));                \
  } while (0)

#define da_free(DA) free((DA).da_items)

#define DECL_DA_STRUCT(T, NAME)                                                                                        \
  typedef struct NAME {                                                                                                \
    T *da_items;                                                                                                       \
    size_t da_len;                                                                                                     \
    size_t da_cap;                                                                                                     \
  } NAME;

#define DA_FOR(ARR, IDX, ITEM, BLOCK)                                                                                  \
  do {                                                                                                                 \
    __auto_type ARR_ = (ARR);                                                                                          \
    for (size_t IDX = 0; IDX < ARR_->da_len; ++IDX) {                                                                  \
      __auto_type ITEM = da_get(ARR_, IDX);                                                                            \
      if (1)                                                                                                           \
        BLOCK;                                                                                                         \
    }                                                                                                                  \
  } while (0);

#define da_get(ARR, IDX) (&(ARR)->da_items[(IDX)])

#define da_get_checked(ARR, IDX) (assert((IDX) < (ARR)->da_len), &(ARR)->da_items[(IDX)])
