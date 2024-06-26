#pragma once

#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void yeb_bootstrap();

#define DA_INIT_CAP 256

#define da_push(DA, ITEM)                                                      \
  do {                                                                         \
    __auto_type DA_ = (DA);                                                    \
    ++DA_->da_len;                                                             \
    if (DA_->da_items == NULL) {                                               \
      DA_->da_cap = DA_INIT_CAP;                                               \
      DA_->da_items = malloc(sizeof(DA_->da_items[0]) * DA_INIT_CAP);          \
      assert(DA_->da_items != NULL);                                           \
    } else if (DA_->da_len > DA_->da_cap) {                                    \
      DA_->da_cap *= 2;                                                        \
      DA_->da_items =                                                          \
          realloc(DA_->da_items, sizeof(DA_->da_items[0]) * DA_->da_cap);      \
      assert(DA_->da_items != NULL);                                           \
    }                                                                          \
    DA_->da_items[DA_->da_len - 1] = (ITEM);                                   \
  } while (0)

#define da_append(DA, ITEMS, N)                                                \
  do {                                                                         \
    __auto_type DA_ = (DA);                                                    \
    size_t N_ = (N);                                                           \
    size_t I_ = DA_->da_len;                                                   \
    DA_->da_len += N_;                                                         \
    if (DA_->da_items == NULL) {                                               \
      DA_->da_cap = DA_INIT_CAP > N_ ? DA_INIT_CAP : N_;                       \
      DA_->da_items = malloc(sizeof(DA_->da_items[0]) * DA_->da_cap);          \
      assert(DA_->da_items != NULL);                                           \
    } else if (DA_->da_len > DA_->da_cap) {                                    \
      DA_->da_cap *= 2;                                                        \
      DA_->da_cap = DA_->da_cap > DA_->da_len ? DA_->da_cap : DA_->da_len;     \
      DA_->da_items =                                                          \
          realloc(DA_->da_items, sizeof(DA_->da_items[0]) * DA_->da_cap);      \
      assert(DA_->da_items != NULL);                                           \
    }                                                                          \
    memcpy(&DA_->da_items[I_], (ITEMS), sizeof(DA_->da_items[0]) * N_);        \
  } while (0)

#define da_append_multiple(DA, ...)                                            \
  do {                                                                         \
    __auto_type DA__ = (DA);                                                   \
    typeof((DA__)->da_items[0]) items[] = {__VA_ARGS__};                       \
    da_append(DA__, items, sizeof(items) / sizeof(items[0]));                  \
  } while (0)

#define da_reserve(DA, N)                                                      \
  do {                                                                         \
    __auto_type DA_ = (DA);                                                    \
    size_t n = (N);                                                            \
    if (DA_->da_items == NULL) {                                               \
      DA_->da_cap = n;                                                         \
      DA_->da_items = malloc(sizeof(DA_->da_items[0]) * DA_->da_cap);          \
      assert(DA_->da_items != NULL);                                           \
    } else if (DA_->da_len + n > DA_->da_cap) {                                \
      DA_->da_cap = DA_->da_len + n;                                           \
      DA_->da_items =                                                          \
          realloc(DA_->da_items, sizeof(DA_->da_items[0]) * DA_->da_cap);      \
      assert(DA_->da_items != NULL);                                           \
    }                                                                          \
  } while (0)

#define da_remove_last(DA) (--(DA)->da_len)

#define da_insert(DA, I, ITEM)                                                 \
  do {                                                                         \
    __auto_type DA_ = (DA);                                                    \
    size_t I_ = (I);                                                           \
    ++DA_->da_len;                                                             \
    if (DA_->da_items == NULL) {                                               \
      DA_->da_cap = DA_INIT_CAP;                                               \
      DA_->da_items = malloc(sizeof(DA_->da_items[0]) * DA_INIT_CAP);          \
      assert(DA_->da_items != NULL);                                           \
    } else if (DA_->da_len > DA_->da_cap) {                                    \
      DA_->da_cap *= 2;                                                        \
      DA_->da_items =                                                          \
          realloc(DA_->da_items, sizeof(DA_->da_items[0]) * DA_->da_cap);      \
      assert(DA_->da_items != NULL);                                           \
    }                                                                          \
    memmove(&DA_->da_items[I_ + 1], &DA_->da_items[I_],                        \
            (DA_->da_len - I_ - 1) * sizeof(DA_->da_items[0]));                \
    DA_->da_items[I_] = (ITEM);                                                \
  } while (0)

#define da_remove(DA, I)                                                       \
  do {                                                                         \
    __auto_type DA_ = (DA);                                                    \
    size_t I_ = (I);                                                           \
    --DA_->da_len;                                                             \
    memmove(&DA_->da_items[I_], &DA_->da_items[I_ + 1],                        \
            (DA_->da_len - I_) * sizeof(DA_->da_items[0]));                    \
  } while (0)

#define DECL_DA_STRUCT(T, NAME)                                                \
  typedef struct NAME {                                                        \
    T *da_items;                                                               \
    size_t da_len;                                                             \
    size_t da_cap;                                                             \
  } NAME;

#define DA_FOR(ARR, IDX, ITEM, BLOCK)                                          \
  do {                                                                         \
    __auto_type ARR_ = (ARR);                                                  \
    for (size_t IDX = 0; IDX < ARR_->da_len; ++IDX) {                          \
      __auto_type ITEM = da_get(ARR_, IDX);                                    \
      if (1)                                                                   \
        BLOCK;                                                                 \
    }                                                                          \
  } while (0);

#define da_get(ARR, IDX) (&(ARR)->da_items[(IDX)])

#define da_get_checked(ARR, IDX)                                               \
  (assert((IDX) < (ARR)->da_len), &(ARR)->da_items[(IDX)])

DECL_DA_STRUCT(const char *, ConstStrings);

typedef struct DynString {
  /// Non nullable.
  char *cstr;
  /// `len` does not count the null character in the end.
  size_t len;
  size_t cap;
} DynString;

static inline DynString dynstring_new() {
  char *chars = malloc(DA_INIT_CAP);
  assert(chars != NULL);
  chars[0] = '\0';
  return (DynString){
      .cstr = chars,
      .len = 0,
      .cap = DA_INIT_CAP,
  };
}

static inline char *dynstring_get(DynString *s, size_t i) {
  return &s->cstr[i];
}

static inline char *dynstring_get_checked(DynString *s, size_t i) {
  assert(i < s->len);
  return dynstring_get(s, i);
}

static inline void dynstring_push(DynString *s, char c) {
  ++s->len;
  if (s->len + 1 > s->cap) {
    s->cap *= 2;
    s->cstr = realloc(s->cstr, s->cap);
    assert(s->cstr != NULL);
  }
  *dynstring_get(s, s->len - 1) = c;
  *dynstring_get(s, s->len) = '\0';
}

static inline void dynstring_append(DynString *s, const char *src, size_t len) {
  size_t i = s->len;
  s->len += len;
  if (s->len + 1 > s->cap) {
    s->cap = (s->len + 1 > s->cap) ? s->len + 1 : s->cap * 2;
    s->cstr = realloc(s->cstr, s->cap);
    assert(s->cstr != NULL);
  }
  memcpy(dynstring_get(s, i), src, len);
  *dynstring_get(s, s->len) = '\0';
}

static inline void dynstring_append_cstr(DynString *s, const char *cstr) {
  size_t len = strlen(cstr);
  size_t i = s->len;
  s->len += len;
  if (s->len + 1 > s->cap) {
    s->cap = (s->len + 1 > s->cap) ? s->len + 1 : s->cap * 2;
    s->cstr = realloc(s->cstr, s->cap);
    assert(s->cstr != NULL);
  }
  memcpy(dynstring_get(s, i), cstr, len + 1);
}

#define dynstring_append_literal(S, LITERAL)                                   \
  dynstring_append((S), LITERAL, sizeof(LITERAL) - 1)

typedef struct cmd {
  ConstStrings args;
} Cmd;

#define CMD_APPEND(CMD, ...) da_append_multiple(&(CMD)->args, __VA_ARGS__)

void execute(Cmd cmd);

/// Null means absent of value.
typedef union MaybeDynString {
  DynString ds;
} MaybeDynString;

/// djb2 hash.
static inline uint64_t hash_cstr(const uint8_t *bytes) {
  uint64_t hash = 5381;
  uint8_t c;

  while ((c = *bytes++))
    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

  return hash;
}

/// NOT SECURE hash map from string to maybe string.
/// Owns the strings.
/// The map uses a very simple hashing algorithm which is very easily reversable.
typedef struct Options {
  struct OptionsKV {
    DynString k;
    MaybeDynString v;
  } **map;
} Options;

static inline struct OptionsKV parse_arg(const char *arg) {
  for (size_t i = 0;; ++i) {
    char c = arg[i];
    if (c == '\0') {
      DynString k = dynstring_new();
      dynstring_append(&k, arg, i);
      return (struct OptionsKV){k, {0}};
    } else if (c == '=') {
      DynString k = dynstring_new();
      dynstring_append(&k, arg, i);
      size_t j = i + 1;
      while (arg[j] != '\0')
        ++j;
      DynString v = dynstring_new();
      if (j != i)
        dynstring_append(&v, &arg[i + 1], j - 1 - i);
      return (struct OptionsKV){k, {v}};
    }
  }
  __builtin_unreachable();
}

static inline Options parse_argv(int argc, char **argv) {
  const size_t map_size = 65536;
  struct OptionsKV **map = malloc(map_size * sizeof(struct OptionsKV *));
  assert(map != NULL);
  memset(map, 0, map_size);
  for (int i = 0; i < argc; ++i) {
    const char *arg = argv[i];
    struct OptionsKV kv = parse_arg(arg);
    size_t idx = hash_cstr((uint8_t *)kv.k.cstr) % 65536;
    struct OptionsKV *kv_ = malloc(sizeof(kv));
    assert(kv_ != NULL);
    memcpy(kv_, &kv, sizeof(kv));
    map[idx] = kv_;
  }
  return (Options){map};
}

typedef struct OptionFlag {
  bool exists;
  const MaybeDynString *value;
} OptionFlag;

static inline OptionFlag opts_get(Options o, const char *key) {
  size_t idx = hash_cstr((uint8_t *)key) % 65536;
  struct OptionsKV *kv = o.map[idx];
  if (kv == NULL)
    return (OptionFlag){.exists = false, 0};
  else
    return (OptionFlag){.exists = true, &kv->v};
}

#ifndef YEB_NO_IMPL

#ifdef YEB_INTERNAL

void yeb_bootstrap(){};

static inline DynString concat_strings_with_space(ConstStrings strings) {
  if (strings.da_len == 0) {
    return dynstring_new();
  }
  DynString s = {0};
  DA_FOR(&strings, i, s_, {
    dynstring_append_cstr(&s, *s_);
    dynstring_push(&s, ' ');
  });
  *dynstring_get_checked(&s, s.len - 1) = '\0';
  return s;
}

void execute(Cmd cmd) {
  const char *s = concat_strings_with_space(cmd.args).cstr;
  printf("$ %s\n", s);
  int exit_code = system(s);
  if (exit_code != 0) {
    printf("YEB: command finished with non-zero exit code\n");
    exit(exit_code);
  }
}

#else
#ifdef YEB_MAIN

int main(int argc, char **argv) {
  int exit_code;
  exit_code = system("clang -c -DYEB_NO_IMPL build.c -o yeb/build.o");
  if (exit_code != 0)
    return exit_code;
  exit_code = system("clang yeb/yeb.o yeb/build.o -o yeb/a.out");
  if (exit_code != 0)
    return exit_code;
  DynString cmd = {0};
  dynstring_append_cstr(&cmd, "./yeb/a.out");
  for (int i = 1; i < argc; ++i) {
    dynstring_push(&cmd, ' ');
    dynstring_append_cstr(&cmd, argv[i]);
  }
  exit_code = system(cmd.cstr);
  return exit_code;
}

#else

#define YEB_ERROR_NOT_BOOTSTRAPPED()                                           \
  (printf("You need to call `yeb_bootstrap()` before doing anything in "       \
          "`build.c`\n"),                                                      \
   exit(1))

void execute(Cmd cmd) { YEB_ERROR_NOT_BOOTSTRAPPED(); }

void yeb_bootstrap() {
  FILE *yeb_is_bootstrapped_txt = fopen("yeb/yeb_is_bootstrapped.txt", "r");
  if (yeb_is_bootstrapped_txt != NULL) {
    fclose(yeb_is_bootstrapped_txt);
    printf(
        "Yeb was already bootstrapped, use `./yeb/yeb` to build the project.\n"
        "If you want to boostrap again because it's not working correctly, "
        "delete `yeb/` and boostrap again.\n");
    exit(0);
  }

  int exit_code;

#define yeb_bootstrap_execute(S)                                               \
  printf("$ %s\n", S);                                                         \
  exit_code = system(S);                                                       \
  if (exit_code != 0)                                                          \
    if (exit_code != 0) {                                                      \
      printf("command `%s` exited with non-zero exit code\n", S);              \
      exit(1);                                                                 \
    }

  yeb_bootstrap_execute("mkdir -p yeb/");
  yeb_bootstrap_execute("cp yeb.h yeb/yeb.h");
  FILE *yeb_c = fopen("yeb/yeb.c", "w");
  assert(yeb_c != NULL);
  fprintf(yeb_c, //
          "#define YEB_INTERNAL\n"
          "#include \"yeb.h\"\n");
  fclose(yeb_c);
  yeb_bootstrap_execute("clang -c yeb/yeb.c -o yeb/yeb.o");
  FILE *run_c = fopen("yeb/run.c", "w");
  assert(run_c != NULL);
  fprintf(run_c, //
          "#define YEB_MAIN\n"
          "#include \"yeb.h\"\n");
  fclose(run_c);
  yeb_bootstrap_execute("clang yeb/run.c -o yeb/yeb");
  yeb_bootstrap_execute("touch yeb/yeb_is_bootstrapped.txt");
  yeb_is_bootstrapped_txt = fopen("yeb/yeb_is_bootstrapped.txt", "w");
  fprintf(yeb_is_bootstrapped_txt,
          "This file is here so yeb knows it's already been bootstrapped.\n"
          "If you're experiencing problems with yeb, try removing this file "
          "(or the yeb/ directory) and bootstrap again.\n");
  fclose(yeb_is_bootstrapped_txt);
  printf("Yeb is bootstrapped, now do `./yeb/yeb` to build the project.\n");
  exit(0);
}

#endif
#endif

#endif // YEB_NO_IMPL
