#include "yeb.h"

bool is_release = false;

void cc(Cmd *cmd) {
  CMD_APPEND(cmd, "clang");
}

void cflags(Cmd *cmd) {
  CMD_APPEND(cmd, "-Wall -Wextra --std=gnu17");
  if (is_release)
    CMD_APPEND(cmd, "-O2");
  else
    CMD_APPEND(cmd, "-g -O1 -DDEBUG");
}

Cmd mkdir_bin() {
  Cmd cmd = {0};
  CMD_APPEND(&cmd, "mkdir -p bin/");
  return cmd;
}

Cmd build_main() {
  Cmd cmd = {0};
  cc(&cmd);
  cflags(&cmd);
  CMD_APPEND(&cmd, "src/main.c");
  CMD_APPEND(&cmd, "-c -o bin/main.o");
  return cmd;
}

Cmd link() {
  Cmd cmd = {0};
  cc(&cmd);
  CMD_APPEND(&cmd, "bin/main.o");
  CMD_APPEND(&cmd, "-o bin/ml");
  return cmd;
}

int main(int argc, char **argv) {
  yeb_bootstrap();
  Options opts = parse_argv(argc, argv);
  is_release = opts_get(opts, "--release").exists;
  execute(mkdir_bin());
  execute(build_main());
  execute(link());
  return 0;
}
