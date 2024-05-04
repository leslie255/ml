#include "yeb.h"

void cc(Cmd *cmd) {
  CMD_APPEND(cmd, "clang");
}

void cflags(Cmd *cmd) {
  CMD_APPEND(cmd, "-Wall -Wextra --std=gnu17 -O2");
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

int main() {
  yeb_bootstrap();
  execute(mkdir_bin());
  execute(build_main());
  execute(link());
  return 0;
}
