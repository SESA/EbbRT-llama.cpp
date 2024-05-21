#!/bin/bash
set -x

CC=~/sysroot_llm/native/usr/bin/x86_64-pc-ebbrt-gcc

$CC  -Isrc -Isrc/common -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_ -std=c11   -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -march=native -mtune=native -Wdouble-promotion    -c src/ggml.c -o src/ggml.o
