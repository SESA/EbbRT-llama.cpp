#!/bin/bash
set -x

CC=~/sysroot_llm/native/usr/bin/x86_64-pc-ebbrt-gcc
CXX=~/sysroot_llm/native/usr/bin/x86_64-pc-ebbrt-g++
SYSROOT=~/sysroot_llm/native

$CC --sysroot=$SYSROOT -Isrc -Isrc/common -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_  -fPIC -O4 -flto -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -march=native -mtune=native -Wdouble-promotion    -c src/ggml.c -o src/ggml.o
#-std=c++11,gnu+14

$CXX --sysroot=$SYSROOT -fPIC -O4 -flto -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Isrc -Isrc/common -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_ -c src/llama.cpp -o src/llama.o

$CXX --sysroot=$SYSROOT -fPIC -O4 -flto -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Isrc -Isrc/common -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_ -c src/common/common.cpp -o src/common.o

$CXX --sysroot=$SYSROOT -fPIC -O4 -flto -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Isrc -Isrc/common -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_ -c src/common/sampling.cpp -o src/sampling.o

$CXX --sysroot=$SYSROOT -fPIC -O4 -flto -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Isrc -Isrc/common -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_ -c src/common/grammar-parser.cpp -o src/grammar-parser.o

$CXX --sysroot=$SYSROOT -fPIC -O4 -flto -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Isrc -Isrc/common -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_ -c src/common/build-info.cpp -o src/build-info.o

$CXX --sysroot=$SYSROOT -fPIC -O4 -flto -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Isrc -Isrc/common -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_ -c src/common/json-schema-to-grammar.cpp -o src/json-schema-to-grammar.o

$CXX --sysroot=$SYSROOT -fPIC -O4 -flto -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Isrc -Isrc/common -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_ -c src/sgemm.cpp -o src/sgemm.o

$CC --sysroot=$SYSROOT -Isrc -Isrc/common -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_   -fPIC -O4 -flto -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -march=native -mtune=native -Wdouble-promotion    -c src/ggml-alloc.c -o src/ggml-alloc.o

$CC --sysroot=$SYSROOT -Isrc -Isrc/common -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_   -fPIC -O4 -flto -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -march=native -mtune=native -Wdouble-promotion    -c src/ggml-backend.c -o src/ggml-backend.o

$CC --sysroot=$SYSROOT -Isrc -Isrc/common -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_   -fPIC -O4 -flto -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -march=native -mtune=native -Wdouble-promotion     -c src/ggml-quants.c -o src/ggml-quants.o

$CXX --sysroot=$SYSROOT -fPIC -O4 -flto -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Isrc -Isrc/common -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_ -c src/unicode.cpp -o src/unicode.o

$CXX --sysroot=$SYSROOT -fPIC -O4 -flto -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Isrc -Isrc/common -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_  -c src/unicode-data.cpp -o src/unicode-data.o

$CXX --sysroot=$SYSROOT -fPIC -O4 -flto -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Isrc -Isrc/common -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_ -c src/simple.cpp -o src/simple.o

$CXX --sysroot=$SYSROOT -fPIC -O4 -flto -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Isrc -Isrc/common -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_  src/ggml.o src/llama.o src/common.o src/sampling.o src/grammar-parser.o src/build-info.o src/json-schema-to-grammar.o src/sgemm.o src/ggml-alloc.o src/ggml-backend.o src/ggml-quants.o src/unicode.o src/unicode-data.o src/simple.o -o simple
