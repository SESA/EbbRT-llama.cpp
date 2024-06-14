project("ebbrt-llama.cpp" LANGUAGES C CXX)
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(CMAKE_C_FLAGS_DEBUG          "-O0 -g3")
set(CMAKE_C_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE        "-O4 -flto -DNDEBUG")
#set(CMAKE_C_FLAGS_RELEASE        "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGGML_USE_LLAMAFILE -D_EBBRT_ -I../src -I../src/common -Ofast -march=native -mtune=native")

#-fPIC -Ofast -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -march=native -mtune=native -Wdouble-promotion

set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g3")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -flto -DNDEBUG")
#set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -DGGML_USE_LLAMAFILE -D_EBBRT_ -I../src -I../src/common -Ofast -march=native -mtune=native")
#-fPIC -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -Ofast -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi

#cc  -I. -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE  -std=c11   -fPIC -Ofast -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -pthread -march=native -mtune=native -Wdouble-promotion    -c ggml.c -o ggml.o
#c++ -std=c++11 -fPIC -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread  -Ofast -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -I. -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE  -c sgemm.cpp -o sgemm.o

include_directories(${PROJECT_SRC_DIR})
add_executable(simple.elf src/ggml.c src/llama.cpp src/common/common.cpp src/common/sampling.cpp src/common/grammar-parser.cpp src/common/build-info.cpp src/common/json-schema-to-grammar.cpp src/sgemm.cpp src/ggml-alloc.c src/ggml-backend.c src/ggml-quants.c src/unicode.cpp src/unicode-data.cpp src/simple.cpp)
add_custom_command(TARGET simple.elf POST_BUILD 
  COMMAND objcopy -O elf32-i386 simple.elf simple.elf32)

add_executable(benchmark-matmult.elf src/ggml.c src/ggml-test.cpp src/llama.cpp src/common/common.cpp src/common/sampling.cpp src/common/grammar-parser.cpp src/common/build-info.cpp src/common/json-schema-to-grammar.cpp src/sgemm.cpp src/ggml-alloc.c src/ggml-backend.c src/ggml-quants.c src/unicode.cpp src/unicode-data.cpp src/benchmark-matmult.cpp)
add_custom_command(TARGET benchmark-matmult.elf POST_BUILD 
  COMMAND objcopy -O elf32-i386 benchmark-matmult.elf benchmark-matmult.elf32)

add_executable(simple-ctx.elf src/ggml.c src/ggml-test.cpp src/sgemm.cpp src/ggml-alloc.c src/ggml-backend.c src/ggml-quants.c src/unicode.cpp src/unicode-data.cpp src/simple-ctx.cpp)
add_custom_command(TARGET simple-ctx.elf POST_BUILD 
  COMMAND objcopy -O elf32-i386 simple-ctx.elf simple-ctx.elf32)

add_executable(custom.elf src/ggml.c src/ggml-test.cpp src/sgemm.cpp src/ggml-alloc.c src/ggml-backend.c src/ggml-quants.c src/unicode.cpp src/unicode-data.cpp src/custom.cpp)
add_custom_command(TARGET custom.elf POST_BUILD 
  COMMAND objcopy -O elf32-i386 custom.elf custom.elf32)
