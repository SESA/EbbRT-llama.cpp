project("ebbrt-llama.cpp" LANGUAGES C CXX)
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(CMAKE_C_FLAGS_DEBUG          "-O0 -g3")
set(CMAKE_C_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE        "-O4 -flto -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGGML_USE_LLAMAFILE -D_EBBRT_ -I../src -I../src/common")

set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g3")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -DGGML_USE_LLAMAFILE -D_EBBRT_ -I../src -I../src/common")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O4 -flto -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -Wno-array-bounds -Wno-format-truncation -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_ -I../src -I../src/common")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -DCONFIG_H=config/config-perf.h -DNDB_MASSTREE -DNO_MYSQL -include ../src/masstree/config.h")

include_directories(${PROJECT_SRC_DIR})
add_executable(simple.elf src/ggml.c src/llama.cpp src/common/common.cpp src/common/sampling.cpp src/common/grammar-parser.cpp src/common/build-info.cpp src/common/json-schema-to-grammar.cpp src/sgemm.cpp src/ggml-alloc.c src/ggml-backend.c src/ggml-quants.c src/unicode.cpp src/unicode-data.cpp src/simple.cpp)

add_custom_command(TARGET simple.elf POST_BUILD 
  COMMAND objcopy -O elf32-i386 simple.elf simple.elf32)
