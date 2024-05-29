project("ebbrt-llama.cpp" CXX)
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g3")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O4 -flto -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -D_EBBRT_ -I../src -I../src/common")

include_directories(${PROJECT_SRC_DIR})
set_source_files_properties(src/ggml.c PROPERTIES LANGUAGE C)
add_executable(test.elf src/llama2.cpp src/test.cpp)

add_custom_command(TARGET test.elf POST_BUILD 
  COMMAND objcopy -O elf32-i386 test.elf test.elf32)
