project("benchmark-matmult" LANGUAGES C CXX)
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(CMAKE_C_FLAGS_DEBUG          "-O0 -g3")
set(CMAKE_C_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE        "-O4 -flto -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g3")

set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g3")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g3")

if( ${CMAKE_SYSTEM_NAME} STREQUAL "Linux" )
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DGGML_USE_LLAMAFILE -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -pthread -march=native -mtune=native -fopenmp -I../src -I../src/common")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=gnu++14 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_LLAMAFILE -pthread -fopenmp -march=native -mtune=native -I../src -I../src/common")
  
  add_executable(benchmark-matmult src/ggml.c src/common/build-info.cpp src/sgemm.cpp src/ggml-alloc.c src/ggml-backend.c src/ggml-quants.c src/unicode.cpp src/unicode-data.cpp src/benchmark-matmult.cpp)
  
elseif( ${CMAKE_SYSTEM_NAME} STREQUAL "EbbRT" )
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -D_EBBRT_ -DGGML_USE_LLAMAFILE -D_GNU_SOURCE -DNDEBUG -march=native -mtune=native -I../src -I../src/common")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=gnu++14 -D_EBBRT_ -D_GNU_SOURCE -DNDEBUG -DGGML_USE_LLAMAFILE -march=native -mtune=native -I../src -I../src/common")
  
  add_executable(benchmark-matmult.elf src/ggml.c src/ggml.cpp src/common/build-info.cpp src/sgemm.cpp src/ggml-alloc.c src/ggml-backend.c src/ggml-quants.c src/unicode.cpp src/unicode-data.cpp src/benchmark-matmult.cpp)
  add_custom_command(TARGET benchmark-matmult.elf POST_BUILD COMMAND objcopy -O elf32-i386 benchmark-matmult.elf benchmark-matmult.elf32)
  
  add_executable(simple.elf src/ggml.c src/ggml.cpp src/llama.cpp src/common/common.cpp src/common/sampling.cpp src/common/grammar-parser.cpp src/common/build-info.cpp src/common/json-schema-to-grammar.cpp src/sgemm.cpp src/ggml-alloc.c src/ggml-backend.c src/ggml-quants.c src/unicode.cpp src/unicode-data.cpp src/benchmark-matmult.cpp)
  add_custom_command(TARGET simple.elf POST_BUILD
    COMMAND objcopy -O elf32-i386 simple.elf simple.elf32)  
else()
  message(FATAL_ERROR "System name unsupported: ${CMAKE_SYSTEM_NAME}")
endif()


