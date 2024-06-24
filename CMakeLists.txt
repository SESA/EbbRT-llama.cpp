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
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DGGML_USE_LLAMAFILE -pthread -march=native -mtune=native -fopenmp -I../src -I../src/common -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=gnu++14 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_LLAMAFILE -I../src -I../src/common -pthread -fopenmp -march=native -mtune=native")
  
  add_executable(benchmark-matmult src/ggml.c src/common/build-info.cpp src/sgemm.cpp src/ggml-alloc.c src/ggml-backend.c src/ggml-quants.c src/unicode.cpp src/unicode-data.cpp src/benchmark-matmult.cpp)
else()
  message(FATAL_ERROR "System name unsupported: ${CMAKE_SYSTEM_NAME}")
endif()


