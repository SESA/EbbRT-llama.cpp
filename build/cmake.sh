set -x

EBBRT_SYSROOT=~/sysroot_llm/native cmake -DCMAKE_TOOLCHAIN_FILE=~/sysroot_llm/native/usr/misc/ebbrt.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_NAME=EbbRT ..
