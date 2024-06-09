set -x

EBBRT_SYSROOT=~/sysroot_avx/native cmake -DCMAKE_TOOLCHAIN_FILE=~/sysroot_avx/native/usr/misc/ebbrt.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_NAME=EbbRT ..
