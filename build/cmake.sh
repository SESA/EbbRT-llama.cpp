set -x

EBBRT_SYSROOT=~/sysroot/native cmake -DCMAKE_TOOLCHAIN_FILE=~/sysroot/native/usr/misc/ebbrt.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_NAME=EbbRT ..
