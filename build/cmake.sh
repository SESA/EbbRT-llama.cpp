set -x

EBBRT_SYSROOT=~/sysroot_vm/native cmake -DCMAKE_TOOLCHAIN_FILE=~/sysroot_vm/native/usr/misc/ebbrt.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_NAME=EbbRT ..
