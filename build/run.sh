sudo qemu-system-x86_64 -m 1G -smp cpus=4 -cpu host -serial stdio -display none -enable-kvm -kernel benchmark-matmult.elf32
