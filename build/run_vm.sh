#
# To send model:
#  cat llama-2-7b-chat.Q2_K.gguf | socat - TCP:10.10.1.10:8888
#
# To process model:
#   echo "hu" | socat - TCP:10.10.1.10:8889
#

sudo qemu-system-x86_64 -m 50G -smp cpus=4 -cpu host -serial stdio -display none -enable-kvm -kernel simple.elf32 -device virtio-net-pci,mq=on,netdev=network0,mac=52:55:00:d1:55:01 -netdev tap,id=network0,ifname=tap0,script=no,downscript=no,vhost=on,queues=4
