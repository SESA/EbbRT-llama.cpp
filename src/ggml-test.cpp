#include <ebbrt/Debug.h>
#include <ebbrt/EbbAllocator.h>
#include <ebbrt/native/Net.h>
#include <ebbrt/native/Msr.h>
#include <ebbrt/native/EventManager.h>
#include <ebbrt/native/Cpu.h>
#include <ebbrt/Future.h>
#include <ebbrt/SpinBarrier.h>

typedef void * thread_ret_t;
typedef thread_ret_t (*FunctionPtr)(void *);
//typedef void (*FunctionPtr)(void*);
extern "C" void eprint(char *s) {
  ebbrt::kprintf("%s\n", s);
}

// Define the C++ function that takes a function pointer and data
extern "C" void cppFunction(FunctionPtr ptr, void* workers, int n_threads, size_t workerSize) {
  ebbrt::kprintf("cppFunction START\n");
  size_t mainCPU = ebbrt::Cpu::GetMine();
  ebbrt::EventManager::EventContext context;
  std::atomic<size_t> count(0);
  static ebbrt::SpinBarrier bar(n_threads);
  
  char* workerPtr = static_cast<char*>(workers);
  
  for (int i = 0; i < n_threads; i++) {
    ebbrt::event_manager->SpawnRemote([&context, &count, i, n_threads, mainCPU, ptr, workerPtr, workerSize]() {
      ebbrt::kprintf("cppFunction SpawnRemote %d, workerPtr %p, workerSize %u \n", static_cast<int>(ebbrt::Cpu::GetMine()), workerPtr, workerSize);

      ptr(static_cast<void*>(workerPtr));
      count ++;
      bar.Wait();
      while(count < n_threads);
      if (ebbrt::Cpu::GetMine() == mainCPU)
	ebbrt::event_manager->ActivateContext(std::move(context));
    }, i);
    workerPtr += workerSize;
  }
  ebbrt::event_manager->SaveContext(context);
  ebbrt::kprintf("cppFunction END\n");
}



