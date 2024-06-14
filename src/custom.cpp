#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <ebbrt/SpinBarrier.h>
#include <ebbrt/Debug.h>
#include <ebbrt/native/Cpu.h>
#include <ebbrt/native/Clock.h>
#include <ebbrt/EventManager.h>
#include <ebbrt/SharedIOBufRef.h>
#include <ebbrt/StaticIOBuf.h>
#include <ebbrt/UniqueIOBuf.h>
#include <ebbrt/StaticSharedEbb.h>
#include <ebbrt/AtomicUniquePtr.h>
#include <ebbrt/CacheAligned.h>
#include <ebbrt/SpinLock.h>
#include <ebbrt/Future.h>
#include <ebbrt/native/Msr.h>
#include <ebbrt/native/Net.h>
#include <ebbrt/native/NetTcpHandler.h>
#include <ebbrt/native/RcuTable.h>
#include <ebbrt/native/IxgbeDriver.h>
#include <ebbrt/native/Trace.h>

#define IA32_PERF_STATUS 0x198
#define IA32_PERF_CTL    0x199
#define IA32_MISC_ENABLE 0x1A0

void AppMain(void) {
  uint32_t ncores = static_cast<uint32_t>(ebbrt::Cpu::Count());  
  for (uint32_t i = 0; i < ncores; i++) {
    ebbrt::Promise<void> p;
    auto f = p.GetFuture();
    ebbrt::event_manager->SpawnRemote(
      [ncores, i, &p] () mutable {
	// disables turbo boost, thermal control circuit
	ebbrt::msr::Write(IA32_MISC_ENABLE, 0x850089);
	
        // same p state as Linux with performance governor
	ebbrt::msr::Write(IA32_PERF_CTL, 0x100001800);

	uint64_t ii, jj, sum=0, sum2=0;
	for(ii=0;ii<ncores;ii++) {	  
	  for(jj=0;jj<IXGBE_LOG_SIZE;jj++) {
	    sum += 0;
	  }
	  
	  uint8_t* ptr = bsendbufs[ii]->MutData();
	  for(jj=0;jj<IXGBE_MAX_DATA_PER_TXD;jj++) {
	    sum2 += ptr[ii];
	  }
	}

	ebbrt::kprintf_force("Cpu=%u Sum=%llu Sum2=%llu\n", i, sum, sum2);
	p.SetValue();
      }, i);
    f.Block();
  }

  size_t nt = static_cast<size_t>(ebbrt::Cpu::Count()); 
  size_t mainCPU = ebbrt::Cpu::GetMine();
  ebbrt::EventManager::EventContext context;
  std::atomic<size_t> count(0);
  static ebbrt::SpinBarrier bar(nt);  
  for (int i = 0; i < static_cast<int>(nt); i++) {
    ebbrt::event_manager->SpawnRemote([&context, &count, i, nt, mainCPU]() {
      ebbrt::kprintf("AppMain SpawnRemote %d\n", static_cast<int>(ebbrt::Cpu::GetMine()));
      count ++;
      bar.Wait();
      while(count < nt);
      if (ebbrt::Cpu::GetMine() == mainCPU)
	ebbrt::event_manager->ActivateContext(std::move(context));
    }, i);
  }    
  ebbrt::event_manager->SaveContext(context);  
  
  uint32_t mcore = static_cast<uint32_t>(ebbrt::Cpu::GetMine());
  ebbrt::kprintf_force("custom core %u\n", mcore);

  //const int sizey = 4096;
  //const int sizex = 11008;
  //const int sizez = 128;
  const int sizey = 2;
  const int sizex = 2;
  const int sizez = 1;
  const ggml_type qtype = GGML_TYPE_Q4_1;

  size_t ctx_size = 0;
  ctx_size += ggml_row_size(GGML_TYPE_F32, sizex*sizey);
  ctx_size += ggml_row_size(GGML_TYPE_F32, sizex*sizey);
  ctx_size += ggml_row_size(GGML_TYPE_F32, sizex*sizez);
  ctx_size += ggml_row_size(qtype,         sizex*sizey);
  ctx_size += ggml_row_size(qtype,         sizex*sizey);
  ctx_size += ggml_row_size(GGML_TYPE_F32, sizex*sizey); // BLAS
  ctx_size += ggml_row_size(GGML_TYPE_F32, sizex*sizey); // BLAS
  ctx_size += 1024*1024*16;
  
  ebbrt::kprintf("Allocating Memory of size %ld bytes, %ld MB\n",ctx_size, (ctx_size/1024/1024));

  struct ggml_init_params params {
    ctx_size,
    NULL,
    0, // NOTE: this should be false when using the legacy API
  };
  
  struct ggml_context *ctx = ggml_init(params);  
  struct ggml_tensor * m11 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizex, sizey);
  ggml_set_f32(m11, 1.0f);
  struct ggml_tensor * m2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizex, sizez);
  ggml_set_f32(m2, 2.0f);

  struct ggml_cgraph * gf = ggml_new_graph(ctx);    
  ebbrt::kprintf("\n------ Test 1 - Matrix Mult via F32 code\n");
  struct ggml_tensor * m11xm2 = ggml_mul_mat(ctx, m11, m2);
  ggml_build_forward_expand(gf, m11xm2);
  ggml_graph_compute_with_ctx(ctx, gf, 2);  
  ggml_free(ctx);
  
  // const int rows_A = 4, cols_A = 2;  
  // float matrix_A[rows_A * cols_A] = {
  //   2, 8,
  //   5, 1,
  //   4, 2,
  //   8, 6
  // };
  
  // const int rows_B = 3, cols_B = 2;
  // float matrix_B[rows_B * cols_B] = {
  //   10, 5,
  //   9, 9,
  //   5, 4
  // };

  // size_t ctx_size = 0;
  // {
  //   ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F32); // tensor a
  //   ctx_size += rows_B * cols_B * ggml_type_size(GGML_TYPE_F32); // tensor b
  //   ctx_size += 2 * ggml_tensor_overhead(), // tensors
  //   ctx_size += ggml_graph_overhead(); // compute graph
  //   ctx_size += 1024; // some overhead
  // }
  
  // struct ggml_init_params params {
  //   /*.mem_size   =*/ ctx_size,
  //   /*.mem_buffer =*/ NULL,
  //   /*.no_alloc   =*/ 0, // NOTE: this should be false when using the legacy API
  // };

  // struct ggml_context * ctx = ggml_init(params);
  // struct ggml_tensor * a;
  // struct ggml_tensor * b;
  // a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
  // b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);

  // memcpy(a->data, a, ggml_nbytes(a));
  // memcpy(b->data, b, ggml_nbytes(b));

  // struct ggml_cgraph  * gf = ggml_new_graph(ctx);
  // struct ggml_tensor * result = ggml_mul_mat(ctx, a, b);
  // ggml_build_forward_expand(gf, result);
    
  // ggml_graph_compute_with_ctx(ctx, gf, 2);
  
  // // free memory
  // ggml_free(ctx);      
}
