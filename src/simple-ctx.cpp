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

// Vol. 3C Page 35-3, Table 35-2. IA-32 Architectural MSRs
#define IA32_APIC_BASE 0x1B
#define IA32_FEATURE_CONTROL 0x3A
#define IA32_SMM_MONITOR_CTL 0x9B
#define IA32_MTRRCAP 0xFE
#define IA32_SYSENTER_CS 0x174
#define IA32_MCG_CAP 0x179
#define IA32_PERF_STATUS 0x198
#define IA32_PERF_CTL    0x199
#define IA32_CLOCK_MODULATION 0x19A
#define IA32_THERM_INTERRUPT 0x19B
#define IA32_THERM_STATUS 0x19C
#define IA32_MISC_ENABLE 0x1A0
#define IA32_PACKAGE_THERM_STATUS 0x1B1
#define IA32_PACKAGE_THERM_INTERRUPT 0x1B2
#define IA32_PLATFORM_DCA_CAP 0x1F8
#define IA32_CPU_DCA_CAP 0x1F9
#define IA32_DCA_0_CAP 0x1FA

// Vol. 3C Page 35-143, Table 35-18. Intel Sandy Bridge MSRs
#define MSR_PLATFORM_INFO 0xCE
#define MSR_PKG_CST_CONFIG_CONTROL 0xE2
#define MSR_PMG_IO_CAPTURE_BASE 0xE4
#define MSR_TEMPERATURE_TARGET 0x1A2
#define MSR_MISC_FEATURE_CONTROL 0x1A4
#define MSR_PEBS_LD_LAT 0x3F6
#define MSR_PKG_C3_RESIDENCY 0x3F8
#define MSR_PKG_C6_RESIDENCY 0x3F9

// TODO
#define MSR_PKGC3_IRTL 0x60A
#define MSR_PKGC6_IRTL 0x60B

// This is a simple model with two tensors a and b
struct simple_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;

    // the context to define the tensor information (dimensions, size, memory data)
    struct ggml_context * ctx;
};

#define TIME_CONVERSION_khz 2394230*1000
inline static uint64_t rdtscp(void) {
  uint64_t tsc;
  asm volatile("rdtsc;"
               "shl $32,%%rdx;"
	       "or %%rdx,%%rax"
               : "=a"(tsc)
               :
               : "%rcx", "%rdx");
  return tsc;
}

void load_model(simple_model & model, float * a, float * b, int rows_A, int cols_A, int rows_B, int cols_B);
struct ggml_cgraph * build_graph(const simple_model& model);
struct ggml_tensor * compute(const simple_model & model);

// initialize the tensors of the model in this case two matrices 2x2
void load_model(simple_model & model, float * a, float * b, int rows_A, int cols_A, int rows_B, int cols_B) {
    size_t ctx_size = 0;
    {
        ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F32); // tensor a
        ctx_size += rows_B * cols_B * ggml_type_size(GGML_TYPE_F32); // tensor b
        ctx_size += 2 * ggml_tensor_overhead(), // tensors
        ctx_size += ggml_graph_overhead(); // compute graph
        ctx_size += 1024; // some overhead
    }

    struct ggml_init_params params {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false, // NOTE: this should be false when using the legacy API
    };

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_A, rows_A);
    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_B, rows_B);

    memcpy(model.a->data, a, ggml_nbytes(model.a));
    memcpy(model.b->data, b, ggml_nbytes(model.b));
}

// build the compute graph to perform a matrix multiplication
struct ggml_cgraph * build_graph(const simple_model& model) {
    struct ggml_cgraph  * gf = ggml_new_graph(model.ctx);

    // result = a*b^T
    struct ggml_tensor * result = ggml_mul_mat(model.ctx, model.a, model.b);

    ggml_build_forward_expand(gf, result);
    return gf;
}

// compute with backend
struct ggml_tensor * compute(const simple_model & model) {
    struct ggml_cgraph * gf = build_graph(model);
    
    //int n_threads = static_cast<int>(ebbrt::Cpu::Count());; // number of threads to perform some operations with multi-threading
    int n_threads = 2;
    ggml_graph_compute_with_ctx(model.ctx, gf, n_threads);

    // in this case, the output tensor is the last one in the graph
    return gf->nodes[gf->n_nodes - 1];
}

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
  ebbrt::kprintf_force("simple-ctx core %u\n", mcore);
  for (int t = 0; t < 2; t++) {
    uint64_t tsc_start = rdtscp();
    for (int i = 0; i < 1; i++) {
      // initialize data of matrices to perform matrix multiplication
      const int rows_A = 4, cols_A = 2;
    
      float matrix_A[rows_A * cols_A] = {
	2, 8,
	5, 1,
	4, 2,
	8, 6
      };
    
      const int rows_B = 3, cols_B = 2;
      float matrix_B[rows_B * cols_B] = {
	10, 5,
	9, 9,
	5, 4
      };
    
      simple_model model;
      load_model(model, matrix_A, matrix_B, rows_A, cols_A, rows_B, cols_B);
    
      // perform computation in cpu
      volatile struct ggml_tensor * result = compute(model);             
	    
      // free memory
      ggml_free(model.ctx);
    }
    uint64_t tsc_stop = rdtscp();
    uint64_t tsc_diff = tsc_stop - tsc_start;
    float tdiff = (tsc_diff/(float)TIME_CONVERSION_khz)/1000000.0;
    ebbrt::kprintf("TSC: %.3lf seconds\n", tdiff);
  }	  
}
