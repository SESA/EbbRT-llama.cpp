#include "common.h"
#include "ggml.h"

#include <locale.h>
#include <assert.h>
#include <math.h>
#include <cstring>
#include <cstdio>
#include <cinttypes>
#include <unordered_map>
#include <queue>
#include <string.h>
#include <cassert>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>

#ifdef _EBBRT_
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
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define TIME_CONVERSION_khz 2394230*1000
#define IA32_PERF_CTL    0x199
#define IA32_MISC_ENABLE 0x1A0

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

static float tensor_sum_elements(const ggml_tensor * tensor) {
    double sum = 0;
    if (tensor->type == GGML_TYPE_F32) {
        for (int j = 0; j < tensor->ne[1]; j++) {
            for (int k = 0; k < tensor->ne[0]; k++) {
                sum += ((float *) tensor->data)[j*tensor->ne[0] + k];
            }
        }
    }
    return sum;
}

static void tensor_dump(const ggml_tensor * tensor, const char * name) {
#ifndef _EBBRT_
    printf("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi) - ", name,
        tensor->type, ggml_type_name(tensor->type),
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->nb[0], tensor->nb[1], tensor->nb[2]);
    float sum = tensor_sum_elements(tensor);
    printf("Sum of tensor %s is %6.2f\n", name, sum);
#else
    ebbrt::kprintf("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5u, %5u, %5u) - ", name,
        tensor->type, ggml_type_name(tensor->type),
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->nb[0], tensor->nb[1], tensor->nb[2]);
    float sum = tensor_sum_elements(tensor);
    ebbrt::kprintf("Sum of tensor %s is %6.2f\n", name, sum);
#endif
}

#define TENSOR_DUMP(tensor) tensor_dump(tensor, #tensor)

struct benchmark_params_struct {
    int32_t n_threads     = 1;
    int32_t n_iterations  = 10;
};

static void print_usage(int /*argc*/, char ** argv, struct benchmark_params_struct params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -i N, --iter N     number of iterations to use during computation (default: %d)\n", params.n_iterations);
    fprintf(stderr, "\n");
}

void AppMain() {
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
  
  uint32_t mcore = static_cast<uint32_t>(ebbrt::Cpu::GetMine());
  ebbrt::kprintf_force("AppMain() Core %u\n", mcore);
  struct benchmark_params_struct benchmark_params;
  benchmark_params.n_threads = 1;
  benchmark_params.n_iterations = 10;

  ebbrt::kprintf("Starting Test\n");

  // create the ggml context
  struct ggml_context * ctx;
  const int sizey = 4096;
  const int sizex = 11008;
  const int sizez = 128;

  // TODO: perform the bench for all types or for a user specified type
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
  
  ebbrt::kprintf("Allocating Memory of size %u bytes, %u MB\n",ctx_size, (ctx_size/1024/1024));

  struct ggml_init_params params = {
    ctx_size,
    NULL,
    0
  };
  
  ctx = ggml_init(params);
  if (!ctx) {
    ebbrt::kprintf("%s: ggml_init() failed\n", __func__);
    EBBRT_UNIMPLEMENTED();
  }


  ebbrt::kprintf("Creating new tensors\n");
  // printf("Creating new tensor m1\n");
  struct ggml_tensor * m11 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizex, sizey);
  ggml_set_f32(m11, 1.0f);
  
  // printf("Creating new tensor m1\n");
  struct ggml_tensor * m12 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizex, sizey);
  ggml_set_f32(m12, 1.5f);
  
  // printf("Creating new tensor m2\n");
  struct ggml_tensor * m2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizex, sizez);
  ggml_set_f32(m2, 2.0f);
  
  printf("\n------ Test 1 - Matrix Mult via F32 code\n");
  // printf("Creating new tensor m11xm2\n");
  struct ggml_tensor * m11xm2 = ggml_mul_mat(ctx, m11, m2);
  
  // printf("Creating compute graph\n");
  struct ggml_cgraph * gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, m11xm2);
  
  ebbrt::kprintf("n_threads=%i\n", benchmark_params.n_threads);
  
  TENSOR_DUMP(m11);
  TENSOR_DUMP(m2);
  
  std::vector<uint8_t> work_buffer;
  
  ggml_graph_compute_helper(work_buffer, gf, benchmark_params.n_threads);
  
  TENSOR_DUMP(gf->nodes[0]);
  
  ebbrt::kprintf("\n------ Test 2 - Matrix Mult via %s code\n", ggml_type_name(qtype));
  
  int32_t nelements = sizex*sizey;
  
  // Set up a the benchmark matrices
  // printf("Creating new tensor q11 & Running quantize\n");
  struct ggml_tensor * q11 = ggml_new_tensor_2d(ctx, qtype, sizex, sizey);
  ggml_quantize_chunk(qtype, (const float *) m11->data, q11->data, 0, nelements/m11->ne[0], m11->ne[0], nullptr);

  // Set up a the compute graph
  // printf("Creating new tensor q31\n");
  struct ggml_tensor * q31 = ggml_mul_mat(ctx, q11, m2);
  
  // printf("Creating compute graph\n");
  struct ggml_cgraph * gf31 = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf31, q31);
  
  // Set up a second graph computation to make sure we override the CPU cache lines
  // printf("Creating new tensor q12 & Running quantize\n");
  struct ggml_tensor * q12 = ggml_new_tensor_2d(ctx, qtype, sizex, sizey);
  ggml_quantize_chunk(qtype, (const float *) m12->data, q12->data, 0, nelements/m12->ne[0], m12->ne[0], nullptr);
  
  // printf("Creating new tensor q32\n");
  struct ggml_tensor * q32 = ggml_mul_mat(ctx, q12, m2);
  
  //printf("Creating compute graph\n");
  struct ggml_cgraph * gf32 = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf32, q32);
  ebbrt::kprintf("n_threads=%i\n", benchmark_params.n_threads);

  const int dimx = sizex;
  const int dimy = sizey;
  const int dimz = sizez;
  long long int flops_per_dot_product = dimy + dimy;
  long long int flops_per_matrix = flops_per_dot_product * dimx * dimz; ;
  ebbrt::kprintf("Matrix Multiplication of (%i,%i,%i) x (%i,%i,%i) - about %6.2f gFLOPS\n\n", sizex, sizey, 1, sizex, sizez, 1, 1.0f*flops_per_matrix / 1000 / 1000 / 1000);
  

  // Let's use the F32 result from above as a reference for the quantized multiplication
  float sum_of_F32_reference = tensor_sum_elements(gf->nodes[0]);
  
  ebbrt::kprintf("Iteration;NThreads; SizeX; SizeY; SizeZ; Required_FLOPS; Elapsed_u_Seconds; gigaFLOPS\n");
  ebbrt::kprintf("=====================================================================================\n");

  double  gflops_sum = 0;
  for (int i=0;i<benchmark_params.n_iterations ;i++) {
    
    long long int start = ggml_time_us();
    //printf("Running ggml_graph_compute\n");
    uint64_t tsc_start = ebbrt::trace::rdtsc();
    ggml_graph_compute_helper(work_buffer, gf31, benchmark_params.n_threads);
    uint64_t tsc_stop = ebbrt::trace::rdtsc();
    uint64_t tsc_diff = tsc_stop - tsc_start;
    float tdiff = (tsc_diff/(float)TIME_CONVERSION_khz)/1000000.0;
    ebbrt::kprintf("TSC: %.3lf seconds\n", tdiff);
	
    long long int stop = ggml_time_us();
    long long int usec = stop-start;
    double gflops = (double)(flops_per_matrix)/usec/1000.0;
    gflops_sum += gflops;
    ebbrt::kprintf("%9i;%8i;%6i;%6i;%6i;%15lli;%18lli;%10.2f\n",
		   i,
		   benchmark_params.n_threads,
		   sizex, sizey, sizez, flops_per_matrix,
		   usec,gflops);
          
      // Check that the matrix multiplication result is in the right ballpark
      // We cannot use the exact value from the F32 multiplication because the quantizuation will be slightly different
      float sum_of_Q4_result = tensor_sum_elements(gf31->nodes[0]);
    float delta = std::abs(sum_of_Q4_result - sum_of_F32_reference);
    float allowed_delta = (sum_of_F32_reference) / 1000 / 1000; //  Let's accept an epsilon of 10^-6
  
    if (delta > allowed_delta)  {
      ebbrt::kprintf("\nABORT - ERROR in Matrix Multiplication result - expected %6.2f, got %6.2f (delta %6.2f > allowed_delta %6.2f)\n",
		     sum_of_F32_reference,
		     sum_of_Q4_result,
		     delta,
		     allowed_delta
		     );
      EBBRT_UNIMPLEMENTED();
      //exit(0);
    }
    
    // Running a different graph computation to make sure we override the CPU cache lines
    ggml_graph_compute_helper(work_buffer, gf32, benchmark_params.n_threads);
  }
  ebbrt::kprintf("\n");
  ebbrt::kprintf("Average%78.2f\n",gflops_sum/((double)benchmark_params.n_iterations));
  ebbrt::kprintf("=====================================================================================\n");
}
