#include <ebbrt/Debug.h>
#include <ebbrt/EbbAllocator.h>
#include <ebbrt/native/Clock.h>
#include <ebbrt/native/Net.h>
#include <ebbrt/native/Msr.h>
#include <ebbrt/native/EventManager.h>
#include <ebbrt/native/Cpu.h>
#include <ebbrt/Future.h>
#include <ebbrt/SpinBarrier.h>

#include "ggml.h"

typedef void * thread_ret_t;
typedef thread_ret_t (*FunctionPtr)(void *);
//typedef void (*FunctionPtr)(void*);
extern "C" void eassert(char* s, int line, bool b) {
  ebbrt::kprintf("%s: %d\n", s, line);
  kassert(b);
}
extern "C" void eprint(char *s) {
  ebbrt::kprintf("%s", s);
}

struct ggml_compute_state_shared {
  struct ggml_cgraph * cgraph;
  struct ggml_cplan * cplan;
  int n_threads;
  int n_barrier;
  int n_barrier_passed;
  ggml_abort_callback abort_callback;
  void * abort_callback_data;
  std::atomic<int> current_chunk;
  int ec;  
};

struct ggml_compute_state {
  int thrd;
  int ith;
  struct ggml_compute_state_shared * shared;
};


struct ggml_compute_params {
  int ith;
  int nth;
  size_t wsize;
  uint8_t * wdata;
  struct ggml_compute_state_shared * shared;
};

// static void eggml_compute_forward_mul_mat(
//         const struct ggml_compute_params * params,
//               struct ggml_tensor * dst) {

//     const struct ggml_tensor * src0 = dst->src[0];
//     const struct ggml_tensor * src1 = dst->src[1];

//     GGML_TENSOR_BINARY_OP_LOCALS

//     const int ith = params->ith;
//     const int nth = params->nth;

//     const enum ggml_type type = src0->type;

//     enum ggml_type    const vec_dot_type          = type_traits[type].vec_dot_type;
//     ggml_from_float_t const from_float_to_vec_dot = type_traits[vec_dot_type].from_float;
//     int64_t           const vec_dot_num_rows      = type_traits[type].nrows;

//     GGML_ASSERT(ne0 == ne01);
//     GGML_ASSERT(ne1 == ne11);
//     GGML_ASSERT(ne2 == ne12);
//     GGML_ASSERT(ne3 == ne13);

//     // we don't support permuted src0 or src1
//     GGML_ASSERT(nb00 == ggml_type_size(type));
//     GGML_ASSERT(nb10 == ggml_type_size(src1->type));

//     // dst cannot be transposed or permuted
//     GGML_ASSERT(nb0 == sizeof(float));
//     GGML_ASSERT(nb0 <= nb1);
//     GGML_ASSERT(nb1 <= nb2);
//     GGML_ASSERT(nb2 <= nb3);

//     // nb01 >= nb00 - src0 is not transposed
//     //   compute by src0 rows

// #if GGML_USE_LLAMAFILE
//     // broadcast factors
//     const int64_t r2 = ne12 / ne02;
//     const int64_t r3 = ne13 / ne03;

//     const bool src1_cont = ggml_is_contiguous(src1);

//     if (src1_cont) {
//         for (int64_t i13 = 0; i13 < ne13; i13++)
//             for (int64_t i12 = 0; i12 < ne12; i12++)
//                 if (!llamafile_sgemm(ne01, ne11, ne00/ggml_blck_size(src0->type),
//                                      (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
//                                      nb01/ggml_type_size(src0->type),
//                                      (const char *)src1->data + i12*nb12 + i13*nb13,
//                                      nb11/ggml_type_size(src1->type),
//                                      (char *)dst->data + i12*nb2 + i13*nb3,
//                                      nb1/ggml_type_size(dst->type),
//                                      ith, nth,
//                                      src0->type,
//                                      src1->type,
//                                      dst->type))
//                     goto UseGgmlGemm1;
//         return;
//     }
// UseGgmlGemm1:;
// #endif

//     if (src1->type != vec_dot_type) {
//         char * wdata = params->wdata;

//         const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
//         const size_t nbw2 = nbw1*ne11;
//         const size_t nbw3 = nbw2*ne12;

//         assert(params->wsize >= ne13*nbw3);
//         GGML_ASSERT(src1->type == GGML_TYPE_F32);

//         for (int64_t i13 = 0; i13 < ne13; ++i13) {
//             for (int64_t i12 = 0; i12 < ne12; ++i12) {
//                 for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
//                     from_float_to_vec_dot((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
//                                           (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
//                                            ne10);
//                 }
//             }
//         }
//     }

//     if (ith == 0) {
//         // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
//         atomic_store(&params->shared->current_chunk, nth);
//     }

//     ggml_barrier(params->shared);

// #if GGML_USE_LLAMAFILE
//     if (src1->type != vec_dot_type) {
//         const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
//         const size_t row_size = ggml_row_size(vec_dot_type, ne10);

//         for (int64_t i13 = 0; i13 < ne13; i13++)
//             for (int64_t i12 = 0; i12 < ne12; i12++)
//                 if (!llamafile_sgemm(ne01, ne11, ne00/ggml_blck_size(src0->type),
//                                      (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
//                                      nb01/ggml_type_size(src0->type),
//                                      (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
//                                      row_size/ggml_type_size(vec_dot_type),
//                                      (char *)dst->data + i12*nb2 + i13*nb3,
//                                      nb1/ggml_type_size(dst->type),
//                                      ith, nth,
//                                      src0->type,
//                                      vec_dot_type,
//                                      dst->type))
//                     goto UseGgmlGemm2;
//         return;
//     }
// UseGgmlGemm2:;
// #endif

//     // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
//     const int64_t nr0 = ne0;

//     // This is the size of the rest of the dimensions of the result
//     const int64_t nr1 = ne1 * ne2 * ne3;

//     // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
//     int64_t num_rows_per_vec_dot = vec_dot_num_rows;
//     // TODO: currently the mmla kernels support only even numbered rows/cols.
//     // this check can be removed once they are extended to support odd numbered rows/cols too
//     if ((nr0 % 2 != 0) || (ne11 % 2 != 0)) {
//         num_rows_per_vec_dot = 1;
//     }

//     // Now select a reasonable chunk size.
//     int chunk_size = 16;

//     // We need to step up the size if it's small
//     if (nr0 == 1 || nr1 == 1) {
//         chunk_size = 64;
//     }

//     // distribute the work across the inner or outer loop based on which one is larger
//     // The number of chunks in the 0/1 dim.
//     // CEIL(nr0/chunk_size)
//     int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
//     int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

//     // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
//     //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggerganov/llama.cpp/pull/6915
//     //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
//     if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
//         // distribute the thread work across the inner or outer loop based on which one is larger
//         nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
//         nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
//     }

//     // The number of elements in each chunk
//     const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
//     const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

//     // The first chunk comes from our thread_id, the rest will get auto-assigned.
//     int current_chunk = ith;

//     while (current_chunk < nchunk0 * nchunk1) {
//         const int64_t ith0 = current_chunk % nchunk0;
//         const int64_t ith1 = current_chunk / nchunk0;

//         const int64_t ir0_start = dr0 * ith0;
//         const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

//         const int64_t ir1_start = dr1 * ith1;
//         const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

//         ggml_compute_forward_mul_mat_one_chunk(params, dst, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

//         if (nth >= nchunk0 * nchunk1) {
//             break;
//         }

//         current_chunk = atomic_fetch_add(&params->shared->current_chunk, 1);
//     }
// }

void eggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
  if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
    ebbrt::kprintf("tensor->op == GGML_OP_NONE\n");
    return;
  }

  switch (tensor->op) {
  case GGML_OP_MUL_MAT:
    ebbrt::kprintf("GGML_OP_MUL_MAT\n");
    //eggml_compute_forward_mul_mat(params, tensor);
    break;
  default:    
    ebbrt::kprintf("Unknown OP\n");
    break;
  }
}
extern "C" void ebbrt_ggml_graph_compute(void* ecgraph, void* ecplan) {
  struct ggml_compute_state_shared * state_shared = (struct ggml_compute_state_shared *)malloc(sizeof(struct ggml_compute_state_shared));
  state_shared->cgraph = static_cast<struct ggml_cgraph *>(ecgraph);
  state_shared->cplan = static_cast<struct ggml_cplan *>(ecplan);
  int n_threads = state_shared->cplan->n_threads;
  state_shared->n_threads = n_threads;
  state_shared->n_barrier = 0;
  state_shared->n_barrier_passed = 0;
  state_shared->abort_callback = NULL;
  state_shared->abort_callback_data = NULL;
  state_shared->current_chunk = 0;
  state_shared->ec = GGML_STATUS_SUCCESS;

  size_t mainCPU = ebbrt::Cpu::GetMine();  
  ebbrt::kprintf("[TEST] mainCPU=%d n_threads=%d\n", mainCPU, n_threads);
  ebbrt::EventManager::EventContext context;
  static ebbrt::SpinBarrier bar(n_threads);
  
  for (int i = 0; i < n_threads; i++) {
    ebbrt::event_manager->SpawnRemote([i, n_threads, mainCPU, &context, state_shared]() {
	int mycpu = static_cast<int>(ebbrt::Cpu::GetMine());
	ebbrt::kprintf("[TEST CPU:%d] SpawnRemote\n", mycpu);

	struct ggml_compute_state state;
	state.thrd = 0;
	state.ith = mycpu;
	state.shared = state_shared;
	
	struct ggml_cgraph * cgraph = state.shared->cgraph;
	struct ggml_cplan  * cplan  = state.shared->cplan;
	struct ggml_compute_params params;
	params.ith = state.ith;
	params.nth = state.shared->n_threads;
	params.wsize = cplan->work_size;
	params.wdata = cplan->work_data;
	params.shared = state.shared;

	for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
	  struct ggml_tensor * node = cgraph->nodes[node_n];

	  eggml_compute_forward(&params, node);
	  
	  if (state.ith == 0 && cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
	    ebbrt::kprintf("[TEST] GGML_STATUS_ABORTED %d\n", mycpu);
	    state.shared->ec = GGML_STATUS_ABORTED;
	  }
	  
	  bar.Wait();
	  
	  if (state.shared->ec != GGML_STATUS_SUCCESS) {
	    ebbrt::kprintf("[TEST] state.shared->ec != GGML_STATUS_SUCCESS %d\n", mycpu);
	    break;
	  }
	}
	
	bar.Wait();
	if (ebbrt::Cpu::GetMine() == mainCPU)
	  ebbrt::event_manager->ActivateContext(std::move(context));
    }, i);
  }
  ebbrt::event_manager->SaveContext(context);
  ebbrt::kprintf("[TEST] END\n");  
}

// Define the C++ function that takes a function pointer and data
extern "C" void cppFunction(FunctionPtr ptr, void* workers, int n_threads, size_t workerSize) {
  //ebbrt::kprintf("cppFunction START\n");
  size_t mainCPU = ebbrt::Cpu::GetMine();
  ebbrt::EventManager::EventContext context;
  std::atomic<size_t> count(0);
  static ebbrt::SpinBarrier bar(n_threads);
  
  char* workerPtr = static_cast<char*>(workers);
  
  for (int i = 0; i < n_threads; i++) {
    ebbrt::event_manager->SpawnRemote([&context, &count, i, n_threads, mainCPU, ptr, workerPtr, workerSize]() {
      //ebbrt::kprintf("cppFunction SpawnRemote %d, workerPtr %p, workerSize %u \n", static_cast<int>(ebbrt::Cpu::GetMine()), workerPtr, workerSize);

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
  //ebbrt::kprintf("cppFunction END\n");
}

void eeggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
  if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
    ebbrt::kprintf("tensor->op == GGML_OP_NONE\n");
    return;
  }

  switch (tensor->op) {
  case GGML_OP_MUL_MAT:
    ebbrt::kprintf("GGML_OP_MUL_MAT\n");
    break;
  default:    
    ebbrt::kprintf("Unknown OP\n");
    break;
  }
}
