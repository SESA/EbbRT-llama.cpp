#include "simple.h"
#include "common.h"
//#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <cstdlib>
#include <sstream>

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

#define MCPU 1
#define PORT 8888

std::unique_ptr<ebbrt::MutIOBuf> GLLMbuf = nullptr;
std::unique_ptr<ebbrt::MutIOBuf> tGLLMbuf = nullptr;

namespace ebbrt {    
class TcpCommand : public StaticSharedEbb<TcpCommand>, public CacheAligned {
public:
  TcpCommand() {}
  void Start(uint16_t port) {
    listening_pcb_.Bind(port, [this](NetworkManager::TcpPcb pcb) {
      // new connection callback
      static std::atomic<size_t> cpu_index{MCPU};
      pcb.BindCpu(MCPU);
      auto connection = new TcpSession(std::move(pcb));
      connection->Install();
      ebbrt::kprintf_force("Core %u: TcpCommand connection created\n", static_cast<uint32_t>(ebbrt::Cpu::GetMine()));
    });
  }
  
private:
  class TcpSession : public ebbrt::TcpHandler {
  public:
    TcpSession(ebbrt::NetworkManager::TcpPcb pcb)
      : ebbrt::TcpHandler(std::move(pcb)) {}
    void Close() {}
    void Abort() {}
    
    void Receive(std::unique_ptr<MutIOBuf> b) {
      kassert(b->Length() != 0);

      if(GLLMbuf) {
	GLLMbuf->PrependChain(std::move(b));	  
      } else {
	GLLMbuf = std::move(b);
      }
      //ebbrt::kprintf_force("GLLMbuf len=%u\n", GLLMbuf->ComputeChainDataLength());
    }
          
  private:
    ebbrt::NetworkManager::TcpPcb pcb_;
  };

  NetworkManager::ListeningTcpPcb listening_pcb_;
};

  class RunCommand : public StaticSharedEbb<RunCommand>, public CacheAligned {
public:
  RunCommand() {}
  void Start(uint16_t port) {
    listening_pcb_.Bind(port, [this](NetworkManager::TcpPcb pcb) {
      // new connection callback
      static std::atomic<size_t> cpu_index{MCPU};
      pcb.BindCpu(MCPU);
      auto connection = new RunSession(std::move(pcb));
      connection->Install();
      ebbrt::kprintf_force("Core %u: RunCommand connection created\n", static_cast<uint32_t>(ebbrt::Cpu::GetMine()));
    });
  }
  
private:
  class RunSession : public ebbrt::TcpHandler {
  public:
    RunSession(ebbrt::NetworkManager::TcpPcb pcb) : ebbrt::TcpHandler(std::move(pcb)) {}
    void Close() {}
    void Abort() {}
    
    void Receive(std::unique_ptr<MutIOBuf> b) {
      kassert(b->Length() != 0);

      std::string s(reinterpret_cast<const char*>(b->Data()));
      //ebbrt::kprintf_force("*** %s ****\n", s.c_str());
      ebbrt::kprintf_force("GLLMbuf len=%u chain_elements:%u\n", GLLMbuf->ComputeChainDataLength(), GLLMbuf->CountChainElements());
      tGLLMbuf = MakeUniqueIOBuf(GLLMbuf->ComputeChainDataLength());
      
      auto GLLMfile = tGLLMbuf->MutData();
      for (auto& buf_it : *GLLMbuf) {
	memcpy(GLLMfile, buf_it.Data(), buf_it.Length());
        GLLMfile += buf_it.Length();
      }
      ebbrt::kprintf_force("tGLLMbuf len=%u chain_elements:%u addr:%p\n", tGLLMbuf->ComputeChainDataLength(), tGLLMbuf->CountChainElements(), tGLLMbuf->MutData());
      
      gpt_params params;    
      params.prompt = "Hello my name is";
      params.n_predict = 32;

      // total length of the sequence including the prompt
      const int n_predict = params.n_predict;
      
      // init LLM      
      llama_backend_init();
      llama_model_params model_params = llama_model_params_from_gpt_params(params);      
      llama_model * model = llama_load_model_from_file("", model_params);
      if (model == NULL) {
	ebbrt::kprintf("%s: error: unable to load model\n" , __func__);
        return;
      }
      
      // initialize the context
    
      llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
      
      llama_context * ctx = llama_new_context_with_model(model, ctx_params);
      
      if (ctx == NULL) {
        ebbrt::kprintf("%s: error: failed to create the llama_context\n" , __func__);
        return;
      }

      /*
      // tokenize the prompt

      std::vector<llama_token> tokens_list;
      tokens_list = ::llama_tokenize(ctx, params.prompt, true);

      const int n_ctx    = llama_n_ctx(ctx);
      const int n_kv_req = tokens_list.size() + (n_predict - tokens_list.size());
      
      ebbrt::kprintf("\n%s: n_predict = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_predict, n_ctx, n_kv_req);

      // make sure the KV cache is big enough to hold all the prompt and generated tokens
      if (n_kv_req > n_ctx) {
        ebbrt::kprintf("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
        ebbrt::kprintf("%s:        either reduce n_predict or increase n_ctx\n", __func__);
        return;
      }

      // print the prompt token-by-token
      ebbrt::kprintf("\n");

      for (auto id : tokens_list) {
        ebbrt::kprintf("%s", llama_token_to_piece(ctx, id).c_str());
      }
      

      // create a llama_batch with size 512
      // we use this object to submit token data for decoding

      llama_batch batch = llama_batch_init(512, 0, 1);

      // evaluate the initial prompt
      for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
      }

      // llama_decode will output logits only for the last token of the prompt
      batch.logits[batch.n_tokens - 1] = true;

      if (llama_decode(ctx, batch) != 0) {
        ebbrt::kprintf("%s: llama_decode() failed\n", __func__);
        return;
      }
      */
    }
    
  private:
    ebbrt::NetworkManager::TcpPcb pcb_;
  };

  NetworkManager::ListeningTcpPcb listening_pcb_;
  };
}

void AppMain() {
  ebbrt::kprintf_force("AppMain()\n");

  ebbrt::event_manager->SpawnRemote(
    [] () mutable {
      auto id = ebbrt::ebb_allocator->AllocateLocal();
      auto mc = ebbrt::EbbRef<ebbrt::TcpCommand>(id);
      mc->Start(PORT);
      ebbrt::kprintf_force("TcpCommand server listening on port %d\n", PORT);

      auto rid = ebbrt::ebb_allocator->AllocateLocal();
      auto rmc = ebbrt::EbbRef<ebbrt::RunCommand>(rid);
      rmc->Start(PORT+1);
      ebbrt::kprintf_force("RunCommand server listening on port %d\n", PORT+1);
      
    }, MCPU);
}


/*
int main(int argc, char ** argv) {
    gpt_params params;

    params.prompt = "Hello my name is";
    params.n_predict = 32;

    if (!gpt_params_parse(argc, argv, params)) {
        print_usage(argc, argv, params);
        return 1;
    }

    // total length of the sequence including the prompt
    const int n_predict = params.n_predict;

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model

    llama_model_params model_params = llama_model_params_from_gpt_params(params);

    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // initialize the context
    
    llama_context_params ctx_params = llama_context_params_from_gpt_params(params);

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // tokenize the prompt

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx, params.prompt, true);

    const int n_ctx    = llama_n_ctx(ctx);
    const int n_kv_req = tokens_list.size() + (n_predict - tokens_list.size());

    LOG_TEE("\n%s: n_predict = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_predict, n_ctx, n_kv_req);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        LOG_TEE("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
        LOG_TEE("%s:        either reduce n_predict or increase n_ctx\n", __func__);
        return 1;
    }

    // print the prompt token-by-token

    fprintf(stderr, "\n");

    for (auto id : tokens_list) {
        fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
    }

    fflush(stderr);

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding

    llama_batch batch = llama_batch_init(512, 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    // main loop

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();

    while (n_cur <= n_predict) {
        // sample the next token
        {
            auto   n_vocab = llama_n_vocab(model);
            auto * logits  = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
            }

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            // sample the most likely token
            const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

            // is it an end of generation?
            if (llama_token_is_eog(model, new_token_id) || n_cur == n_predict) {
                LOG_TEE("\n");

                break;
            }

            LOG_TEE("%s", llama_token_to_piece(ctx, new_token_id).c_str());
            fflush(stdout);

            // prepare the next batch
            llama_batch_clear(batch);

            // push this new token for next evaluation
            llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);

            n_decode += 1;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    LOG_TEE("\n");

    const auto t_main_end = ggml_time_us();

    LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    llama_print_timings(ctx);

    fprintf(stderr, "\n");

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
*/
