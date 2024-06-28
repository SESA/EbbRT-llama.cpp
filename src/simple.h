#ifndef _SIMPLE_H_
#define _SIMPLE_H_

#include <ebbrt/SharedIOBufRef.h>
#include <ebbrt/StaticIOBuf.h>
#include <ebbrt/UniqueIOBuf.h>
#include <ebbrt/StaticSharedEbb.h>
#include <ebbrt/AtomicUniquePtr.h>

std::unique_ptr<ebbrt::MutIOBuf> GLLMbuf{nullptr};

#endif //_SIMPLE_H_
