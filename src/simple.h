#ifndef _SIMPLE_H_
#define _SIMPLE_H_

#include <ebbrt/SharedIOBufRef.h>
#include <ebbrt/StaticIOBuf.h>
#include <ebbrt/UniqueIOBuf.h>
#include <ebbrt/StaticSharedEbb.h>
#include <ebbrt/AtomicUniquePtr.h>

extern std::unique_ptr<ebbrt::MutIOBuf> GLLMbuf;

#endif //_SIMPLE_H_
