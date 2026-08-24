#pragma once

#if defined(__APPLE__) && defined(_OPENMP)
extern "C" __attribute__((visibility("hidden")))
void __kmpc_dispatch_deinit(void*, int) {}
#endif
