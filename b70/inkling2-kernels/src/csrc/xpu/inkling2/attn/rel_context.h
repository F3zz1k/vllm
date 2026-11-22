#pragma once
// Inkling2 relative-position bias for the next chunk-prefill / paged-decode
// launch on this thread (set by the _inkling2_C ops, read where the vendor
// dispatchers build their kernel arguments; rel == nullptr: no bias).
namespace inkling2::xe2 {
struct RelContext {
  const void* rel = nullptr;  // bf16 [T, Hq, E]: bias at backward distance d
  int stride_t = 0;
  int stride_h = 0;
  int extent = 0;
};
inline thread_local RelContext g_rel;
}  // namespace inkling2::xe2
