// LD_PRELOAD shim: stop Level Zero peer residency from mirroring device memory into host RAM.
//
// Unified Runtime makes every device USM allocation resident on all P2P peers
// (zeContextMakeMemoryResident per peer). On compute-runtime 26.18-26.35 with B70s that pins a
// host-side copy of the whole allocation, so a TP=2 KV cache costs ~30 GB of host RAM
// (intel/compute-runtime#968). Peer access works without it. This shim registers a Level Zero
// tracing-layer prologue that redirects such a call to the allocation's own device, where making
// device memory resident is a no-op. Requires ZE_ENABLE_TRACING_LAYER=1.
//
// Build: gcc -O2 -shared -fPIC -o libl0_owner_residency.so l0_owner_residency.c -lze_loader
#include <level_zero/ze_api.h>
#include <level_zero/layers/zel_tracing_api.h>
#include <level_zero/layers/zel_tracing_register_cb.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

static atomic_ulong redirected, redirected_bytes, passed;

static void prologue(ze_context_make_memory_resident_params_t *p, ze_result_t r, void *ud, void **inst)
{
    (void)r; (void)ud; (void)inst;
    ze_memory_allocation_properties_t props = {ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES, NULL};
    ze_device_handle_t owner = NULL;
    if (zeMemGetAllocProperties(*p->phContext, *p->pptr, &props, &owner) == ZE_RESULT_SUCCESS &&
        props.type == ZE_MEMORY_TYPE_DEVICE && owner != NULL && owner != *p->phDevice) {
        *p->phDevice = owner;
        unsigned long n = atomic_fetch_add(&redirected, 1) + 1;
        unsigned long b = atomic_fetch_add(&redirected_bytes, *p->psize) + *p->psize;
        if ((n & (n - 1)) == 0)
            fprintf(stderr, "[l0-owner-residency] redirected %lu peer residency calls (%.2f GiB)\n",
                    n, b / 1073741824.0);
    } else {
        atomic_fetch_add(&passed, 1);
    }
}

__attribute__((constructor)) static void init(void)
{
    if (!getenv("ZE_ENABLE_TRACING_LAYER")) {
        fprintf(stderr, "[l0-owner-residency] ZE_ENABLE_TRACING_LAYER not set, shim inactive\n");
        return;
    }
    if (zeInit(0) != ZE_RESULT_SUCCESS) return;
    zel_tracer_desc_t desc = {ZEL_STRUCTURE_TYPE_TRACER_EXP_DESC, NULL, NULL};
    zel_tracer_handle_t t;
    if (zelTracerCreate(&desc, &t) != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "[l0-owner-residency] zelTracerCreate failed, shim inactive\n");
        return;
    }
    zelTracerContextMakeMemoryResidentRegisterCallback(t, ZEL_REGISTER_PROLOGUE, prologue);
    zelTracerSetEnabled(t, 1);
}

__attribute__((destructor)) static void fini(void)
{
    if (atomic_load(&redirected))
        fprintf(stderr, "[l0-owner-residency] total: %lu redirected (%.2f GiB), %lu passed through\n",
                atomic_load(&redirected), atomic_load(&redirected_bytes) / 1073741824.0, atomic_load(&passed));
}
