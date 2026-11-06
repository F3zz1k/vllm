// l0copy: KV offload copies on an immediate command list of our own, in the engine's own Level Zero context.
//
// On a B70 the UR adapter sends every queue's copies in a process through one ring on the copy engine, behind the
// forward's per-step output copies, which wait on compute: a waited offload copy moved once per forward step
// (bench/copy-hol: 2.9 GB/s in 64 MB pieces). A list created here gets a ring of its own and copies at the
// engine's rate while the forward runs (bench/copy-ipc copyhol2: 11.2 GB/s, compute steps unchanged).
//
// The context is the engine's (device memory is only usable in its own context): a tracer, active when the library
// is preloaded with ZE_ENABLE_TRACING_LAYER=1, records the context of device allocations. l0copy_open(ptr) takes
// the device that holds ptr (a KV tensor) and creates the list on its copy-only queue group.
//
// Build: gcc -O2 -shared -fPIC -o libl0copy.so l0copy.c -lze_loader (the image: /opt/b70/libl0copy.so)
#include <level_zero/ze_api.h>
#include <level_zero/layers/zel_tracing_api.h>
#include <level_zero/layers/zel_tracing_register_cb.h>
#include <stdint.h>
#include <stdlib.h>

static ze_context_handle_t ctx;  // context of the last device allocation
static ze_device_handle_t dev;
static ze_command_list_handle_t list;
static ze_event_pool_handle_t pool;
static ze_event_handle_t *events;
static int num_events;

static void alloc_epilogue(ze_mem_alloc_device_params_t *p, ze_result_t r, void *ud, void **inst)
{
    (void)ud; (void)inst;
    if (r == ZE_RESULT_SUCCESS)
        ctx = *p->phContext;
}

__attribute__((constructor)) static void init(void)
{
    if (!getenv("ZE_ENABLE_TRACING_LAYER") || zeInit(0) != ZE_RESULT_SUCCESS)
        return;
    zel_tracer_desc_t desc = {ZEL_STRUCTURE_TYPE_TRACER_EXP_DESC, NULL, NULL};
    zel_tracer_handle_t t;
    if (zelTracerCreate(&desc, &t) != ZE_RESULT_SUCCESS) return;
    zelTracerMemAllocDeviceRegisterCallback(t, ZEL_REGISTER_EPILOGUE, alloc_epilogue);
    zelTracerSetEnabled(t, 1);
}

// The list on the copy engine of the device holding ptr, and n events. Returns 0 or an error code.
int l0copy_open(void *ptr, int n)
{
    if (!ctx) return 0x100;
    ze_memory_allocation_properties_t ap = {ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES, NULL};
    ze_result_t r = zeMemGetAllocProperties(ctx, ptr, &ap, &dev);
    if (r) return 0x200 | r;
    if (ap.type != ZE_MEMORY_TYPE_DEVICE || !dev) return 0x300;
    uint32_t groups = 16;
    ze_command_queue_group_properties_t props[16];
    for (uint32_t i = 0; i < 16; i++)
        props[i] = (ze_command_queue_group_properties_t){ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, NULL};
    if ((r = zeDeviceGetCommandQueueGroupProperties(dev, &groups, props))) return 0x400 | r;
    int ordinal = -1;
    for (uint32_t i = 0; i < groups && ordinal < 0; i++)
        if ((props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) &&
            !(props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE))
            ordinal = (int)i;
    if (ordinal < 0) return 0x500;
    ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, NULL, (uint32_t)ordinal, 0, 0,
                                  ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    if ((r = zeCommandListCreateImmediate(ctx, dev, &qd, &list))) return 0x600 | r;
    ze_event_pool_desc_t pd = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, NULL, ZE_EVENT_POOL_FLAG_HOST_VISIBLE, (uint32_t)n};
    if ((r = zeEventPoolCreate(ctx, &pd, 1, &dev, &pool))) return 0x700 | r;
    if (!(events = calloc((size_t)n, sizeof(*events)))) return 0x800;
    for (int i = 0; i < n; i++) {
        ze_event_desc_t ed = {ZE_STRUCTURE_TYPE_EVENT_DESC, NULL, (uint32_t)i, ZE_EVENT_SCOPE_FLAG_HOST,
                              ZE_EVENT_SCOPE_FLAG_HOST};
        if ((r = zeEventCreate(pool, &ed, &events[i]))) return 0x900 | r;
    }
    num_events = n;
    return 0;
}

// Pinned host memory in the engine's context, or NULL.
void *l0copy_host_alloc(uint64_t size)
{
    ze_host_mem_alloc_desc_t hd = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, NULL, 0};
    void *p = NULL;
    return ctx && !zeMemAllocHost(ctx, &hd, size, 4096, &p) ? p : NULL;
}

// Appends n copies that signal event ev once all are done (callers serialise calls).
int l0copy_copy_async(void **dst, void **src, uint64_t *size, int n, int ev)
{
    if (ev < 0 || ev >= num_events) return 0xa00;
    ze_result_t r = zeEventHostReset(events[ev]);
    if (r) return 0xb00 | r;
    for (int i = 0; i < n; i++)
        if ((r = zeCommandListAppendMemoryCopy(list, dst[i], src[i], size[i], i == n - 1 ? events[ev] : NULL, 0,
                                               NULL)))
            return 0xc00 | r;
    return 0;
}

int l0copy_wait(int ev)
{
    if (ev < 0 || ev >= num_events) return 0xa00;
    ze_result_t r = zeEventHostSynchronize(events[ev], UINT64_MAX);
    return r ? 0xd00 | r : 0;
}
