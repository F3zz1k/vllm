// l0ipc: let a copy helper process write an engine's device memory (Level Zero IPC), so the helper's copies run in
// its own context on the copy engine instead of queueing behind the engine's per-step copies (bench/copy-hol:
// in one process each waited copy moves once per forward step; a helper copies at 13.8 GB/s).
//
// Engine side (LD_PRELOAD, ZE_ENABLE_TRACING_LAYER=1): a tracer records the context of device allocations;
// l0ipc_export(ptr) returns a file descriptor for the allocation holding ptr and ptr's offset in it.
// Helper side: l0ipc_open(index) makes its own context on visible device ``index`` (the worker's) and an immediate
// command list on the copy engine; l0ipc_import(fd, offset) opens the allocation; l0ipc_copy() copies and waits.
//
// Build: gcc -O2 -shared -fPIC -o libl0ipc.so l0ipc.c -lze_loader (the image: /opt/b70/libl0ipc.so)
#include <level_zero/ze_api.h>
#include <level_zero/layers/zel_tracing_api.h>
#include <level_zero/layers/zel_tracing_register_cb.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static ze_context_handle_t engine_ctx;  // engine side: context of the last device allocation

static void alloc_epilogue(ze_mem_alloc_device_params_t *p, ze_result_t r, void *ud, void **inst)
{
    (void)ud; (void)inst;
    if (r == ZE_RESULT_SUCCESS)
        engine_ctx = *p->phContext;
}

__attribute__((constructor)) static void init(void)
{
    if (!getenv("L0IPC_ENGINE") || !getenv("ZE_ENABLE_TRACING_LAYER"))
        return;
    if (zeInit(0) != ZE_RESULT_SUCCESS) return;
    zel_tracer_desc_t desc = {ZEL_STRUCTURE_TYPE_TRACER_EXP_DESC, NULL, NULL};
    zel_tracer_handle_t t;
    if (zelTracerCreate(&desc, &t) != ZE_RESULT_SUCCESS) return;
    zelTracerMemAllocDeviceRegisterCallback(t, ZEL_REGISTER_EPILOGUE, alloc_epilogue);
    zelTracerSetEnabled(t, 1);
}

// Engine side. Returns 0 and fills fd, offset (of ptr in its allocation), size (of the allocation) and the raw
// 64-byte IPC handle (raw).
int l0ipc_export(void *ptr, int *fd, uint64_t *offset, uint64_t *size, unsigned char *raw)
{
    if (!engine_ctx) return -1;
    void *base = NULL;
    size_t sz = 0;
    ze_result_t r = zeMemGetAddressRange(engine_ctx, ptr, &base, &sz);
    if (r) return 0x1000 | r;
    ze_ipc_mem_handle_t h;
    if ((r = zeMemGetIpcHandle(engine_ctx, base, &h))) return 0x2000 | r;
    for (int i = 0; i < ZE_MAX_IPC_HANDLE_SIZE; i++) raw[i] = (unsigned char)h.data[i];
    uint64_t f = 0;
    *fd = zeMemGetFileDescriptorFromIpcHandleExp(engine_ctx, h, &f) ? -1 : (int)f;
    *offset = (uint64_t)((char *)ptr - (char *)base);
    *size = sz;
    return 0;
}

static ze_context_handle_t ctx;
static ze_device_handle_t dev;
static ze_command_list_handle_t list;

// Helper side: context, visible device ``index`` and an immediate command list on the first copy-only queue group.
int l0ipc_open(int index)
{
    ze_result_t r = zeInit(0);
    if (r) return 0x100 | r;
    uint32_t n = 1;
    ze_driver_handle_t drv;
    if ((r = zeDriverGet(&n, &drv))) return 0x200 | r;
    ze_device_handle_t devs[16];
    n = 16;
    if ((r = zeDeviceGet(drv, &n, devs))) return 0x300 | r;
    if (index < 0 || (uint32_t)index >= n) return 0x380;
    dev = devs[index];
    ze_context_desc_t cd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, NULL, 0};
    if ((r = zeContextCreate(drv, &cd, &ctx))) return 0x400 | r;
    uint32_t groups = 0;
    zeDeviceGetCommandQueueGroupProperties(dev, &groups, NULL);
    ze_command_queue_group_properties_t props[16];
    for (uint32_t i = 0; i < groups && i < 16; i++)
        props[i] = (ze_command_queue_group_properties_t){ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, NULL};
    if (groups > 16) groups = 16;
    zeDeviceGetCommandQueueGroupProperties(dev, &groups, props);
    int ordinal = -1;
    for (uint32_t i = 0; i < groups; i++) {
        if (ordinal < 0 && (props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) &&
            !(props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE))
            ordinal = (int)i;
    }
    if (ordinal < 0) return 0x500;
    ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, NULL, (uint32_t)ordinal, 0, 0,
                                  ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    if ((r = zeCommandListCreateImmediate(ctx, dev, &qd, &list))) return 0x600 | r;
    return 0;
}

// Opens the allocation: mode 2 (use this) puts the received fd into the raw handle's first int, the scheme oneCCL
// uses on these drivers; mode 1 opens the raw handle as is; mode 0 converts fd with the IPC extension (rejected
// with INVALID_ARGUMENT on compute-runtime 26.35).
// Returns 0 and the device pointer in *out, or an error code.
int l0ipc_import(int mode, int fd, const unsigned char *raw, uint64_t offset, void **out)
{
    ze_ipc_mem_handle_t h;
    ze_result_t r;
    for (int i = 0; i < ZE_MAX_IPC_HANDLE_SIZE; i++) h.data[i] = (char)raw[i];
    if (mode == 0 && (r = zeMemGetIpcHandleFromFileDescriptorExp(ctx, (uint64_t)fd, &h))) return 0x900 | r;
    if (mode == 2) *(int *)h.data = fd;
    void *base = NULL;
    if ((r = zeMemOpenIpcHandle(ctx, dev, h, 0, &base))) return 0xa00 | r;
    *out = (char *)base + offset;
    return 0;
}

void *l0ipc_host_alloc(uint64_t size)
{
    ze_host_mem_alloc_desc_t hd = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, NULL, 0};
    void *p = NULL;
    return zeMemAllocHost(ctx, &hd, size, 4096, &p) ? NULL : p;
}

// Enqueue copies (dst, src, size) x n back to back, then wait for all of them.
int l0ipc_copy(void **dst, void **src, uint64_t *size, int n)
{
    ze_result_t r;
    for (int i = 0; i < n; i++)
        if ((r = zeCommandListAppendMemoryCopy(list, dst[i], src[i], size[i], NULL, 0, NULL))) return 0x700 | r;
    if ((r = zeCommandListHostSynchronize(list, UINT64_MAX))) return 0x800 | r;
    return 0;
}

// Events for rings of in-flight copies: l0ipc_events(n) makes n host-visible events (once); l0ipc_copy_async
// appends n copies that signal event ev when all are done; l0ipc_event_wait waits for ev and resets it.
static ze_event_pool_handle_t pool;
static ze_event_handle_t *events;
static int num_events;

int l0ipc_events(int n)
{
    ze_event_pool_desc_t pd = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, NULL, ZE_EVENT_POOL_FLAG_HOST_VISIBLE, (uint32_t)n};
    ze_result_t r = zeEventPoolCreate(ctx, &pd, 1, &dev, &pool);
    if (r) return 0xb00 | r;
    events = calloc((size_t)n, sizeof(*events));
    for (int i = 0; i < n; i++) {
        ze_event_desc_t ed = {ZE_STRUCTURE_TYPE_EVENT_DESC, NULL, (uint32_t)i, ZE_EVENT_SCOPE_FLAG_HOST,
                              ZE_EVENT_SCOPE_FLAG_HOST};
        if ((r = zeEventCreate(pool, &ed, &events[i]))) return 0xc00 | r;
    }
    num_events = n;
    return 0;
}

int l0ipc_copy_async(void **dst, void **src, uint64_t *size, int n, int ev)
{
    if (ev < 0 || ev >= num_events) return 0xd00;
    ze_result_t r;
    for (int i = 0; i < n; i++)
        if ((r = zeCommandListAppendMemoryCopy(list, dst[i], src[i], size[i], i == n - 1 ? events[ev] : NULL, 0,
                                               NULL)))
            return 0xe00 | r;
    return 0;
}

int l0ipc_event_wait(int ev)
{
    if (ev < 0 || ev >= num_events) return 0xd00;
    ze_result_t r = zeEventHostSynchronize(events[ev], UINT64_MAX);
    if (r) return 0xf00 | r;
    return zeEventHostReset(events[ev]) ? 0xf80 : 0;
}
