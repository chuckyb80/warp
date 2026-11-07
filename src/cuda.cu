/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * cuda.cu — the CUDA backend: mvq_rows_f32 on an NVIDIA GPU.
 *
 * Built only with WASTE_ENABLE_CUDA=1 (see the Makefile). Like the Metal
 * backend it overwrites one slot, mvq_rows_f32 — quantized trunk weights
 * (Q8G / Q4G / Q3G, one fp16 scale per group) times f32 activations — and
 * leaves every other kernel on the CPU. waste_register_cuda() declines
 * (returns NULL) when no device is usable, so a CUDA build still runs on a
 * machine without one.
 *
 * The one real difference from Metal is memory. Apple GPUs read host
 * memory in place, so metal.m wraps the model's buffers with no copy. A
 * discrete GPU has its own memory, so the weights cross PCIe. By default
 * every call copies what it reads (always correct). WASTE_CUDA_CACHE_MB=N
 * keeps up to N MB of weights resident on the device, keyed by host
 * address and size AND a sampled fingerprint of the bytes, because the
 * expert cache reuses host buffers for different experts: an address alone
 * would silently hand back last token's expert.
 *
 * Numbers: every output row is the same sum the CPU computes, in a
 * different order (32 lanes, then a tree reduction), so results match to
 * float rounding, not bit for bit — the same contract metal.m has.
 * tests/test_cuda.c checks it against waste_mvq_rows_f32.
 *
 * Env:
 *   WASTE_CUDA=0             decline, use the CPU (same as WASTE_BACKEND=cpu
 *                            for this slot)
 *   WASTE_CUDA_DEVICE=N      pick a device (default 0)
 *   WASTE_CUDA_MIN_KB=N      matvecs smaller than this stay on the CPU pool
 *                            (default 2048: below it the copy and launch
 *                            cost more than the work)
 *   WASTE_CUDA_CACHE_MB=N    device-resident weight cache, off by default
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "waste_backend.h"
#include "simd.h"

/* Counters, read by tests/test_cuda.c. */
extern "C" {
unsigned long long waste_cuda_n_calls = 0, waste_cuda_n_fallback = 0;
unsigned long long waste_cuda_bytes_h2d = 0, waste_cuda_cache_hits = 0;
void waste_cuda_release_host_buffers(void);
const char *waste_register_cuda(waste_kernels *t);
}

/* ---- the kernel ----------------------------------------------------------
 * One warp per output row. Lane l walks the row's inputs l, l+32, ...;
 * each input is dequantized on the fly and scaled by its group's fp16
 * scale, exactly the terms of the CPU sum. */

__device__ __forceinline__ int cu_q3_at(const uint8_t *p, long i)
{
    const long off = i * 3;
    const int v = (p[off >> 3] >> (off & 7)) | (p[(off >> 3) + 1] << (8 - (off & 7)));
    return (v & 7) - 4;
}

__global__ void mvq_kernel(float *y, const uint8_t *W, const uint16_t *ws,
                           const float *x, int in, int ng, int g, int bits,
                           unsigned long long rowbytes, int rows)
{
    const int warps = blockDim.x >> 5;
    const int row = blockIdx.x * warps + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint8_t *r = W + (size_t)row * rowbytes;
    const __half *sc = (const __half *)(ws + (size_t)row * ng);
    float acc = 0.f;
    if (bits == 8) {
        const int8_t *r8 = (const int8_t *)r;
        for (int i = lane; i < in; i += 32)
            acc += __half2float(sc[i / g]) * ((float)r8[i] * x[i]);
    } else if (bits == 4) {
        /* two signed nibbles per byte, low first; g is even, so byte i/2 */
        for (int i = lane; i < in; i += 32) {
            const uint8_t b = r[i >> 1];
            const int w = (i & 1) ? ((int)(b >> 4) - 8) : ((int)(b & 0x0F) - 8);
            acc += __half2float(sc[i / g]) * ((float)w * x[i]);
        }
    } else {
        for (int i = lane; i < in; i += 32)
            acc += __half2float(sc[i / g]) * ((float)cu_q3_at(r, i) * x[i]);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

/* ---- device memory --------------------------------------------------------- */

struct cu_buf { void *p; size_t cap; };

struct cu_entry {            /* one resident weight buffer */
    const void *host; size_t bytes; uint64_t fp;
    void *dev; unsigned long long used;
};

static struct {
    int ready;               /* 0 untried, 1 usable, -1 declined */
    int dev;
    cudaStream_t s;
    cu_buf w, ws, x, y;      /* per-call scratch, grow-only */
    cu_entry *cache; int ncache, capcache;
    size_t cache_bytes, cache_limit;
    unsigned long long tick;
    char name[300];          /* "CUDA (" + cudaDeviceProp.name (256 chars) + ")" */
} g;

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

static void *cu_scratch(cu_buf *b, size_t need)
{
    if (b->p && b->cap >= need) return b->p;
    if (b->p) cudaFree(b->p);
    b->p = NULL; b->cap = 0;
    const size_t pad = (need + 4095) & ~(size_t)4095;   /* q3 reads one byte past a row */
    if (cudaMalloc(&b->p, pad + 64) != cudaSuccess) { b->p = NULL; return NULL; }
    b->cap = pad;
    return b->p;
}

/* FNV-1a over 64 evenly spaced 8-byte samples plus the first and last
 * 64 bytes. Cheap (a few hundred bytes read), and enough to tell one
 * expert from another in the same host buffer. */
static uint64_t cu_fingerprint(const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 1469598103934665603ull ^ (uint64_t)n;
    #define MIX(v) do { h ^= (uint64_t)(v); h *= 1099511628211ull; } while (0)
    const size_t edge = n < 64 ? n : 64;
    for (size_t i = 0; i < edge; i++) MIX(b[i]);
    for (size_t i = 0; i < edge; i++) MIX(b[n - edge + i]);
    if (n >= 8)
        for (int k = 0; k < 64; k++) {
            uint64_t v; memcpy(&v, b + ((n - 8) / 63) * (size_t)k, 8); MIX(v);
        }
    #undef MIX
    return h;
}

/* Device pointer holding [host, host+bytes): from the cache when enabled
 * and the fingerprint still matches, else copied into `scratch`. */
static const void *cu_upload(const void *host, size_t bytes, cu_buf *scratch)
{
    if (g.cache_limit && bytes <= g.cache_limit / 2) {
        const uint64_t fp = cu_fingerprint(host, bytes);
        for (int i = 0; i < g.ncache; i++) {
            cu_entry *e = &g.cache[i];
            if (e->host == host && e->bytes == bytes) {
                if (e->fp == fp) { e->used = ++g.tick; waste_cuda_cache_hits++; return e->dev; }
                /* same buffer, new contents: refresh in place */
                if (cudaMemcpyAsync(e->dev, host, bytes, cudaMemcpyHostToDevice, g.s) != cudaSuccess) return NULL;
                waste_cuda_bytes_h2d += bytes; e->fp = fp; e->used = ++g.tick; return e->dev;
            }
        }
        while (g.cache_bytes + bytes > g.cache_limit && g.ncache) {   /* evict least recently used */
            int lru = 0;
            for (int i = 1; i < g.ncache; i++) if (g.cache[i].used < g.cache[lru].used) lru = i;
            cudaFree(g.cache[lru].dev); g.cache_bytes -= g.cache[lru].bytes;
            g.cache[lru] = g.cache[--g.ncache];
        }
        void *d = NULL;
        if (g.ncache == g.capcache) {
            const int nc = g.capcache ? 2 * g.capcache : 64;
            cu_entry *ne = (cu_entry *)realloc(g.cache, (size_t)nc * sizeof *ne);
            if (ne) { g.cache = ne; g.capcache = nc; }
        }
        if (g.ncache < g.capcache && cudaMalloc(&d, bytes + 64) == cudaSuccess) {
            if (cudaMemcpyAsync(d, host, bytes, cudaMemcpyHostToDevice, g.s) != cudaSuccess) { cudaFree(d); return NULL; }
            waste_cuda_bytes_h2d += bytes;
            g.cache[g.ncache++] = (cu_entry){ host, bytes, fp, d, ++g.tick };
            g.cache_bytes += bytes;
            return d;
        }
        /* no room: fall through to the scratch copy */
    }
    void *d = cu_scratch(scratch, bytes);
    if (!d) return NULL;
    if (cudaMemcpyAsync(d, host, bytes, cudaMemcpyHostToDevice, g.s) != cudaSuccess) return NULL;
    waste_cuda_bytes_h2d += bytes;
    return d;
}

/* ---- the slot ---------------------------------------------------------------- */

/* The CPU kernel, for shapes the device does not take. One call over the
 * whole range: this path is rare (no container this project converts
 * produces such a shape), so it does not go through the pool. */
static void cu_cpu(int b, int e, void *p)
{
    waste_cuda_n_fallback++;
    waste_k.mvq_rows_cpu(b, e, p);
}

static void mvq_rows_f32_cuda(int b, int e, void *p)
{
    mvq_arg *a = (mvq_arg *)p;
    /* run_rows hands the device the whole range; anything else (or an odd
     * group, or a bit width the kernel does not know) is the CPU's. */
    if (b != 0 || e <= 0 || a->group <= 0 || (a->group & 1) ||
        (a->bits != 8 && a->bits != 4 && a->bits != 3)) { cu_cpu(b, e, p); return; }
    pthread_mutex_lock(&g_mu);
    const size_t wbytes = (size_t)e * a->rowbytes;
    const size_t sbytes = (size_t)e * a->ng * sizeof(uint16_t);
    const void *dW = cu_upload(a->W, wbytes, &g.w);
    const void *dS = dW ? cu_upload(a->ws, sbytes, &g.ws) : NULL;
    float *dX = (float *)cu_scratch(&g.x, (size_t)a->in * sizeof(float));
    float *dY = (float *)cu_scratch(&g.y, (size_t)e * sizeof(float));
    if (!dW || !dS || !dX || !dY ||
        cudaMemcpyAsync(dX, a->xs, (size_t)a->in * sizeof(float), cudaMemcpyHostToDevice, g.s) != cudaSuccess) {
        pthread_mutex_unlock(&g_mu);
        cu_cpu(b, e, p);
        return;
    }
    const int warps = 8, threads = warps * 32;
    const int blocks = (e + warps - 1) / warps;
    mvq_kernel<<<blocks, threads, 0, g.s>>>(dY, (const uint8_t *)dW, (const uint16_t *)dS, dX,
                                            a->in, a->ng, a->group, a->bits,
                                            (unsigned long long)a->rowbytes, e);
    cudaError_t err = cudaMemcpyAsync(a->y, dY, (size_t)e * sizeof(float), cudaMemcpyDeviceToHost, g.s);
    if (err == cudaSuccess) err = cudaStreamSynchronize(g.s);
    waste_cuda_n_calls++;
    pthread_mutex_unlock(&g_mu);
    if (err != cudaSuccess) {
        fprintf(stderr, "waste: CUDA matvec failed (%s); this call ran on the CPU\n", cudaGetErrorString(err));
        cu_cpu(b, e, p);
    }
}

const char *waste_register_cuda(waste_kernels *t)
{
    const char *off = getenv("WASTE_CUDA");
    if (off && !strcmp(off, "0")) return NULL;
    pthread_mutex_lock(&g_mu);
    if (g.ready == 0) {
        int n = 0;
        g.ready = -1;
        if (cudaGetDeviceCount(&n) == cudaSuccess && n > 0) {
            const char *d = getenv("WASTE_CUDA_DEVICE");
            g.dev = d ? atoi(d) : 0;
            cudaDeviceProp pr;
            if (g.dev >= 0 && g.dev < n && cudaSetDevice(g.dev) == cudaSuccess &&
                cudaGetDeviceProperties(&pr, g.dev) == cudaSuccess &&
                cudaStreamCreateWithFlags(&g.s, cudaStreamNonBlocking) == cudaSuccess) {
                snprintf(g.name, sizeof g.name, "CUDA (%s)", pr.name);
                const char *c = getenv("WASTE_CUDA_CACHE_MB");
                g.cache_limit = (size_t)(c ? atoll(c) : 0) << 20;
                g.ready = 1;
            }
        }
    }
    const int ok = g.ready == 1;
    if (ok) {
        t->mvq_rows_f32 = mvq_rows_f32_cuda;
        t->on_device = 1;   /* one launch for the whole range, not one per pool thread */
        const char *m = getenv("WASTE_CUDA_MIN_KB");
        t->device_min_bytes = (size_t)(m ? atoi(m) : 2048) << 10;
    }
    pthread_mutex_unlock(&g_mu);
    return ok ? g.name : NULL;
}

void waste_cuda_release_host_buffers(void)
{
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g.ncache; i++) cudaFree(g.cache[i].dev);
    g.ncache = 0; g.cache_bytes = 0;
    pthread_mutex_unlock(&g_mu);
}
