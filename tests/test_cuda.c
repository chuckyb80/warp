/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * test_cuda.c — the CUDA mvq_rows_f32 against the portable CPU kernel.
 *
 *   make WASTE_ENABLE_CUDA=1 cuda-check
 *   WASTE_CUDA_CACHE_MB=2048 ./test_cuda      (same, with weights resident)
 *
 * For every bit width the engine stores (8, 4, 3), group sizes 128 and 64,
 * and an input length that leaves a ragged last group, it fills random
 * weights and scales, runs both kernels, and requires the largest error to
 * stay under 1e-4 of the largest output (float rounding in a different
 * summation order, not a different answer). Then it times one trunk-sized
 * Q4G matvec on the device and on one CPU thread.
 * Exit status: 0 pass, 1 fail, 77 skip (no usable CUDA device).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/waste_backend.h"
#include "../src/simd.h"

void waste_mvq_rows_f32(int b, int e, void *p);
extern unsigned long long waste_cuda_n_calls, waste_cuda_n_fallback,
                          waste_cuda_bytes_h2d, waste_cuda_cache_hits;

static uint32_t rs = 12345u;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static float frand(void) { return (float)(rnd() & 0xffffff) / 16777216.0f; }

/* f32 -> f16 bits for normal, positive values (all this test makes). */
static uint16_t to_f16(float f)
{
    uint32_t u; memcpy(&u, &f, 4);
    int e = (int)((u >> 23) & 0xff) - 127 + 15;
    uint32_t m = (u >> 13) & 0x3ff;
    if (e <= 0) return 0;
    if (e >= 31) return 0x7bff;
    return (uint16_t)((e << 10) | m);
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static size_t row_bytes(int bits, int ng, int g) { return ((size_t)ng * g * bits + 7) / 8 + 1; }

static int fails = 0;

static void run_case(int rows, int in, int g, int bits)
{
    const int ng = (in + g - 1) / g;
    const size_t rb = row_bytes(bits, ng, g);
    uint8_t *W = malloc((size_t)rows * rb);
    uint16_t *ws = malloc((size_t)rows * ng * sizeof(uint16_t));
    float *x = malloc((size_t)in * sizeof(float));
    float *y0 = malloc((size_t)rows * sizeof(float)), *y1 = malloc((size_t)rows * sizeof(float));
    for (size_t i = 0; i < (size_t)rows * rb; i++) W[i] = (uint8_t)rnd();
    for (size_t i = 0; i < (size_t)rows * ng; i++) ws[i] = to_f16(0.001f + 0.05f * frand());
    for (int i = 0; i < in; i++) x[i] = 2.0f * frand() - 1.0f;
    mvq_arg a = { y0, (const int8_t *)W, ws, NULL, x, in, ng, g, bits, rb };
    waste_mvq_rows_f32(0, rows, &a);                 /* CPU reference */
    a.y = y1;
    const unsigned long long before = waste_cuda_n_calls;
    waste_k.mvq_rows_f32(0, rows, &a);               /* the device */
    double maxd = 0, maxr = 0;
    for (int i = 0; i < rows; i++) {
        const double d = fabs((double)y1[i] - y0[i]);
        if (d > maxd) maxd = d;
        if (fabs(y0[i]) > maxr) maxr = fabs(y0[i]);
    }
    const int on_dev = waste_cuda_n_calls > before;
    const int ok = on_dev && maxd <= 1e-4 * (maxr > 0 ? maxr : 1);
    printf("  %s  Q%dG g=%-3d %5d x %-5d  max err %.2e of %.3g%s\n", ok ? "PASS" : "FAIL",
           bits, g, rows, in, maxd, maxr, on_dev ? "" : "  (ran on the CPU, not the device)");
    if (!ok) fails++;
    free(W); free(ws); free(x); free(y0); free(y1);
}

static void bench(int rows, int in, int g, int bits)
{
    const int ng = (in + g - 1) / g;
    const size_t rb = row_bytes(bits, ng, g);
    uint8_t *W = malloc((size_t)rows * rb);
    uint16_t *ws = malloc((size_t)rows * ng * sizeof(uint16_t));
    float *x = malloc((size_t)in * sizeof(float)), *y = malloc((size_t)rows * sizeof(float));
    for (size_t i = 0; i < (size_t)rows * rb; i++) W[i] = (uint8_t)rnd();
    for (size_t i = 0; i < (size_t)rows * ng; i++) ws[i] = to_f16(0.01f);
    for (int i = 0; i < in; i++) x[i] = frand();
    mvq_arg a = { y, (const int8_t *)W, ws, NULL, x, in, ng, g, bits, rb };
    const double mb = (double)rows * rb / 1e6;
    waste_k.mvq_rows_f32(0, rows, &a);               /* warm up (and fill the cache if on) */
    const int reps = 20;
    double t0 = now();
    for (int r = 0; r < reps; r++) waste_k.mvq_rows_f32(0, rows, &a);
    const double gpu = (now() - t0) / reps;
    t0 = now();
    waste_mvq_rows_f32(0, rows, &a);
    const double cpu1 = now() - t0;
    printf("  bench Q%dG %d x %d (%.1f MB of weights): device %.2f ms (%.1f GB/s), one CPU thread %.2f ms; cache hits %llu\n",
           bits, rows, in, mb, gpu * 1e3, mb / 1e3 / gpu, cpu1 * 1e3, waste_cuda_cache_hits);
    free(W); free(ws); free(x); free(y);
}

int main(void)
{
    setenv("WASTE_CUDA_MIN_KB", "0", 0);
    waste_backend_init(WASTE_BE_AUTO);
    const char *name = waste_backend_name();
    printf("backend: %s\n", name);
    if (!strstr(name, "CUDA")) { printf("SKIP  no usable CUDA device (or WASTE_CUDA=0)\n"); return 77; }
    const int bits[3] = { 8, 4, 3 };
    for (int b = 0; b < 3; b++) {
        run_case(4096, 2048, 128, bits[b]);
        run_case(1000, 2048 + 77, 128, bits[b]);     /* ragged last group */
        run_case(513, 1536, 64, bits[b]);
    }
    bench(12288, 7168, 128, 4);
    bench(12288, 7168, 128, 8);
    printf("calls on the device %llu, CPU fallbacks %llu, %.1f MB copied to the device\n",
           waste_cuda_n_calls, waste_cuda_n_fallback, waste_cuda_bytes_h2d / 1e6);
    printf(fails ? "\n%d FAILED\n" : "\nall passed\n", fails);
    waste_backend_release_host_buffers();
    return fails ? 1 : 0;
}
