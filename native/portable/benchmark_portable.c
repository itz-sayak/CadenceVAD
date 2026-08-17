/*
 * Per-hop latency benchmark for the portable CadenceVAD runtime.
 *
 * Reports the distribution of single-hop wall time rather than an average, since
 * a voice pipeline is judged on its tail: one late hop is a late barge-in. Timing
 * uses QueryPerformanceCounter on Windows and CLOCK_MONOTONIC elsewhere.
 *
 * Usage: cadencevad_bench [iterations] [warmup]
 * Output is JSON on stdout so it can be retained as a benchmark artifact.
 */

#include "cadencevad_native.h"
#include "fv_dsp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
static double now_seconds(void) {
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}
static const char *platform_name(void) { return "windows"; }
#else
#include <time.h>
static double now_seconds(void) {
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec * 1e-9;
}
static const char *platform_name(void) {
#if defined(__APPLE__)
    return "macos";
#else
    return "linux";
#endif
}
#endif

static int compare_double(const void *left, const void *right) {
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double percentile(const double *sorted, size_t count, double fraction) {
    if (count == 0) {
        return 0.0;
    }
    size_t index = (size_t)(fraction * (double)(count - 1) + 0.5);
    if (index >= count) {
        index = count - 1;
    }
    return sorted[index];
}

int main(int argc, char **argv) {
    const size_t iterations = argc > 1 ? (size_t)strtoul(argv[1], NULL, 10) : 20000;
    const size_t warmup = argc > 2 ? (size_t)strtoul(argv[2], NULL, 10) : 2000;

    CadenceVadState *state = (CadenceVadState *)malloc(cadencevad_state_size());
    if (state == NULL || cadencevad_init(state) != 0) {
        fprintf(stderr, "cadencevad_init failed\n");
        return 1;
    }

    /* A deterministic non-trivial signal; silence would under-exercise the maths. */
    float hop[FV_HOP_SAMPLES];
    unsigned seed = 12345U;
    for (size_t index = 0; index < FV_HOP_SAMPLES; ++index) {
        seed = seed * 1103515245U + 12345U;
        hop[index] = (float)((seed >> 16) & 0x7FFFU) / 16384.0f - 1.0f;
    }

    for (size_t index = 0; index < warmup; ++index) {
        cadencevad_process_hop(state, hop);
    }

    double *samples = (double *)malloc(iterations * sizeof(double));
    if (samples == NULL) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    volatile float sink = 0.0f;
    for (size_t index = 0; index < iterations; ++index) {
        const double started = now_seconds();
        sink += cadencevad_process_hop(state, hop);
        samples[index] = (now_seconds() - started) * 1e6; /* microseconds */
    }
    (void)sink;

    double total = 0.0;
    for (size_t index = 0; index < iterations; ++index) {
        total += samples[index];
    }
    qsort(samples, iterations, sizeof(double), compare_double);

    printf("{\n");
    printf("  \"schema\": \"cadencevad-portable-runtime-v1\",\n");
    printf("  \"platform\": \"%s\",\n", platform_name());
    printf("  \"dsp_backend\": \"%s\",\n", fv_dsp_backend());
    printf("  \"scope\": \"causal frontend + model, one 10 ms hop, single thread\",\n");
    printf("  \"iterations\": %zu,\n", iterations);
    printf("  \"warmup\": %zu,\n", warmup);
    printf("  \"state_bytes\": %zu,\n", cadencevad_state_size());
    printf("  \"hop_microseconds\": {\n");
    printf("    \"mean\": %.4f,\n", total / (double)iterations);
    printf("    \"p50\": %.4f,\n", percentile(samples, iterations, 0.50));
    printf("    \"p95\": %.4f,\n", percentile(samples, iterations, 0.95));
    printf("    \"p99\": %.4f,\n", percentile(samples, iterations, 0.99));
    printf("    \"min\": %.4f,\n", samples[0]);
    printf("    \"max\": %.4f\n", samples[iterations - 1]);
    printf("  },\n");
    printf(
        "  \"real_time_factor_p50\": %.8f\n",
        percentile(samples, iterations, 0.50) / 10000.0
    );
    printf("}\n");

    free(samples);
    cadencevad_destroy(state);
    free(state);
    return 0;
}
