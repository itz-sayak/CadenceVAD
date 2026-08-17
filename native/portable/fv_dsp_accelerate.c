/*
 * Apple Accelerate backend for the CadenceVAD DSP primitives.
 *
 * This preserves the original macOS hot path exactly: the same vDSP real DFT and
 * the same cblas kernels the runtime shipped with. It exists so the shared core
 * can call one neutral API while Apple platforms keep using Accelerate.
 */

#include "fv_dsp.h"

#include <Accelerate/Accelerate.h>
#include <stdlib.h>

struct FvRealFft {
    vDSP_DFT_Setup setup;
};

FvRealFft *fv_rfft_create(size_t n_fft) {
    FvRealFft *plan = (FvRealFft *)calloc(1, sizeof(FvRealFft));
    if (plan == NULL) {
        return NULL;
    }
    plan->setup = vDSP_DFT_zrop_CreateSetup(NULL, n_fft, vDSP_DFT_FORWARD);
    if (plan->setup == NULL) {
        free(plan);
        return NULL;
    }
    return plan;
}

void fv_rfft_destroy(FvRealFft *plan) {
    if (plan == NULL) {
        return;
    }
    if (plan->setup != NULL) {
        vDSP_DFT_DestroySetup(plan->setup);
    }
    free(plan);
}

void fv_rfft_execute(
    const FvRealFft *plan,
    const float *even,
    const float *odd,
    float *real,
    float *imag
) {
    vDSP_DFT_Execute(plan->setup, even, odd, real, imag);
}

void fv_sgemv(
    int rows,
    int columns,
    float alpha,
    const float *weight,
    const float *input,
    float beta,
    float *output
) {
    cblas_sgemv(
        CblasRowMajor,
        CblasNoTrans,
        rows,
        columns,
        alpha,
        weight,
        columns,
        input,
        1,
        beta,
        output,
        1
    );
}

float fv_sdot(int count, const float *left, const float *right) {
    return cblas_sdot(count, left, 1, right, 1);
}

const char *fv_dsp_backend(void) {
    return "accelerate";
}
