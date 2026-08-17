/*
 * Backend-neutral DSP primitives for the embedded CadenceVAD runtime.
 *
 * The hot path needs exactly three things from a math library: a real forward
 * DFT, a matrix-vector product, and a dot product. Apple platforms get them from
 * Accelerate. Everywhere else - Windows above all - gets the self-contained
 * implementations in fv_dsp_portable.c, so the runtime builds with MSVC,
 * clang-cl, GCC or Clang without any external dependency.
 *
 * The DFT contract deliberately matches vDSP_DFT_zrop so the shared core needs no
 * conditional arithmetic: input is the even/odd split of a real frame of length
 * FV_N_FFT, and output is the packed half-spectrum scaled by two, where
 * real[0] carries twice the DC term and imag[0] twice the Nyquist term.
 */
#ifndef FV_DSP_H
#define FV_DSP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque per-stream DFT plan. */
typedef struct FvRealFft FvRealFft;

FvRealFft *fv_rfft_create(size_t n_fft);
void fv_rfft_destroy(FvRealFft *plan);

/*
 * even[i] = x[2i], odd[i] = x[2i+1] for a real frame x of length n_fft.
 * real/imag receive n_fft/2 packed bins using the scaling described above.
 */
void fv_rfft_execute(
    const FvRealFft *plan,
    const float *even,
    const float *odd,
    float *real,
    float *imag
);

/* output = beta * output + alpha * (weight[rows x columns] * input) */
void fv_sgemv(
    int rows,
    int columns,
    float alpha,
    const float *weight,
    const float *input,
    float beta,
    float *output
);

float fv_sdot(int count, const float *left, const float *right);

/* Name of the active backend, for benchmark provenance. */
const char *fv_dsp_backend(void);

#ifdef __cplusplus
}
#endif

#endif
