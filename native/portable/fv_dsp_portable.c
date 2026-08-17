/*
 * Self-contained DSP backend for platforms without Apple Accelerate.
 *
 * Nothing here depends on an external library, so the CadenceVAD runtime builds on
 * Windows with MSVC or clang-cl exactly as it does with GCC or Clang elsewhere.
 * The AVX2 paths are opt-in at compile time and fall back to portable scalar code
 * that every C11 compiler can vectorise on its own.
 */

#include "fv_dsp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define FV_HAVE_AVX2 1
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct FvRealFft {
    size_t n_fft;
    size_t half;  /* n_fft / 2: size of the complex FFT actually performed */
    float *cos_table;
    float *sin_table;
    unsigned *reversal;
    float *scratch_real;
    float *scratch_imag;
    float *unpack_cos;
    float *unpack_sin;
};

static unsigned reverse_bits(unsigned value, unsigned bits) {
    unsigned result = 0;
    for (unsigned index = 0; index < bits; ++index) {
        result = (result << 1) | ((value >> index) & 1U);
    }
    return result;
}

static unsigned log2_exact(size_t value) {
    unsigned bits = 0;
    while ((size_t)1 << bits < value) {
        ++bits;
    }
    return bits;
}

FvRealFft *fv_rfft_create(size_t n_fft) {
    if (n_fft < 4 || (n_fft & (n_fft - 1)) != 0) {
        return NULL; /* power of two only */
    }
    FvRealFft *plan = (FvRealFft *)calloc(1, sizeof(FvRealFft));
    if (plan == NULL) {
        return NULL;
    }
    plan->n_fft = n_fft;
    plan->half = n_fft / 2;

    const unsigned bits = log2_exact(plan->half);
    plan->cos_table = (float *)malloc(plan->half / 2 * sizeof(float));
    plan->sin_table = (float *)malloc(plan->half / 2 * sizeof(float));
    plan->reversal = (unsigned *)malloc(plan->half * sizeof(unsigned));
    plan->scratch_real = (float *)malloc(plan->half * sizeof(float));
    plan->scratch_imag = (float *)malloc(plan->half * sizeof(float));
    plan->unpack_cos = (float *)malloc((plan->half / 2 + 1) * sizeof(float));
    plan->unpack_sin = (float *)malloc((plan->half / 2 + 1) * sizeof(float));
    if (plan->cos_table == NULL || plan->sin_table == NULL || plan->reversal == NULL ||
        plan->scratch_real == NULL || plan->scratch_imag == NULL ||
        plan->unpack_cos == NULL || plan->unpack_sin == NULL) {
        fv_rfft_destroy(plan);
        return NULL;
    }

    for (size_t index = 0; index < plan->half / 2; ++index) {
        const double angle = -2.0 * M_PI * (double)index / (double)plan->half;
        plan->cos_table[index] = (float)cos(angle);
        plan->sin_table[index] = (float)sin(angle);
    }
    for (size_t index = 0; index < plan->half; ++index) {
        plan->reversal[index] = reverse_bits((unsigned)index, bits);
    }
    for (size_t index = 0; index <= plan->half / 2; ++index) {
        const double angle = -2.0 * M_PI * (double)index / (double)n_fft;
        plan->unpack_cos[index] = (float)cos(angle);
        plan->unpack_sin[index] = (float)sin(angle);
    }
    return plan;
}

void fv_rfft_destroy(FvRealFft *plan) {
    if (plan == NULL) {
        return;
    }
    free(plan->cos_table);
    free(plan->sin_table);
    free(plan->reversal);
    free(plan->scratch_real);
    free(plan->scratch_imag);
    free(plan->unpack_cos);
    free(plan->unpack_sin);
    free(plan);
}

/* In-place iterative radix-2 complex FFT over plan->half points. */
static void complex_fft(const FvRealFft *plan, float *real, float *imag) {
    const size_t count = plan->half;

    for (size_t index = 0; index < count; ++index) {
        const size_t target = plan->reversal[index];
        if (target > index) {
            float swap = real[index];
            real[index] = real[target];
            real[target] = swap;
            swap = imag[index];
            imag[index] = imag[target];
            imag[target] = swap;
        }
    }

    for (size_t length = 2; length <= count; length <<= 1) {
        const size_t half = length / 2;
        const size_t stride = count / length;
        for (size_t start = 0; start < count; start += length) {
            for (size_t offset = 0; offset < half; ++offset) {
                const size_t twiddle = offset * stride;
                const float cosine = plan->cos_table[twiddle];
                const float sine = plan->sin_table[twiddle];
                const size_t upper = start + offset + half;
                const size_t lower = start + offset;
                const float product_real = real[upper] * cosine - imag[upper] * sine;
                const float product_imag = real[upper] * sine + imag[upper] * cosine;
                real[upper] = real[lower] - product_real;
                imag[upper] = imag[lower] - product_imag;
                real[lower] += product_real;
                imag[lower] += product_imag;
            }
        }
    }
}

void fv_rfft_execute(
    const FvRealFft *plan,
    const float *even,
    const float *odd,
    float *real,
    float *imag
) {
    if (plan == NULL) {
        return;
    }
    const size_t half = plan->half;
    float *zr = plan->scratch_real;
    float *zi = plan->scratch_imag;

    /*
     * The even/odd split already is the complex sequence z[n] = x[2n] + i x[2n+1],
     * so one half-length complex FFT recovers the real spectrum.
     */
    memcpy(zr, even, half * sizeof(float));
    memcpy(zi, odd, half * sizeof(float));
    complex_fft(plan, zr, zi);

    /*
     * Unpack: X[k] = E[k] + e^{-2 pi i k / N} O[k], where E and O are the even and
     * odd DFTs recovered from the conjugate-symmetric halves of Z. Outputs are
     * scaled by two to match the vDSP_DFT_zrop convention the shared core expects,
     * with the Nyquist term folded into imag[0].
     */
    const float dc_even = zr[0] + zi[0];
    const float dc_odd = zr[0] - zi[0];
    real[0] = 2.0f * dc_even;   /* twice X[0] */
    imag[0] = 2.0f * dc_odd;    /* twice X[N/2] */

    for (size_t k = 1; k < half; ++k) {
        const size_t mirror = half - k;
        const float even_real = 0.5f * (zr[k] + zr[mirror]);
        const float even_imag = 0.5f * (zi[k] - zi[mirror]);
        const float odd_real = 0.5f * (zi[k] + zi[mirror]);
        const float odd_imag = -0.5f * (zr[k] - zr[mirror]);

        float cosine;
        float sine;
        if (k <= half / 2) {
            cosine = plan->unpack_cos[k];
            sine = plan->unpack_sin[k];
        } else {
            /* e^{-2 pi i k / N} for k > N/4 mirrors the stored quarter table. */
            cosine = -plan->unpack_cos[half - k];
            sine = plan->unpack_sin[half - k];
        }
        const float rotated_real = odd_real * cosine - odd_imag * sine;
        const float rotated_imag = odd_real * sine + odd_imag * cosine;
        real[k] = 2.0f * (even_real + rotated_real);
        imag[k] = 2.0f * (even_imag + rotated_imag);
    }
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
    for (int row = 0; row < rows; ++row) {
        const float *line = weight + (size_t)row * (size_t)columns;
        float total = 0.0f;
#if FV_HAVE_AVX2
        __m256 accumulator = _mm256_setzero_ps();
        int column = 0;
        for (; column + 8 <= columns; column += 8) {
            accumulator = _mm256_fmadd_ps(
                _mm256_loadu_ps(line + column),
                _mm256_loadu_ps(input + column),
                accumulator
            );
        }
        __m128 low = _mm256_castps256_ps128(accumulator);
        __m128 high = _mm256_extractf128_ps(accumulator, 1);
        low = _mm_add_ps(low, high);
        low = _mm_hadd_ps(low, low);
        low = _mm_hadd_ps(low, low);
        total = _mm_cvtss_f32(low);
        for (; column < columns; ++column) {
            total += line[column] * input[column];
        }
#else
        for (int column = 0; column < columns; ++column) {
            total += line[column] * input[column];
        }
#endif
        output[row] = beta * output[row] + alpha * total;
    }
}

float fv_sdot(int count, const float *left, const float *right) {
    float total = 0.0f;
#if FV_HAVE_AVX2
    __m256 accumulator = _mm256_setzero_ps();
    int index = 0;
    for (; index + 8 <= count; index += 8) {
        accumulator = _mm256_fmadd_ps(
            _mm256_loadu_ps(left + index),
            _mm256_loadu_ps(right + index),
            accumulator
        );
    }
    __m128 low = _mm256_castps256_ps128(accumulator);
    __m128 high = _mm256_extractf128_ps(accumulator, 1);
    low = _mm_add_ps(low, high);
    low = _mm_hadd_ps(low, low);
    low = _mm_hadd_ps(low, low);
    total = _mm_cvtss_f32(low);
    for (; index < count; ++index) {
        total += left[index] * right[index];
    }
#else
    for (int index = 0; index < count; ++index) {
        total += left[index] * right[index];
    }
#endif
    return total;
}

const char *fv_dsp_backend(void) {
#if FV_HAVE_AVX2
    return "portable-avx2";
#else
    return "portable-scalar";
#endif
}
