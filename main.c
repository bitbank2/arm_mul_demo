//
// Arm NEON integer fractional (fixed point) scaling demo
//
// Demonstrates how to use the saturating doubling multiply high
// instruction to reduce the total instruction count of fixed point
// math. Specifically, this demo includes an improved JPEG quantization
// function.
//
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>

typedef signed short DCTELEM;
typedef unsigned short UDCTELEM;
typedef signed short * JCOEFPTR;
#define DCTSIZE 8
#define DCTSIZE2 64

signed short sOut[DCTSIZE2];

unsigned short usDivisors[DCTSIZE2 * 4] = {
 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234,
 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234,
 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234,
 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234,
 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000
};

const short sCoeffs[] = {
  55, 123, 29, 108, 62, 37, 23, 17,
  88, 90, -20, 77, 51, 25, 9, 5,
  16,16,-15,-15,12,12,6,6,
  -1,-1,1,1,-1,-1,1,1,
  -1,-1,1,1,-1,-1,1,1,
  -1,-1,1,1,-1,-1,1,1,
  -1,-1,1,1,-1,-1,1,1,
  -1,-1,1,1,-1,-1,1,1
};

int Micros(void)
{
int iTime;
struct timespec res;

    clock_gettime(CLOCK_MONOTONIC, &res);
    iTime = 1000000*res.tv_sec + res.tv_nsec/1000;

    return iTime;
} /* Micros() */

//
// The following code was taken from release 2.2.1 of libjpeg-turbo
// jquanti-neon.c
//
void jsimd_quantize_opt(JCOEFPTR coef_block, DCTELEM *divisors,
                         DCTELEM *workspace)
{
  JCOEFPTR out_ptr = coef_block;
  UDCTELEM *recip_ptr = (UDCTELEM *)divisors;
  UDCTELEM *corr_ptr = (UDCTELEM *)divisors + DCTSIZE2;
  DCTELEM *shift_ptr = divisors + 3 * DCTSIZE2;
  int i;

  for (i = 0; i < DCTSIZE; i += DCTSIZE / 2) {
    /* Load DCT coefficients. */
      int16x8_t row0 = vld1q_s16(workspace); workspace += DCTSIZE;
      int16x8_t row1 = vld1q_s16(workspace); workspace += DCTSIZE;
      int16x8_t row2 = vld1q_s16(workspace); workspace += DCTSIZE;
      int16x8_t row3 = vld1q_s16(workspace); workspace += DCTSIZE;
    /* Load reciprocals of quantization values. */
      uint16x8_t recip0 = vld1q_u16(recip_ptr); recip_ptr += DCTSIZE;
      uint16x8_t recip1 = vld1q_u16(recip_ptr); recip_ptr += DCTSIZE;
      uint16x8_t recip2 = vld1q_u16(recip_ptr); recip_ptr += DCTSIZE;
      uint16x8_t recip3 = vld1q_u16(recip_ptr); recip_ptr += DCTSIZE;
      uint16x8_t corr0 = vld1q_u16(corr_ptr); corr_ptr += DCTSIZE;
      uint16x8_t corr1 = vld1q_u16(corr_ptr); corr_ptr += DCTSIZE;
      uint16x8_t corr2 = vld1q_u16(corr_ptr); corr_ptr += DCTSIZE;
      uint16x8_t corr3 = vld1q_u16(corr_ptr); corr_ptr += DCTSIZE;
      int16x8_t shift0 = vld1q_s16(shift_ptr); shift_ptr += DCTSIZE;
      int16x8_t shift1 = vld1q_s16(shift_ptr); shift_ptr += DCTSIZE;
      int16x8_t shift2 = vld1q_s16(shift_ptr); shift_ptr += DCTSIZE;
      int16x8_t shift3 = vld1q_s16(shift_ptr); shift_ptr += DCTSIZE;

    /* Extract sign from coefficients. */
    int16x8_t sign_row0 = vshrq_n_s16(row0, 15);
    int16x8_t sign_row1 = vshrq_n_s16(row1, 15);
    int16x8_t sign_row2 = vshrq_n_s16(row2, 15);
    int16x8_t sign_row3 = vshrq_n_s16(row3, 15);
    /* Get absolute value of DCT coefficients. */
    uint16x8_t abs_row0 = vreinterpretq_u16_s16(vabsq_s16(row0));
    uint16x8_t abs_row1 = vreinterpretq_u16_s16(vabsq_s16(row1));
    uint16x8_t abs_row2 = vreinterpretq_u16_s16(vabsq_s16(row2));
    uint16x8_t abs_row3 = vreinterpretq_u16_s16(vabsq_s16(row3));
    /* Add correction. */
    abs_row0 = vaddq_u16(abs_row0, corr0);
    abs_row1 = vaddq_u16(abs_row1, corr1);
    abs_row2 = vaddq_u16(abs_row2, corr2);
    abs_row3 = vaddq_u16(abs_row3, corr3);

    /* Multiply DCT coefficients by quantization reciprocals. */
    // this doubles, saturates and takes the high 16-bits of the result
    // from a signed 16x16 multiply
    row0 = vqdmulhq_s16(vreinterpretq_s16_u16(abs_row0), vreinterpretq_s16_u16(recip0));
    row1 = vqdmulhq_s16(vreinterpretq_s16_u16(abs_row1), vreinterpretq_s16_u16(recip1));
    row2 = vqdmulhq_s16(vreinterpretq_s16_u16(abs_row2), vreinterpretq_s16_u16(recip2));
    row3 = vqdmulhq_s16(vreinterpretq_s16_u16(abs_row3), vreinterpretq_s16_u16(recip3));

    /* Since VSHR only supports an immediate as its second argument, negate the
     * shift value and shift left.
     */
    row0 = vreinterpretq_s16_u16(vshlq_u16(vreinterpretq_u16_s16(row0),
                                           vnegq_s16(shift0)));
    row1 = vreinterpretq_s16_u16(vshlq_u16(vreinterpretq_u16_s16(row1),
                                           vnegq_s16(shift1)));
    row2 = vreinterpretq_s16_u16(vshlq_u16(vreinterpretq_u16_s16(row2),
                                           vnegq_s16(shift2)));
    row3 = vreinterpretq_s16_u16(vshlq_u16(vreinterpretq_u16_s16(row3),
                                           vnegq_s16(shift3)));

    /* Restore sign to original product. */
    row0 = veorq_s16(row0, sign_row0);
    row0 = vsubq_s16(row0, sign_row0);
    row1 = veorq_s16(row1, sign_row1);
    row1 = vsubq_s16(row1, sign_row1);
    row2 = veorq_s16(row2, sign_row2);
    row2 = vsubq_s16(row2, sign_row2);
    row3 = veorq_s16(row3, sign_row3);
    row3 = vsubq_s16(row3, sign_row3);

    /* Store quantized coefficients to memory. */
    vst1q_s16(out_ptr, row0);      out_ptr += DCTSIZE;
    vst1q_s16(out_ptr, row1);      out_ptr += DCTSIZE;
    vst1q_s16(out_ptr, row2);      out_ptr += DCTSIZE;
    vst1q_s16(out_ptr, row3);      out_ptr += DCTSIZE;
  }
} /* jsimd_quantize_opt() */

//
// The following code was taken from release 2.2.1 of libjpeg-turbo
// jquanti-neon.c
//
void jsimd_quantize_neon(JCOEFPTR coef_block, DCTELEM *divisors,
                         DCTELEM *workspace)
{
  JCOEFPTR out_ptr = coef_block;
  UDCTELEM *recip_ptr = (UDCTELEM *)divisors;
  UDCTELEM *corr_ptr = (UDCTELEM *)divisors + DCTSIZE2;
  DCTELEM *shift_ptr = divisors + 3 * DCTSIZE2;
  int i;

  for (i = 0; i < DCTSIZE; i += DCTSIZE / 2) {
    /* Load DCT coefficients. */
    int16x8_t row0 = vld1q_s16(workspace + (i + 0) * DCTSIZE);
    int16x8_t row1 = vld1q_s16(workspace + (i + 1) * DCTSIZE);
    int16x8_t row2 = vld1q_s16(workspace + (i + 2) * DCTSIZE);
    int16x8_t row3 = vld1q_s16(workspace + (i + 3) * DCTSIZE);
    /* Load reciprocals of quantization values. */
    uint16x8_t recip0 = vld1q_u16(recip_ptr + (i + 0) * DCTSIZE);
    uint16x8_t recip1 = vld1q_u16(recip_ptr + (i + 1) * DCTSIZE);
    uint16x8_t recip2 = vld1q_u16(recip_ptr + (i + 2) * DCTSIZE);
    uint16x8_t recip3 = vld1q_u16(recip_ptr + (i + 3) * DCTSIZE);
    uint16x8_t corr0 = vld1q_u16(corr_ptr + (i + 0) * DCTSIZE);
    uint16x8_t corr1 = vld1q_u16(corr_ptr + (i + 1) * DCTSIZE);
    uint16x8_t corr2 = vld1q_u16(corr_ptr + (i + 2) * DCTSIZE);
    uint16x8_t corr3 = vld1q_u16(corr_ptr + (i + 3) * DCTSIZE);
    int16x8_t shift0 = vld1q_s16(shift_ptr + (i + 0) * DCTSIZE);
    int16x8_t shift1 = vld1q_s16(shift_ptr + (i + 1) * DCTSIZE);
    int16x8_t shift2 = vld1q_s16(shift_ptr + (i + 2) * DCTSIZE);
    int16x8_t shift3 = vld1q_s16(shift_ptr + (i + 3) * DCTSIZE);

    /* Extract sign from coefficients. */
    int16x8_t sign_row0 = vshrq_n_s16(row0, 15);
    int16x8_t sign_row1 = vshrq_n_s16(row1, 15);
    int16x8_t sign_row2 = vshrq_n_s16(row2, 15);
    int16x8_t sign_row3 = vshrq_n_s16(row3, 15);
    /* Get absolute value of DCT coefficients. */
    uint16x8_t abs_row0 = vreinterpretq_u16_s16(vabsq_s16(row0));
    uint16x8_t abs_row1 = vreinterpretq_u16_s16(vabsq_s16(row1));
    uint16x8_t abs_row2 = vreinterpretq_u16_s16(vabsq_s16(row2));
    uint16x8_t abs_row3 = vreinterpretq_u16_s16(vabsq_s16(row3));
    /* Add correction. */
    abs_row0 = vaddq_u16(abs_row0, corr0);
    abs_row1 = vaddq_u16(abs_row1, corr1);
    abs_row2 = vaddq_u16(abs_row2, corr2);
    abs_row3 = vaddq_u16(abs_row3, corr3);

    /* Multiply DCT coefficients by quantization reciprocals. */
    int32x4_t row0_l = vreinterpretq_s32_u32(vmull_u16(vget_low_u16(abs_row0),
                                                       vget_low_u16(recip0)));
    int32x4_t row0_h = vreinterpretq_s32_u32(vmull_u16(vget_high_u16(abs_row0),
                                                       vget_high_u16(recip0)));
    int32x4_t row1_l = vreinterpretq_s32_u32(vmull_u16(vget_low_u16(abs_row1),
                                                       vget_low_u16(recip1)));
    int32x4_t row1_h = vreinterpretq_s32_u32(vmull_u16(vget_high_u16(abs_row1),
                                                       vget_high_u16(recip1)));
    int32x4_t row2_l = vreinterpretq_s32_u32(vmull_u16(vget_low_u16(abs_row2),
                                                       vget_low_u16(recip2)));
    int32x4_t row2_h = vreinterpretq_s32_u32(vmull_u16(vget_high_u16(abs_row2),
                                                       vget_high_u16(recip2)));
    int32x4_t row3_l = vreinterpretq_s32_u32(vmull_u16(vget_low_u16(abs_row3),
                                                       vget_low_u16(recip3)));
    int32x4_t row3_h = vreinterpretq_s32_u32(vmull_u16(vget_high_u16(abs_row3),
                                                       vget_high_u16(recip3)));
    /* Narrow back to 16-bit. */
    row0 = vcombine_s16(vshrn_n_s32(row0_l, 16), vshrn_n_s32(row0_h, 16));
    row1 = vcombine_s16(vshrn_n_s32(row1_l, 16), vshrn_n_s32(row1_h, 16));
    row2 = vcombine_s16(vshrn_n_s32(row2_l, 16), vshrn_n_s32(row2_h, 16));
    row3 = vcombine_s16(vshrn_n_s32(row3_l, 16), vshrn_n_s32(row3_h, 16));

    /* Since VSHR only supports an immediate as its second argument, negate the
     * shift value and shift left.
     */
    row0 = vreinterpretq_s16_u16(vshlq_u16(vreinterpretq_u16_s16(row0),
                                           vnegq_s16(shift0)));
    row1 = vreinterpretq_s16_u16(vshlq_u16(vreinterpretq_u16_s16(row1),
                                           vnegq_s16(shift1)));
    row2 = vreinterpretq_s16_u16(vshlq_u16(vreinterpretq_u16_s16(row2),
                                           vnegq_s16(shift2)));
    row3 = vreinterpretq_s16_u16(vshlq_u16(vreinterpretq_u16_s16(row3),
                                           vnegq_s16(shift3)));

    /* Restore sign to original product. */
    row0 = veorq_s16(row0, sign_row0);
    row0 = vsubq_s16(row0, sign_row0);
    row1 = veorq_s16(row1, sign_row1);
    row1 = vsubq_s16(row1, sign_row1);
    row2 = veorq_s16(row2, sign_row2);
    row2 = vsubq_s16(row2, sign_row2);
    row3 = veorq_s16(row3, sign_row3);
    row3 = vsubq_s16(row3, sign_row3);

    /* Store quantized coefficients to memory. */
    vst1q_s16(out_ptr + (i + 0) * DCTSIZE, row0);
    vst1q_s16(out_ptr + (i + 1) * DCTSIZE, row1);
    vst1q_s16(out_ptr + (i + 2) * DCTSIZE, row2);
    vst1q_s16(out_ptr + (i + 3) * DCTSIZE, row3);
  }
} /* jsimd_quantize_neon() */

int main(int argc, char **argv)
{
int i, iTime, iSum;
unsigned short us;

	printf("Arm quantization demo\n");
        // Prepare divisors/corrections/shift values
	for (i=0; i<DCTSIZE2; i++) {
	   us = usDivisors[i];
           usDivisors[i+DCTSIZE2] = us/2; // correction
           usDivisors[i+DCTSIZE2*3] = 0; // no shifting
        }

	// run the original code thousands of times to measure the perf
	iTime = Micros();
        iSum = 0;
	for (i=0; i<100000000; i++) {
	   jsimd_quantize_neon(sOut, (DCTELEM *)usDivisors, (DCTELEM *)sCoeffs); 
           iSum += sOut[0];
	}
        iTime = Micros() - iTime;
        printf("Original time = %d us, val0 = %d\n", iTime, iSum);
        // Prepare divisors for the new code
        for (i=0; i<DCTSIZE2; i++) {
           us = usDivisors[i];
           // use value/2 for the doubling-multiply instruction
           usDivisors[i] = usDivisors[i+DCTSIZE2] = us/2; // correction
           usDivisors[i+DCTSIZE2*3] = 0; // no shifting
        }

        // run the optimized code thousands of times to measure the perf
        memset(sOut, 0, sizeof(sOut));
        iTime = Micros();
        iSum = 0;
        for (i=0; i<100000000; i++) {
           jsimd_quantize_opt(sOut, (DCTELEM *)usDivisors, (DCTELEM *)sCoeffs);
           iSum += sOut[0];
        }
        iTime = Micros() - iTime;
        printf("Optimized time = %d us, val0 = %d\n", iTime, iSum);

	return 0;
}
