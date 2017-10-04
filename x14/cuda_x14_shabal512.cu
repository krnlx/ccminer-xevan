/*
 * Shabal-512 for X14/X15
 * Provos Alexis - 2016
 */
#include "cuda_helper.h"
#include "cuda_vectors.h"

/* $Id: shabal.c 175 2010-05-07 16:03:20Z tp $ */
/*
 * Shabal implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2007-2010 Projet RNRT SAPHIR
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS B[14] LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author Thomas Pornin <thomas.pornin@cryptolog.com>
 */
/*
__device__ __forceinline__ void PERM_ELT(uint32_t &xa0,const uint32_t xa1,uint32_t &xb0,const uint32_t xb1,const uint32_t xb2,const uint32_t xb3,const uint32_t xc,const uint32_t xm){

		uint32_t tmp;
		#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
			asm ("lop3.b32 %0, %1, %2, %3, 0x9A;" : "=r"(tmp)	: "r"(xb2),"r"(xb3),"r"(xm));		// 0x9A = (F0 &(~CC)) ^ (AA)
		#else
			tmp = (xb2 & ~xb3) ^ xm;
		#endif
		
		xa0 = ((xa0 ^ xc ^ (ROTL32(xa1, 15) * 5U)) * 3U) ^ xb1 ^ tmp;
		xb0 = xor3x(0xFFFFFFFF, xa0, ROTL32(xb0, 1));
}

__device__ __forceinline__
void PERM_STEP_0(uint32_t *A,uint32_t *B,const uint32_t *C,const uint32_t* M){
		PERM_ELT(A[ 0], A[11], B[ 0], B[13], B[ 9], B[ 6], C[ 8], M[ 0]); PERM_ELT(A[ 1], A[ 0], B[ 1], B[14], B[10], B[ 7], C[ 7], M[ 1]);
		PERM_ELT(A[ 2], A[ 1], B[ 2], B[15], B[11], B[ 8], C[ 6], M[ 2]); PERM_ELT(A[ 3], A[ 2], B[ 3], B[ 0], B[12], B[ 9], C[ 5], M[ 3]);
		PERM_ELT(A[ 4], A[ 3], B[ 4], B[ 1], B[13], B[10], C[ 4], M[ 4]); PERM_ELT(A[ 5], A[ 4], B[ 5], B[ 2], B[14], B[11], C[ 3], M[ 5]);
		PERM_ELT(A[ 6], A[ 5], B[ 6], B[ 3], B[15], B[12], C[ 2], M[ 6]); PERM_ELT(A[ 7], A[ 6], B[ 7], B[ 4], B[ 0], B[13], C[ 1], M[ 7]);
		PERM_ELT(A[ 8], A[ 7], B[ 8], B[ 5], B[ 1], B[14], C[ 0], M[ 8]); PERM_ELT(A[ 9], A[ 8], B[ 9], B[ 6], B[ 2], B[15], C[15], M[ 9]);
		PERM_ELT(A[10], A[ 9], B[10], B[ 7], B[ 3], B[ 0], C[14], M[10]); PERM_ELT(A[11], A[10], B[11], B[ 8], B[ 4], B[ 1], C[13], M[11]);
		PERM_ELT(A[ 0], A[11], B[12], B[ 9], B[ 5], B[ 2], C[12], M[12]); PERM_ELT(A[ 1], A[ 0], B[13], B[10], B[ 6], B[ 3], C[11], M[13]);
		PERM_ELT(A[ 2], A[ 1], B[14], B[11], B[ 7], B[ 4], C[10], M[14]); PERM_ELT(A[ 3], A[ 2], B[15], B[12], B[ 8], B[ 5], C[ 9], M[15]);
}

__device__ __forceinline__
void PERM_STEP_1(uint32_t *A,uint32_t *B,const uint32_t *C,const uint32_t* M){
		PERM_ELT(A[ 4], A[ 3], B[ 0], B[13], B[ 9], B[ 6], C[ 8], M[ 0]); PERM_ELT(A[ 5], A[ 4], B[ 1], B[14], B[10], B[ 7], C[ 7], M[ 1]);
		PERM_ELT(A[ 6], A[ 5], B[ 2], B[15], B[11], B[ 8], C[ 6], M[ 2]); PERM_ELT(A[ 7], A[ 6], B[ 3], B[ 0], B[12], B[ 9], C[ 5], M[ 3]);
		PERM_ELT(A[ 8], A[ 7], B[ 4], B[ 1], B[13], B[10], C[ 4], M[ 4]); PERM_ELT(A[ 9], A[ 8], B[ 5], B[ 2], B[14], B[11], C[ 3], M[ 5]);
		PERM_ELT(A[10], A[ 9], B[ 6], B[ 3], B[15], B[12], C[ 2], M[ 6]); PERM_ELT(A[11], A[10], B[ 7], B[ 4], B[ 0], B[13], C[ 1], M[ 7]);
		PERM_ELT(A[ 0], A[11], B[ 8], B[ 5], B[ 1], B[14], C[ 0], M[ 8]); PERM_ELT(A[ 1], A[ 0], B[ 9], B[ 6], B[ 2], B[15], C[15], M[ 9]);
		PERM_ELT(A[ 2], A[ 1], B[10], B[ 7], B[ 3], B[ 0], C[14], M[10]); PERM_ELT(A[ 3], A[ 2], B[11], B[ 8], B[ 4], B[ 1], C[13], M[11]);
		PERM_ELT(A[ 4], A[ 3], B[12], B[ 9], B[ 5], B[ 2], C[12], M[12]); PERM_ELT(A[ 5], A[ 4], B[13], B[10], B[ 6], B[ 3], C[11], M[13]);
		PERM_ELT(A[ 6], A[ 5], B[14], B[11], B[ 7], B[ 4], C[10], M[14]); PERM_ELT(A[ 7], A[ 6], B[15], B[12], B[ 8], B[ 5], C[ 9], M[15]);
}

__device__ __forceinline__
void PERM_STEP_2(uint32_t *A,uint32_t *B,const uint32_t *C,const uint32_t* M){
		PERM_ELT(A[ 8], A[ 7], B[ 0], B[13], B[ 9], B[ 6], C[ 8], M[ 0]); PERM_ELT(A[ 9], A[ 8], B[ 1], B[14], B[10], B[ 7], C[ 7], M[ 1]);
		PERM_ELT(A[10], A[ 9], B[ 2], B[15], B[11], B[ 8], C[ 6], M[ 2]); PERM_ELT(A[11], A[10], B[ 3], B[ 0], B[12], B[ 9], C[ 5], M[ 3]);
		PERM_ELT(A[ 0], A[11], B[ 4], B[ 1], B[13], B[10], C[ 4], M[ 4]); PERM_ELT(A[ 1], A[ 0], B[ 5], B[ 2], B[14], B[11], C[ 3], M[ 5]);
		PERM_ELT(A[ 2], A[ 1], B[ 6], B[ 3], B[15], B[12], C[ 2], M[ 6]); PERM_ELT(A[ 3], A[ 2], B[ 7], B[ 4], B[ 0], B[13], C[ 1], M[ 7]);
		PERM_ELT(A[ 4], A[ 3], B[ 8], B[ 5], B[ 1], B[14], C[ 0], M[ 8]); PERM_ELT(A[ 5], A[ 4], B[ 9], B[ 6], B[ 2], B[15], C[15], M[ 9]);
		PERM_ELT(A[ 6], A[ 5], B[10], B[ 7], B[ 3], B[ 0], C[14], M[10]); PERM_ELT(A[ 7], A[ 6], B[11], B[ 8], B[ 4], B[ 1], C[13], M[11]);
		PERM_ELT(A[ 8], A[ 7], B[12], B[ 9], B[ 5], B[ 2], C[12], M[12]); PERM_ELT(A[ 9], A[ 8], B[13], B[10], B[ 6], B[ 3], C[11], M[13]);
		PERM_ELT(A[10], A[ 9], B[14], B[11], B[ 7], B[ 4], C[10], M[14]); PERM_ELT(A[11], A[10], B[15], B[12], B[ 8], B[ 5], C[ 9], M[15]);
}

__device__ __forceinline__
void ADD_BLOCK(uint32_t* A, const uint32_t *B){
	A[11]+= B[ 6]; A[10]+= B[ 5]; A[ 9]+= B[ 4]; A[ 8]+= B[ 3]; A[ 7]+= B[ 2]; A[ 6]+= B[ 1]; A[ 5]+= B[ 0]; A[ 4]+= B[15]; A[ 3]+= B[14]; A[ 2]+= B[13]; A[ 1]+= B[12]; A[ 0]+= B[11];
	A[11]+= B[10]; A[10]+= B[ 9]; A[ 9]+= B[ 8]; A[ 8]+= B[ 7]; A[ 7]+= B[ 6]; A[ 6]+= B[ 5]; A[ 5]+= B[ 4]; A[ 4]+= B[ 3]; A[ 3]+= B[ 2]; A[ 2]+= B[ 1]; A[ 1]+= B[ 0]; A[ 0]+= B[15];
	A[11]+= B[14]; A[10]+= B[13]; A[ 9]+= B[12]; A[ 8]+= B[11]; A[ 7]+= B[10]; A[ 6]+= B[ 9]; A[ 5]+= B[ 8]; A[ 4]+= B[ 7]; A[ 3]+= B[ 6]; A[ 2]+= B[ 5]; A[ 1]+= B[ 4]; A[ 0]+= B[ 3];
}
__device__ __forceinline__
void ROTATE(uint32_t* A){
	#pragma unroll 16
	for(int i=0;i<16;i++){
		A[ i] = ROTL32(A[ i],17);
	}
}

*/
/***************************************************/


#define sph_u32 uint32_t
#define sM    16

#define C32   SPH_C32
#define T32   SPH_T32

#define O1   13
#define O2    9
#define O3    6


#define INPUT_BLOCK_ADD   do { \
		B0 = T32(B0 + M0); \
		B1 = T32(B1 + M1); \
		B2 = T32(B2 + M2); \
		B3 = T32(B3 + M3); \
		B4 = T32(B4 + M4); \
		B5 = T32(B5 + M5); \
		B6 = T32(B6 + M6); \
		B7 = T32(B7 + M7); \
		B8 = T32(B8 + M8); \
		B9 = T32(B9 + M9); \
		BA = T32(BA + MA); \
		BB = T32(BB + MB); \
		BC = T32(BC + MC); \
		BD = T32(BD + MD); \
		BE = T32(BE + ME); \
		BF = T32(BF + MF); \
	} while (0)

#define INPUT_BLOCK_SUB   do { \
		C0 = T32(C0 - M0); \
		C1 = T32(C1 - M1); \
		C2 = T32(C2 - M2); \
		C3 = T32(C3 - M3); \
		C4 = T32(C4 - M4); \
		C5 = T32(C5 - M5); \
		C6 = T32(C6 - M6); \
		C7 = T32(C7 - M7); \
		C8 = T32(C8 - M8); \
		C9 = T32(C9 - M9); \
		CA = T32(CA - MA); \
		CB = T32(CB - MB); \
		CC = T32(CC - MC); \
		CD = T32(CD - MD); \
		CE = T32(CE - ME); \
		CF = T32(CF - MF); \
	} while (0)

#define XOR_W   do { \
		A00 ^= Wlow; \
		A01 ^= Whigh; \
	} while (0)

#define SWAP(v1, v2)   do { \
		sph_u32 tmp = (v1); \
		(v1) = (v2); \
		(v2) = tmp; \
	} while (0)

#define SWAP_BC   do { \
		SWAP(B0, C0); \
		SWAP(B1, C1); \
		SWAP(B2, C2); \
		SWAP(B3, C3); \
		SWAP(B4, C4); \
		SWAP(B5, C5); \
		SWAP(B6, C6); \
		SWAP(B7, C7); \
		SWAP(B8, C8); \
		SWAP(B9, C9); \
		SWAP(BA, CA); \
		SWAP(BB, CB); \
		SWAP(BC, CC); \
		SWAP(BD, CD); \
		SWAP(BE, CE); \
		SWAP(BF, CF); \
	} while (0)

#define PERM_ELT(xa0, xa1, xb0, xb1, xb2, xb3, xc, xm)   do { \
		xa0 = T32((xa0 \
			^ (((xa1 << 15) | (xa1 >> 17)) * 5U) \
			^ xc) * 3U) \
			^ xb1 ^ (xb2 & ~xb3) ^ xm; \
		xb0 = T32(~(((xb0 << 1) | (xb0 >> 31)) ^ xa0)); \
	} while (0)

#define PERM_STEP_0   do { \
		PERM_ELT(A00, A0B, B0, BD, B9, B6, C8, M0); \
		PERM_ELT(A01, A00, B1, BE, BA, B7, C7, M1); \
		PERM_ELT(A02, A01, B2, BF, BB, B8, C6, M2); \
		PERM_ELT(A03, A02, B3, B0, BC, B9, C5, M3); \
		PERM_ELT(A04, A03, B4, B1, BD, BA, C4, M4); \
		PERM_ELT(A05, A04, B5, B2, BE, BB, C3, M5); \
		PERM_ELT(A06, A05, B6, B3, BF, BC, C2, M6); \
		PERM_ELT(A07, A06, B7, B4, B0, BD, C1, M7); \
		PERM_ELT(A08, A07, B8, B5, B1, BE, C0, M8); \
		PERM_ELT(A09, A08, B9, B6, B2, BF, CF, M9); \
		PERM_ELT(A0A, A09, BA, B7, B3, B0, CE, MA); \
		PERM_ELT(A0B, A0A, BB, B8, B4, B1, CD, MB); \
		PERM_ELT(A00, A0B, BC, B9, B5, B2, CC, MC); \
		PERM_ELT(A01, A00, BD, BA, B6, B3, CB, MD); \
		PERM_ELT(A02, A01, BE, BB, B7, B4, CA, ME); \
		PERM_ELT(A03, A02, BF, BC, B8, B5, C9, MF); \
	} while (0)

#define PERM_STEP_1   do { \
		PERM_ELT(A04, A03, B0, BD, B9, B6, C8, M0); \
		PERM_ELT(A05, A04, B1, BE, BA, B7, C7, M1); \
		PERM_ELT(A06, A05, B2, BF, BB, B8, C6, M2); \
		PERM_ELT(A07, A06, B3, B0, BC, B9, C5, M3); \
		PERM_ELT(A08, A07, B4, B1, BD, BA, C4, M4); \
		PERM_ELT(A09, A08, B5, B2, BE, BB, C3, M5); \
		PERM_ELT(A0A, A09, B6, B3, BF, BC, C2, M6); \
		PERM_ELT(A0B, A0A, B7, B4, B0, BD, C1, M7); \
		PERM_ELT(A00, A0B, B8, B5, B1, BE, C0, M8); \
		PERM_ELT(A01, A00, B9, B6, B2, BF, CF, M9); \
		PERM_ELT(A02, A01, BA, B7, B3, B0, CE, MA); \
		PERM_ELT(A03, A02, BB, B8, B4, B1, CD, MB); \
		PERM_ELT(A04, A03, BC, B9, B5, B2, CC, MC); \
		PERM_ELT(A05, A04, BD, BA, B6, B3, CB, MD); \
		PERM_ELT(A06, A05, BE, BB, B7, B4, CA, ME); \
		PERM_ELT(A07, A06, BF, BC, B8, B5, C9, MF); \
	} while (0)

#define PERM_STEP_2   do { \
		PERM_ELT(A08, A07, B0, BD, B9, B6, C8, M0); \
		PERM_ELT(A09, A08, B1, BE, BA, B7, C7, M1); \
		PERM_ELT(A0A, A09, B2, BF, BB, B8, C6, M2); \
		PERM_ELT(A0B, A0A, B3, B0, BC, B9, C5, M3); \
		PERM_ELT(A00, A0B, B4, B1, BD, BA, C4, M4); \
		PERM_ELT(A01, A00, B5, B2, BE, BB, C3, M5); \
		PERM_ELT(A02, A01, B6, B3, BF, BC, C2, M6); \
		PERM_ELT(A03, A02, B7, B4, B0, BD, C1, M7); \
		PERM_ELT(A04, A03, B8, B5, B1, BE, C0, M8); \
		PERM_ELT(A05, A04, B9, B6, B2, BF, CF, M9); \
		PERM_ELT(A06, A05, BA, B7, B3, B0, CE, MA); \
		PERM_ELT(A07, A06, BB, B8, B4, B1, CD, MB); \
		PERM_ELT(A08, A07, BC, B9, B5, B2, CC, MC); \
		PERM_ELT(A09, A08, BD, BA, B6, B3, CB, MD); \
		PERM_ELT(A0A, A09, BE, BB, B7, B4, CA, ME); \
		PERM_ELT(A0B, A0A, BF, BC, B8, B5, C9, MF); \
	} while (0)

#define APPLY_P   do { \
		B0 = T32(B0 << 17) | (B0 >> 15); \
		B1 = T32(B1 << 17) | (B1 >> 15); \
		B2 = T32(B2 << 17) | (B2 >> 15); \
		B3 = T32(B3 << 17) | (B3 >> 15); \
		B4 = T32(B4 << 17) | (B4 >> 15); \
		B5 = T32(B5 << 17) | (B5 >> 15); \
		B6 = T32(B6 << 17) | (B6 >> 15); \
		B7 = T32(B7 << 17) | (B7 >> 15); \
		B8 = T32(B8 << 17) | (B8 >> 15); \
		B9 = T32(B9 << 17) | (B9 >> 15); \
		BA = T32(BA << 17) | (BA >> 15); \
		BB = T32(BB << 17) | (BB >> 15); \
		BC = T32(BC << 17) | (BC >> 15); \
		BD = T32(BD << 17) | (BD >> 15); \
		BE = T32(BE << 17) | (BE >> 15); \
		BF = T32(BF << 17) | (BF >> 15); \
		PERM_STEP_0; \
		PERM_STEP_1; \
		PERM_STEP_2; \
		A0B = T32(A0B + C6); \
		A0A = T32(A0A + C5); \
		A09 = T32(A09 + C4); \
		A08 = T32(A08 + C3); \
		A07 = T32(A07 + C2); \
		A06 = T32(A06 + C1); \
		A05 = T32(A05 + C0); \
		A04 = T32(A04 + CF); \
		A03 = T32(A03 + CE); \
		A02 = T32(A02 + CD); \
		A01 = T32(A01 + CC); \
		A00 = T32(A00 + CB); \
		A0B = T32(A0B + CA); \
		A0A = T32(A0A + C9); \
		A09 = T32(A09 + C8); \
		A08 = T32(A08 + C7); \
		A07 = T32(A07 + C6); \
		A06 = T32(A06 + C5); \
		A05 = T32(A05 + C4); \
		A04 = T32(A04 + C3); \
		A03 = T32(A03 + C2); \
		A02 = T32(A02 + C1); \
		A01 = T32(A01 + C0); \
		A00 = T32(A00 + CF); \
		A0B = T32(A0B + CE); \
		A0A = T32(A0A + CD); \
		A09 = T32(A09 + CC); \
		A08 = T32(A08 + CB); \
		A07 = T32(A07 + CA); \
		A06 = T32(A06 + C9); \
		A05 = T32(A05 + C8); \
		A04 = T32(A04 + C7); \
		A03 = T32(A03 + C6); \
		A02 = T32(A02 + C5); \
		A01 = T32(A01 + C4); \
		A00 = T32(A00 + C3); \
	} while (0)

#define INCR_W   do { \
		if ((Wlow = T32(Wlow + 1)) == 0) \
			Whigh = T32(Whigh + 1); \
	} while (0)

__constant__ static const sph_u32 A_init_512[] = {
	C32(0x20728DFD), C32(0x46C0BD53), C32(0xE782B699), C32(0x55304632),
	C32(0x71B4EF90), C32(0x0EA9E82C), C32(0xDBB930F1), C32(0xFAD06B8B),
	C32(0xBE0CAE40), C32(0x8BD14410), C32(0x76D2ADAC), C32(0x28ACAB7F)
};

__constant__ static const sph_u32 B_init_512[] = {
	C32(0xC1099CB7), C32(0x07B385F3), C32(0xE7442C26), C32(0xCC8AD640),
	C32(0xEB6F56C7), C32(0x1EA81AA9), C32(0x73B9D314), C32(0x1DE85D08),
	C32(0x48910A5A), C32(0x893B22DB), C32(0xC5A0DF44), C32(0xBBC4324E),
	C32(0x72D2F240), C32(0x75941D99), C32(0x6D8BDE82), C32(0xA1A7502B)
};

__constant__ static const sph_u32 C_init_512[] = {
	C32(0xD9BF68D1), C32(0x58BAD750), C32(0x56028CB2), C32(0x8134F359),
	C32(0xB5D469D8), C32(0x941A8CC2), C32(0x418B2A6E), C32(0x04052780),
	C32(0x7F07D787), C32(0x5194358F), C32(0x3C60D665), C32(0xBE97D79A),
	C32(0x950C3434), C32(0xAED9A06D), C32(0x2537DC8D), C32(0x7CDB5969)
};


#define TPB_SHABAL 256
// GPU Hash Function
__global__ __launch_bounds__(TPB_SHABAL,2)
void x14_shabal512_gpu_hash_64(uint32_t threads, uint32_t *g_hash){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint32_t A[]={
			0x20728DFD, 0x46C0BD53, 0xE782B699, 0x55304632, 0x71B4EF90, 0x0EA9E82C, 0xDBB930F1, 0xFAD06B8B,
			0xBE0CAE40, 0x8BD14410, 0x76D2ADAC, 0x28ACAB7F};
	uint32_t B[]={
			0xC1099CB7, 0x07B385F3, 0xE7442C26, 0xCC8AD640, 0xEB6F56C7, 0x1EA81AA9, 0x73B9D314, 0x1DE85D08,
			0x48910A5A, 0x893B22DB, 0xC5A0DF44, 0xBBC4324E, 0x72D2F240, 0x75941D99, 0x6D8BDE82, 0xA1A7502B};
	uint32_t C[]={
			0xD9BF68D1, 0x58BAD750, 0x56028CB2, 0x8134F359, 0xB5D469D8, 0x941A8CC2, 0x418B2A6E, 0x04052780,
			0x7F07D787, 0x5194358F, 0x3C60D665, 0xBE97D79A, 0x950C3434, 0xAED9A06D, 0x2537DC8D, 0x7CDB5969};
	uint32_t M[16];

	if (thread < threads){

		uint32_t *Hash = &g_hash[thread<<4];
		
		*(uint2x4*)&M[ 0] = __ldg4((uint2x4*)&Hash[ 0]);
		*(uint2x4*)&M[ 8] = __ldg4((uint2x4*)&Hash[ 8]);


  // shabal
  sph_u32 A00 = A_init_512[0], A01 = A_init_512[1], A02 = A_init_512[2], A03 = A_init_512[3], A04 = A_init_512[4], A05 = A_init_512[5], A06 = A_init_512[6], A07 = A_init_512[7],
    A08 = A_init_512[8], A09 = A_init_512[9], A0A = A_init_512[10], A0B = A_init_512[11];
  sph_u32 B0 = B_init_512[0], B1 = B_init_512[1], B2 = B_init_512[2], B3 = B_init_512[3], B4 = B_init_512[4], B5 = B_init_512[5], B6 = B_init_512[6], B7 = B_init_512[7],
    B8 = B_init_512[8], B9 = B_init_512[9], BA = B_init_512[10], BB = B_init_512[11], BC = B_init_512[12], BD = B_init_512[13], BE = B_init_512[14], BF = B_init_512[15];
  sph_u32 C0 = C_init_512[0], C1 = C_init_512[1], C2 = C_init_512[2], C3 = C_init_512[3], C4 = C_init_512[4], C5 = C_init_512[5], C6 = C_init_512[6], C7 = C_init_512[7],
    C8 = C_init_512[8], C9 = C_init_512[9], CA = C_init_512[10], CB = C_init_512[11], CC = C_init_512[12], CD = C_init_512[13], CE = C_init_512[14], CF = C_init_512[15];
  sph_u32 M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, MA, MB, MC, MD, ME, MF;
  sph_u32 Wlow = 1, Whigh = 0;

  M0 = M[0];
  M1 = M[1];
  M2 = M[2];
  M3 = M[3];
  M4 = M[4];
  M5 = M[5];
  M6 = M[6];
  M7 = M[7];
  M8 = M[8];
  M9 = M[9];
  MA = M[10];
  MB = M[11];
  MC = M[12];
  MD = M[13];
  ME = M[14];
  MF = M[15];


  INPUT_BLOCK_ADD;
  XOR_W;
  APPLY_P;
  INPUT_BLOCK_SUB;
  SWAP_BC;
  INCR_W;

  M0 = M1 = M2 = M3 = M4 = M5 = M6 = M7 = M8 = M9 = MA = MB = MC = MD = ME = MF = 0;

  INPUT_BLOCK_ADD;
  XOR_W;
  APPLY_P;
  INPUT_BLOCK_SUB;
  SWAP_BC;
  INCR_W;

  M0 = 0x80;
  M1 = M2 = M3 = M4 = M5 = M6 = M7 = M8 = M9 = MA = MB = MC = MD = ME = MF = 0;

  INPUT_BLOCK_ADD;
  XOR_W;
  APPLY_P;

  for (unsigned i = 0; i < 3; i ++)
  {
    SWAP_BC;
    XOR_W;
    APPLY_P;
  }

  B[0] = B0;
  B[1] = B1;
  B[2] = B2;
  B[3] = B3;
  B[4] = B4;
  B[5] = B5;
  B[6] = B6;
  B[7] = B7;
  B[8] = B8;
  B[9] = B9;
  B[10] = BA;
  B[11] = BB;
  B[12] = BC;
  B[13] = BD;
  B[14] = BE;
  B[15] = BF;


		*(uint2x4*)&Hash[ 0] = *(uint2x4*)&B[ 0];
		*(uint2x4*)&Hash[ 8] = *(uint2x4*)&B[ 8];
	}
}

__host__ void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	const uint32_t threadsperblock = TPB_SHABAL;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x14_shabal512_gpu_hash_64<<<grid, block>>>(threads, d_hash);
}

__global__ __launch_bounds__(512,2)
void x14_shabal512_gpu_hash_64_final(uint32_t threads,const uint32_t* __restrict__ g_hash,uint32_t* resNonce, const uint64_t target){
/*
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint32_t A[]={
			0x20728DFD, 0x46C0BD53, 0xE782B699, 0x55304632, 0x71B4EF90, 0x0EA9E82C, 0xDBB930F1, 0xFAD06B8B,
			0xBE0CAE40, 0x8BD14410, 0x76D2ADAC, 0x28ACAB7F};
	uint32_t B[]={
			0xC1099CB7, 0x07B385F3, 0xE7442C26, 0xCC8AD640, 0xEB6F56C7, 0x1EA81AA9, 0x73B9D314, 0x1DE85D08,
			0x48910A5A, 0x893B22DB, 0xC5A0DF44, 0xBBC4324E, 0x72D2F240, 0x75941D99, 0x6D8BDE82, 0xA1A7502B};
	uint32_t C[]={
			0xD9BF68D1, 0x58BAD750, 0x56028CB2, 0x8134F359, 0xB5D469D8, 0x941A8CC2, 0x418B2A6E, 0x04052780,
			0x7F07D787, 0x5194358F, 0x3C60D665, 0xBE97D79A, 0x950C3434, 0xAED9A06D, 0x2537DC8D, 0x7CDB5969};
	uint32_t M[16];

	if (thread < threads){

		const uint32_t *Hash = &g_hash[thread<<4];
		
		*(uint2x4*)&M[ 0] = __ldg4((uint2x4*)&Hash[ 0]);
		*(uint2x4*)&M[ 8] = __ldg4((uint2x4*)&Hash[ 8]);

		*(uint16*)&B[ 0]+= *(uint16*)&M[ 0];
		A[ 0] ^= 1;
		ROTATE(B);
		PERM_STEP_0(A,B,C,M);
		PERM_STEP_1(A,B,C,M);
		PERM_STEP_2(A,B,C,M);
		ADD_BLOCK(A,C);
		*(uint16*)&C[ 0]-= *(uint16*)&M[ 0];
//		SWAP_BC;
		
		M[ 0] = 0x80;
		M[ 1] = M[ 2] = M[ 3] = M[ 4] = M[ 5] = M[ 6] = M[ 7] = M[ 8] = M[ 9] = M[10] = M[11] = M[12] = M[13] = M[14] = M[15] = 0;
		C[ 0]+= M[ 0];
		A[ 0]^= 0x02;
		ROTATE(C);
		PERM_STEP_0(A,C,B,M);
		PERM_STEP_1(A,C,B,M);
		PERM_STEP_2(A,C,B,M);
		ADD_BLOCK(A,B);
		A[ 0] ^= 0x02;
		ROTATE(B);
		PERM_STEP_0(A,B,C,M);
		PERM_STEP_1(A,B,C,M);
		PERM_STEP_2(A,B,C,M);
		ADD_BLOCK(A,C);
		A[ 0] ^= 0x02;
		ROTATE(C);
		PERM_STEP_0(A,C,B,M);
		PERM_STEP_1(A,C,B,M);
		PERM_STEP_2(A,C,B,M);
		ADD_BLOCK(A,B);
		A[ 0] ^= 0x02;
		ROTATE(B);
		PERM_STEP_0(A,B,C,M);
		PERM_STEP_1(A,B,C,M);
//		PERM_STEP_2(A,B,C,M);
		PERM_ELT(A[ 8], A[ 7], B[ 0], B[13], B[ 9], B[ 6], C[ 8], M[ 0]); PERM_ELT(A[ 9], A[ 8], B[ 1], B[14], B[10], B[ 7], C[ 7], M[ 1]);
		PERM_ELT(A[10], A[ 9], B[ 2], B[15], B[11], B[ 8], C[ 6], M[ 2]); PERM_ELT(A[11], A[10], B[ 3], B[ 0], B[12], B[ 9], C[ 5], M[ 3]);
		PERM_ELT(A[ 0], A[11], B[ 4], B[ 1], B[13], B[10], C[ 4], M[ 4]); PERM_ELT(A[ 1], A[ 0], B[ 5], B[ 2], B[14], B[11], C[ 3], M[ 5]);
		PERM_ELT(A[ 2], A[ 1], B[ 6], B[ 3], B[15], B[12], C[ 2], M[ 6]); PERM_ELT(A[ 3], A[ 2], B[ 7], B[ 4], B[ 0], B[13], C[ 1], M[ 7]);
		
		if(*(uint64_t*)&B[ 6] <= target){
			uint32_t tmp = atomicExch(&resNonce[0], thread);
			if (tmp != UINT32_MAX)
				resNonce[1] = tmp;		
		}
	}
*/
}

__host__ void x14_shabal512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target){

	const uint32_t threadsperblock = 512;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x14_shabal512_gpu_hash_64_final<<<grid, block>>>(threads, d_hash, d_resNonce, target);
}
