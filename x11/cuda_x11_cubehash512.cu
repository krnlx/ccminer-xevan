/*
	Based on Tanguy Pruvot's repo
	Provos Alexis - 2016
*/

#include "cuda_helper.h"
#include "cuda_vectors.h"

#define SWAP(a,b) { uint32_t u = a; a = b; b = u; }
//#define SWAP xchg

//#define SHUFFLE

#if defined(SHUFFLE)

#define TPB 1024

__device__ __forceinline__
void rrounds(uint32_t *x){
	#pragma unroll 16
	for (int r = 0; r < 16; ++r) {
		/* "add x_0jklm into x_1jklm modulo 2^32 rotate x_0jklm upwards by 7 bits" */
		x[ 8] += x[ 0]; x[ 9] += x[ 1];
		x[10] += x[ 2]; x[11] += x[ 3];
		x[12] += x[ 4]; x[13] += x[ 5];
		x[14] += x[ 6]; x[15] += x[ 7];
		x[ 0] = ROTL32(x[ 0], 7);
		x[ 1] = ROTL32(x[ 1], 7);
		x[ 2] = ROTL32(x[ 2], 7);
		x[ 3] = ROTL32(x[ 3], 7);
		x[ 4] = ROTL32(x[ 4], 7);
		x[ 5] = ROTL32(x[ 5], 7);
		x[ 6] = ROTL32(x[ 6], 7);
		x[ 7] = ROTL32(x[ 7], 7);
		/* "swap x_00klm with x_01klm" */
		SWAP(x[ 0], x[ 4]); SWAP(x[ 1], x[ 5]);
		SWAP(x[ 2], x[ 6]); SWAP(x[ 3], x[ 7]);
		
		x[ 0] ^= x[ 8]; x[ 4] ^= x[12];
		x[ 1] ^= x[ 9]; x[ 5] ^= x[13];
		x[ 2] ^= x[10]; x[ 6] ^= x[14];
		x[ 3] ^= x[11]; x[ 7] ^= x[15];
		/* "swap x_1jk0m with x_1jk1m" */
		SWAP(x[ 8],x[10]); SWAP(x[ 9],x[11]);
		SWAP(x[12],x[14]); SWAP(x[13],x[15]);
		/* "add x_0jklm into x_1jklm modulo 2^32 rotate x_0jklm upwards by 11 bits" */
		x[ 8] += x[ 0]; x[ 9] += x[ 1];
		x[10] += x[ 2]; x[11] += x[ 3];
		x[12] += x[ 4]; x[13] += x[ 5];
		x[14] += x[ 6]; x[15] += x[ 7];
		x[ 0] = ROTL32(x[ 0],11);
		x[ 1] = ROTL32(x[ 1],11);
		x[ 2] = ROTL32(x[ 2],11);
		x[ 3] = ROTL32(x[ 3],11);
		x[ 4] = ROTL32(x[ 4],11);
		x[ 5] = ROTL32(x[ 5],11);
		x[ 6] = ROTL32(x[ 6],11);
		x[ 7] = ROTL32(x[ 7],11);
		/* "swap x_0j0lm with x_0j1lm" */
		x[ 0] = __shfl(x[ 0],threadIdx.x^1);
		x[ 1] = __shfl(x[ 1],threadIdx.x^1);
		x[ 2] = __shfl(x[ 2],threadIdx.x^1);
		x[ 3] = __shfl(x[ 3],threadIdx.x^1);
		x[ 4] = __shfl(x[ 4],threadIdx.x^1);
		x[ 5] = __shfl(x[ 5],threadIdx.x^1);
		x[ 6] = __shfl(x[ 6],threadIdx.x^1);
		x[ 7] = __shfl(x[ 7],threadIdx.x^1);
	
		x[ 0] ^= x[ 8]; x[ 1] ^= x[ 9];
		x[ 2] ^= x[10]; x[ 3] ^= x[11];
		x[ 4] ^= x[12]; x[ 5] ^= x[13];
		x[ 6] ^= x[14]; x[ 7] ^= x[15];
		/* "swap x_1jkl0 with x_1jkl1" */
		SWAP(x[ 8],x[ 9]); SWAP(x[10],x[11]);
		SWAP(x[12],x[13]); SWAP(x[14],x[15]);
	}
}
__global__ __launch_bounds__(TPB, 1)
void x11_cubehash512_gpu_hash_64(uint32_t threads, uint64_t *g_hash){
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x)>>1;

	const uint32_t even = (threadIdx.x & 1);

	if (thread < threads){
		uint32_t *Hash = (uint32_t*)&g_hash[8 * thread + 2*even];

		uint32_t x[16];

		if(even==0){
			x[ 0] = 0x2AEA2A61;	x[ 1] = 0x50F494D4;	x[ 2] = 0x2D538B8B;	x[ 3] = 0x4167D83E;
			x[ 4] = 0x4D42C787;	x[ 5] = 0xA647A8B3;	x[ 6] = 0x97CF0BEF;	x[ 7] = 0x825B4537;
			x[ 8] = 0xFCD398D9;	x[ 9] = 0x148FE485;	x[10] = 0x1B017BEF;	x[11] = 0xB6444532;
			x[12] = 0xD65C8A2B;	x[13] = 0xA5A70E75;	x[14] = 0xB1C62456;	x[15] = 0xBC796576;
		}else{
			x[ 0] = 0x3FEE2313;	x[ 1] = 0xC701CF8C;	x[ 2] = 0xCC39968E;	x[ 3] = 0x50AC5695;
			x[ 4] = 0xEEF864D2;	x[ 5] = 0xF22090C4;	x[ 6] = 0xD0E5CD33;	x[ 7] = 0xA23911AE;
			x[ 8] = 0x6A536159;	x[ 9] = 0x2FF5781C;	x[10] = 0x91FA7934;	x[11] = 0x0DBADEA9;
			x[12] = 0x1921C8F7;	x[13] = 0xE7989AF1; 	x[14] = 0x7795D246;	x[15] = 0xD43E3B44;
		}
		*(uint4*)&x[ 0]^= __ldg((uint4*)&Hash[ 0]);
		rrounds(x);

		*(uint4*)&x[ 0]^= __ldg((uint4*)&Hash[ 8]);

		rrounds(x);
		
		if(!even)
			x[ 0] ^= 0x80;

		rrounds(x);
		/* "the integer 1 is xored into the last state word x_11111" */
		if(even)
			x[15] ^= 1;
	
		#pragma unroll 10
		for (int i = 0; i < 10; ++i)
			rrounds(x);
	
		*(uint4*)&Hash[ 0] = *(uint4*)&x[ 0];
		*(uint4*)&Hash[ 8] = *(uint4*)&x[ 4];
//		g_hash[thread + (2*even+0) * threads]	= *(uint2*)&x[ 0];
//		g_hash[thread + (2*even+1) * threads]	= *(uint2*)&x[ 2];
	}
}
__host__
void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash){

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((2*threads + TPB-1)/TPB);
    dim3 block(TPB);

    x11_cubehash512_gpu_hash_64<<<grid, block>>>(threads, (uint64_t*)d_hash);

}

#else

#define TPB 512

__device__ __forceinline__
static void rrounds(uint32_t *x){
//	#pragma unroll 2
	for (int r = 0; r < 16; r++) {
		/* "add x_0jklm into x_1jklmn modulo 2^32 rotate x_0jklm upwards by 7 bits" */
		x[16] = x[16] + x[ 0]; x[ 0] = ROTL32(x[ 0], 7);x[17] = x[17] + x[ 1];x[ 1] = ROTL32(x[ 1], 7);
		x[18] = x[18] + x[ 2]; x[ 2] = ROTL32(x[ 2], 7);x[19] = x[19] + x[ 3];x[ 3] = ROTL32(x[ 3], 7);
		x[20] = x[20] + x[ 4]; x[ 4] = ROTL32(x[ 4], 7);x[21] = x[21] + x[ 5];x[ 5] = ROTL32(x[ 5], 7);
		x[22] = x[22] + x[ 6]; x[ 6] = ROTL32(x[ 6], 7);x[23] = x[23] + x[ 7];x[ 7] = ROTL32(x[ 7], 7);
		x[24] = x[24] + x[ 8]; x[ 8] = ROTL32(x[ 8], 7);x[25] = x[25] + x[ 9];x[ 9] = ROTL32(x[ 9], 7);
		x[26] = x[26] + x[10]; x[10] = ROTL32(x[10], 7);x[27] = x[27] + x[11];x[11] = ROTL32(x[11], 7);
		x[28] = x[28] + x[12]; x[12] = ROTL32(x[12], 7);x[29] = x[29] + x[13];x[13] = ROTL32(x[13], 7);
		x[30] = x[30] + x[14]; x[14] = ROTL32(x[14], 7);x[31] = x[31] + x[15];x[15] = ROTL32(x[15], 7);
		/* "swap x_00klm with x_01klm" */
		SWAP(x[ 0], x[ 8]);x[ 0] ^= x[16];x[ 8] ^= x[24];SWAP(x[ 1], x[ 9]);x[ 1] ^= x[17];x[ 9] ^= x[25];
		SWAP(x[ 2], x[10]);x[ 2] ^= x[18];x[10] ^= x[26];SWAP(x[ 3], x[11]);x[ 3] ^= x[19];x[11] ^= x[27];
		SWAP(x[ 4], x[12]);x[ 4] ^= x[20];x[12] ^= x[28];SWAP(x[ 5], x[13]);x[ 5] ^= x[21];x[13] ^= x[29];
		SWAP(x[ 6], x[14]);x[ 6] ^= x[22];x[14] ^= x[30];SWAP(x[ 7], x[15]);x[ 7] ^= x[23];x[15] ^= x[31];
		/* "swap x_1jk0m with x_1jk1m" */
		SWAP(x[16], x[18]); SWAP(x[17], x[19]); SWAP(x[20], x[22]); SWAP(x[21], x[23]);SWAP(x[24], x[26]); SWAP(x[25], x[27]); SWAP(x[28], x[30]); SWAP(x[29], x[31]);
		/* "add x_0jklm into x_1jklm modulo 2^32 rotate x_0jklm upwards by 11 bits" */
		x[16] = x[16] + x[ 0]; x[ 0] = ROTL32(x[ 0],11);x[17] = x[17] + x[ 1];x[ 1] = ROTL32(x[ 1],11);
		x[18] = x[18] + x[ 2]; x[ 2] = ROTL32(x[ 2],11);x[19] = x[19] + x[ 3];x[ 3] = ROTL32(x[ 3],11);
		x[20] = x[20] + x[ 4]; x[ 4] = ROTL32(x[ 4],11);x[21] = x[21] + x[ 5];x[ 5] = ROTL32(x[ 5],11);
		x[22] = x[22] + x[ 6]; x[ 6] = ROTL32(x[ 6],11);x[23] = x[23] + x[ 7];x[ 7] = ROTL32(x[ 7],11);
		x[24] = x[24] + x[ 8]; x[ 8] = ROTL32(x[ 8],11);x[25] = x[25] + x[ 9];x[ 9] = ROTL32(x[ 9],11);
		x[26] = x[26] + x[10]; x[10] = ROTL32(x[10],11);x[27] = x[27] + x[11];x[11] = ROTL32(x[11],11);
		x[28] = x[28] + x[12]; x[12] = ROTL32(x[12],11);x[29] = x[29] + x[13];x[13] = ROTL32(x[13],11);
		x[30] = x[30] + x[14]; x[14] = ROTL32(x[14],11);x[31] = x[31] + x[15];x[15] = ROTL32(x[15],11);
		/* "swap x_0j0lm with x_0j1lm" */
		SWAP(x[ 0], x[ 4]); x[ 0] ^= x[16]; x[ 4] ^= x[20]; SWAP(x[ 1], x[ 5]); x[ 1] ^= x[17]; x[ 5] ^= x[21];
		SWAP(x[ 2], x[ 6]); x[ 2] ^= x[18]; x[ 6] ^= x[22]; SWAP(x[ 3], x[ 7]); x[ 3] ^= x[19]; x[ 7] ^= x[23];
		SWAP(x[ 8], x[12]); x[ 8] ^= x[24]; x[12] ^= x[28]; SWAP(x[ 9], x[13]); x[ 9] ^= x[25]; x[13] ^= x[29];
		SWAP(x[10], x[14]); x[10] ^= x[26]; x[14] ^= x[30]; SWAP(x[11], x[15]); x[11] ^= x[27]; x[15] ^= x[31];
		/* "swap x_1jkl0 with x_1jkl1" */
		SWAP(x[16], x[17]); SWAP(x[18], x[19]); SWAP(x[20], x[21]); SWAP(x[22], x[23]);SWAP(x[24], x[25]); SWAP(x[26], x[27]); SWAP(x[28], x[29]); SWAP(x[30], x[31]);
	}
}


/***************************************************/
// GPU Hash Function
__global__ __launch_bounds__(TPB,2)
void x11_cubehash512_gpu_hash_64(uint32_t threads, uint64_t *g_hash){

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){

		uint32_t *Hash = (uint32_t*)&g_hash[8 * thread];

		uint32_t x[32] = {
			0x2AEA2A61, 0x50F494D4, 0x2D538B8B, 0x4167D83E,
			0x3FEE2313, 0xC701CF8C, 0xCC39968E, 0x50AC5695,
			0x4D42C787, 0xA647A8B3, 0x97CF0BEF, 0x825B4537,
			0xEEF864D2, 0xF22090C4, 0xD0E5CD33, 0xA23911AE,
			0xFCD398D9, 0x148FE485, 0x1B017BEF, 0xB6444532,
			0x6A536159, 0x2FF5781C, 0x91FA7934, 0x0DBADEA9,
			0xD65C8A2B, 0xA5A70E75, 0xB1C62456, 0xBC796576,
			0x1921C8F7, 0xE7989AF1, 0x7795D246, 0xD43E3B44
		};
	
		// erste Hälfte des Hashes (32 bytes)
		//Update32(x, (const BitSequence*)Hash);
		*(uint2x4*)&x[ 0] ^= __ldg4((uint2x4*)&Hash[0]);

		rrounds(x);

		// zweite Hälfte des Hashes (32 bytes)
	//        Update32(x, (const BitSequence*)(Hash+8));
		*(uint2x4*)&x[ 0] ^= __ldg4((uint2x4*)&Hash[8]);
		
		rrounds(x);
		rrounds(x);
		rrounds(x);

		// Padding Block
		x[ 0] ^= 0x80;
		rrounds(x);
	
	//	Final(x, (BitSequence*)Hash);
		x[31] ^= 1;

		/* "the state is then transformed invertibly through 10r identical rounds" */
		#pragma unroll 10
		for (int i = 0;i < 10;++i)
			rrounds(x);

		/* "output the first h/8 bytes of the state" */
		*(uint2x4*)&Hash[ 0] = *(uint2x4*)&x[ 0];
		*(uint2x4*)&Hash[ 8] = *(uint2x4*)&x[ 8];
	}
}


__host__
void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash){

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + TPB-1)/TPB);
    dim3 block(TPB);

    x11_cubehash512_gpu_hash_64<<<grid, block>>>(threads, (uint64_t*)d_hash);

}

#define sph_u32 uint32_t
#define sph_u64 uint64_t


#define AESx(x)   SPH_C32(x)
#define AES0      AES0_S
#define AES1      AES1_S
#define AES2      AES2_S
#define AES3      AES3_S

#define AES_ROUND_LE(X0, X1, X2, X3, K0, K1, K2, K3, Y0, Y1, Y2, Y3)   do { \
    (Y0) = AES0[__byte_perm(X0, 0, 0x4440)] \
      ^ AES1[__byte_perm(X1, 0, 0x4441)] \
      ^ AES2[__byte_perm(X2, 0, 0x4442)] \
      ^ __ldg(&AES3_C[__byte_perm(X3, 0, 0x4443)]) ^ (K0); \
    (Y1) = __ldg(&AES0_C[__byte_perm(X1, 0, 0x4440)]) \
      ^ AES1[__byte_perm(X2, 0, 0x4441)] \
      ^ AES2[__byte_perm(X3, 0, 0x4442)] \
      ^ __ldg(&AES3_C[__byte_perm(X0, 0, 0x4443)]) ^ (K1); \
    (Y2) = AES0[__byte_perm(X2, 0, 0x4440)] \
      ^ AES1[__byte_perm(X3, 0, 0x4441)] \
      ^ AES2[__byte_perm(X0, 0, 0x4442)] \
      ^ __ldg(&AES3_C[__byte_perm(X1, 0, 0x4443)]) ^ (K2); \
    (Y3) = AES0[__byte_perm(X3, 0, 0x4440)] \
      ^ AES1[__byte_perm(X0, 0, 0x4441)] \
      ^ AES2[__byte_perm(X1, 0, 0x4442)] \
      ^ __ldg(&AES3_C[__byte_perm(X2, 0, 0x4443)]) ^ (K3); \
  } while (0)

#define AES_ROUND_NOKEY_LE(X0, X1, X2, X3, Y0, Y1, Y2, Y3) \
  AES_ROUND_LE(X0, X1, X2, X3, 0, 0, 0, 0, Y0, Y1, Y2, Y3)


__constant__ static const sph_u32 AES0_C[256] = {
  AESx(0xA56363C6), AESx(0x847C7CF8), AESx(0x997777EE), AESx(0x8D7B7BF6),
  AESx(0x0DF2F2FF), AESx(0xBD6B6BD6), AESx(0xB16F6FDE), AESx(0x54C5C591),
  AESx(0x50303060), AESx(0x03010102), AESx(0xA96767CE), AESx(0x7D2B2B56),
  AESx(0x19FEFEE7), AESx(0x62D7D7B5), AESx(0xE6ABAB4D), AESx(0x9A7676EC),
  AESx(0x45CACA8F), AESx(0x9D82821F), AESx(0x40C9C989), AESx(0x877D7DFA),
  AESx(0x15FAFAEF), AESx(0xEB5959B2), AESx(0xC947478E), AESx(0x0BF0F0FB),
  AESx(0xECADAD41), AESx(0x67D4D4B3), AESx(0xFDA2A25F), AESx(0xEAAFAF45),
  AESx(0xBF9C9C23), AESx(0xF7A4A453), AESx(0x967272E4), AESx(0x5BC0C09B),
  AESx(0xC2B7B775), AESx(0x1CFDFDE1), AESx(0xAE93933D), AESx(0x6A26264C),
  AESx(0x5A36366C), AESx(0x413F3F7E), AESx(0x02F7F7F5), AESx(0x4FCCCC83),
  AESx(0x5C343468), AESx(0xF4A5A551), AESx(0x34E5E5D1), AESx(0x08F1F1F9),
  AESx(0x937171E2), AESx(0x73D8D8AB), AESx(0x53313162), AESx(0x3F15152A),
  AESx(0x0C040408), AESx(0x52C7C795), AESx(0x65232346), AESx(0x5EC3C39D),
  AESx(0x28181830), AESx(0xA1969637), AESx(0x0F05050A), AESx(0xB59A9A2F),
  AESx(0x0907070E), AESx(0x36121224), AESx(0x9B80801B), AESx(0x3DE2E2DF),
  AESx(0x26EBEBCD), AESx(0x6927274E), AESx(0xCDB2B27F), AESx(0x9F7575EA),
  AESx(0x1B090912), AESx(0x9E83831D), AESx(0x742C2C58), AESx(0x2E1A1A34),
  AESx(0x2D1B1B36), AESx(0xB26E6EDC), AESx(0xEE5A5AB4), AESx(0xFBA0A05B),
  AESx(0xF65252A4), AESx(0x4D3B3B76), AESx(0x61D6D6B7), AESx(0xCEB3B37D),
  AESx(0x7B292952), AESx(0x3EE3E3DD), AESx(0x712F2F5E), AESx(0x97848413),
  AESx(0xF55353A6), AESx(0x68D1D1B9), AESx(0x00000000), AESx(0x2CEDEDC1),
  AESx(0x60202040), AESx(0x1FFCFCE3), AESx(0xC8B1B179), AESx(0xED5B5BB6),
  AESx(0xBE6A6AD4), AESx(0x46CBCB8D), AESx(0xD9BEBE67), AESx(0x4B393972),
  AESx(0xDE4A4A94), AESx(0xD44C4C98), AESx(0xE85858B0), AESx(0x4ACFCF85),
  AESx(0x6BD0D0BB), AESx(0x2AEFEFC5), AESx(0xE5AAAA4F), AESx(0x16FBFBED),
  AESx(0xC5434386), AESx(0xD74D4D9A), AESx(0x55333366), AESx(0x94858511),
  AESx(0xCF45458A), AESx(0x10F9F9E9), AESx(0x06020204), AESx(0x817F7FFE),
  AESx(0xF05050A0), AESx(0x443C3C78), AESx(0xBA9F9F25), AESx(0xE3A8A84B),
  AESx(0xF35151A2), AESx(0xFEA3A35D), AESx(0xC0404080), AESx(0x8A8F8F05),
  AESx(0xAD92923F), AESx(0xBC9D9D21), AESx(0x48383870), AESx(0x04F5F5F1),
  AESx(0xDFBCBC63), AESx(0xC1B6B677), AESx(0x75DADAAF), AESx(0x63212142),
  AESx(0x30101020), AESx(0x1AFFFFE5), AESx(0x0EF3F3FD), AESx(0x6DD2D2BF),
  AESx(0x4CCDCD81), AESx(0x140C0C18), AESx(0x35131326), AESx(0x2FECECC3),
  AESx(0xE15F5FBE), AESx(0xA2979735), AESx(0xCC444488), AESx(0x3917172E),
  AESx(0x57C4C493), AESx(0xF2A7A755), AESx(0x827E7EFC), AESx(0x473D3D7A),
  AESx(0xAC6464C8), AESx(0xE75D5DBA), AESx(0x2B191932), AESx(0x957373E6),
  AESx(0xA06060C0), AESx(0x98818119), AESx(0xD14F4F9E), AESx(0x7FDCDCA3),
  AESx(0x66222244), AESx(0x7E2A2A54), AESx(0xAB90903B), AESx(0x8388880B),
  AESx(0xCA46468C), AESx(0x29EEEEC7), AESx(0xD3B8B86B), AESx(0x3C141428),
  AESx(0x79DEDEA7), AESx(0xE25E5EBC), AESx(0x1D0B0B16), AESx(0x76DBDBAD),
  AESx(0x3BE0E0DB), AESx(0x56323264), AESx(0x4E3A3A74), AESx(0x1E0A0A14),
  AESx(0xDB494992), AESx(0x0A06060C), AESx(0x6C242448), AESx(0xE45C5CB8),
  AESx(0x5DC2C29F), AESx(0x6ED3D3BD), AESx(0xEFACAC43), AESx(0xA66262C4),
  AESx(0xA8919139), AESx(0xA4959531), AESx(0x37E4E4D3), AESx(0x8B7979F2),
  AESx(0x32E7E7D5), AESx(0x43C8C88B), AESx(0x5937376E), AESx(0xB76D6DDA),
  AESx(0x8C8D8D01), AESx(0x64D5D5B1), AESx(0xD24E4E9C), AESx(0xE0A9A949),
  AESx(0xB46C6CD8), AESx(0xFA5656AC), AESx(0x07F4F4F3), AESx(0x25EAEACF),
  AESx(0xAF6565CA), AESx(0x8E7A7AF4), AESx(0xE9AEAE47), AESx(0x18080810),
  AESx(0xD5BABA6F), AESx(0x887878F0), AESx(0x6F25254A), AESx(0x722E2E5C),
  AESx(0x241C1C38), AESx(0xF1A6A657), AESx(0xC7B4B473), AESx(0x51C6C697),
  AESx(0x23E8E8CB), AESx(0x7CDDDDA1), AESx(0x9C7474E8), AESx(0x211F1F3E),
  AESx(0xDD4B4B96), AESx(0xDCBDBD61), AESx(0x868B8B0D), AESx(0x858A8A0F),
  AESx(0x907070E0), AESx(0x423E3E7C), AESx(0xC4B5B571), AESx(0xAA6666CC),
  AESx(0xD8484890), AESx(0x05030306), AESx(0x01F6F6F7), AESx(0x120E0E1C),
  AESx(0xA36161C2), AESx(0x5F35356A), AESx(0xF95757AE), AESx(0xD0B9B969),
  AESx(0x91868617), AESx(0x58C1C199), AESx(0x271D1D3A), AESx(0xB99E9E27),
  AESx(0x38E1E1D9), AESx(0x13F8F8EB), AESx(0xB398982B), AESx(0x33111122),
  AESx(0xBB6969D2), AESx(0x70D9D9A9), AESx(0x898E8E07), AESx(0xA7949433),
  AESx(0xB69B9B2D), AESx(0x221E1E3C), AESx(0x92878715), AESx(0x20E9E9C9),
  AESx(0x49CECE87), AESx(0xFF5555AA), AESx(0x78282850), AESx(0x7ADFDFA5),
  AESx(0x8F8C8C03), AESx(0xF8A1A159), AESx(0x80898909), AESx(0x170D0D1A),
  AESx(0xDABFBF65), AESx(0x31E6E6D7), AESx(0xC6424284), AESx(0xB86868D0),
  AESx(0xC3414182), AESx(0xB0999929), AESx(0x772D2D5A), AESx(0x110F0F1E),
  AESx(0xCBB0B07B), AESx(0xFC5454A8), AESx(0xD6BBBB6D), AESx(0x3A16162C)
};

__constant__ static const sph_u32 AES1_C[256] = {
  AESx(0x6363C6A5), AESx(0x7C7CF884), AESx(0x7777EE99), AESx(0x7B7BF68D),
  AESx(0xF2F2FF0D), AESx(0x6B6BD6BD), AESx(0x6F6FDEB1), AESx(0xC5C59154),
  AESx(0x30306050), AESx(0x01010203), AESx(0x6767CEA9), AESx(0x2B2B567D),
  AESx(0xFEFEE719), AESx(0xD7D7B562), AESx(0xABAB4DE6), AESx(0x7676EC9A),
  AESx(0xCACA8F45), AESx(0x82821F9D), AESx(0xC9C98940), AESx(0x7D7DFA87),
  AESx(0xFAFAEF15), AESx(0x5959B2EB), AESx(0x47478EC9), AESx(0xF0F0FB0B),
  AESx(0xADAD41EC), AESx(0xD4D4B367), AESx(0xA2A25FFD), AESx(0xAFAF45EA),
  AESx(0x9C9C23BF), AESx(0xA4A453F7), AESx(0x7272E496), AESx(0xC0C09B5B),
  AESx(0xB7B775C2), AESx(0xFDFDE11C), AESx(0x93933DAE), AESx(0x26264C6A),
  AESx(0x36366C5A), AESx(0x3F3F7E41), AESx(0xF7F7F502), AESx(0xCCCC834F),
  AESx(0x3434685C), AESx(0xA5A551F4), AESx(0xE5E5D134), AESx(0xF1F1F908),
  AESx(0x7171E293), AESx(0xD8D8AB73), AESx(0x31316253), AESx(0x15152A3F),
  AESx(0x0404080C), AESx(0xC7C79552), AESx(0x23234665), AESx(0xC3C39D5E),
  AESx(0x18183028), AESx(0x969637A1), AESx(0x05050A0F), AESx(0x9A9A2FB5),
  AESx(0x07070E09), AESx(0x12122436), AESx(0x80801B9B), AESx(0xE2E2DF3D),
  AESx(0xEBEBCD26), AESx(0x27274E69), AESx(0xB2B27FCD), AESx(0x7575EA9F),
  AESx(0x0909121B), AESx(0x83831D9E), AESx(0x2C2C5874), AESx(0x1A1A342E),
  AESx(0x1B1B362D), AESx(0x6E6EDCB2), AESx(0x5A5AB4EE), AESx(0xA0A05BFB),
  AESx(0x5252A4F6), AESx(0x3B3B764D), AESx(0xD6D6B761), AESx(0xB3B37DCE),
  AESx(0x2929527B), AESx(0xE3E3DD3E), AESx(0x2F2F5E71), AESx(0x84841397),
  AESx(0x5353A6F5), AESx(0xD1D1B968), AESx(0x00000000), AESx(0xEDEDC12C),
  AESx(0x20204060), AESx(0xFCFCE31F), AESx(0xB1B179C8), AESx(0x5B5BB6ED),
  AESx(0x6A6AD4BE), AESx(0xCBCB8D46), AESx(0xBEBE67D9), AESx(0x3939724B),
  AESx(0x4A4A94DE), AESx(0x4C4C98D4), AESx(0x5858B0E8), AESx(0xCFCF854A),
  AESx(0xD0D0BB6B), AESx(0xEFEFC52A), AESx(0xAAAA4FE5), AESx(0xFBFBED16),
  AESx(0x434386C5), AESx(0x4D4D9AD7), AESx(0x33336655), AESx(0x85851194),
  AESx(0x45458ACF), AESx(0xF9F9E910), AESx(0x02020406), AESx(0x7F7FFE81),
  AESx(0x5050A0F0), AESx(0x3C3C7844), AESx(0x9F9F25BA), AESx(0xA8A84BE3),
  AESx(0x5151A2F3), AESx(0xA3A35DFE), AESx(0x404080C0), AESx(0x8F8F058A),
  AESx(0x92923FAD), AESx(0x9D9D21BC), AESx(0x38387048), AESx(0xF5F5F104),
  AESx(0xBCBC63DF), AESx(0xB6B677C1), AESx(0xDADAAF75), AESx(0x21214263),
  AESx(0x10102030), AESx(0xFFFFE51A), AESx(0xF3F3FD0E), AESx(0xD2D2BF6D),
  AESx(0xCDCD814C), AESx(0x0C0C1814), AESx(0x13132635), AESx(0xECECC32F),
  AESx(0x5F5FBEE1), AESx(0x979735A2), AESx(0x444488CC), AESx(0x17172E39),
  AESx(0xC4C49357), AESx(0xA7A755F2), AESx(0x7E7EFC82), AESx(0x3D3D7A47),
  AESx(0x6464C8AC), AESx(0x5D5DBAE7), AESx(0x1919322B), AESx(0x7373E695),
  AESx(0x6060C0A0), AESx(0x81811998), AESx(0x4F4F9ED1), AESx(0xDCDCA37F),
  AESx(0x22224466), AESx(0x2A2A547E), AESx(0x90903BAB), AESx(0x88880B83),
  AESx(0x46468CCA), AESx(0xEEEEC729), AESx(0xB8B86BD3), AESx(0x1414283C),
  AESx(0xDEDEA779), AESx(0x5E5EBCE2), AESx(0x0B0B161D), AESx(0xDBDBAD76),
  AESx(0xE0E0DB3B), AESx(0x32326456), AESx(0x3A3A744E), AESx(0x0A0A141E),
  AESx(0x494992DB), AESx(0x06060C0A), AESx(0x2424486C), AESx(0x5C5CB8E4),
  AESx(0xC2C29F5D), AESx(0xD3D3BD6E), AESx(0xACAC43EF), AESx(0x6262C4A6),
  AESx(0x919139A8), AESx(0x959531A4), AESx(0xE4E4D337), AESx(0x7979F28B),
  AESx(0xE7E7D532), AESx(0xC8C88B43), AESx(0x37376E59), AESx(0x6D6DDAB7),
  AESx(0x8D8D018C), AESx(0xD5D5B164), AESx(0x4E4E9CD2), AESx(0xA9A949E0),
  AESx(0x6C6CD8B4), AESx(0x5656ACFA), AESx(0xF4F4F307), AESx(0xEAEACF25),
  AESx(0x6565CAAF), AESx(0x7A7AF48E), AESx(0xAEAE47E9), AESx(0x08081018),
  AESx(0xBABA6FD5), AESx(0x7878F088), AESx(0x25254A6F), AESx(0x2E2E5C72),
  AESx(0x1C1C3824), AESx(0xA6A657F1), AESx(0xB4B473C7), AESx(0xC6C69751),
  AESx(0xE8E8CB23), AESx(0xDDDDA17C), AESx(0x7474E89C), AESx(0x1F1F3E21),
  AESx(0x4B4B96DD), AESx(0xBDBD61DC), AESx(0x8B8B0D86), AESx(0x8A8A0F85),
  AESx(0x7070E090), AESx(0x3E3E7C42), AESx(0xB5B571C4), AESx(0x6666CCAA),
  AESx(0x484890D8), AESx(0x03030605), AESx(0xF6F6F701), AESx(0x0E0E1C12),
  AESx(0x6161C2A3), AESx(0x35356A5F), AESx(0x5757AEF9), AESx(0xB9B969D0),
  AESx(0x86861791), AESx(0xC1C19958), AESx(0x1D1D3A27), AESx(0x9E9E27B9),
  AESx(0xE1E1D938), AESx(0xF8F8EB13), AESx(0x98982BB3), AESx(0x11112233),
  AESx(0x6969D2BB), AESx(0xD9D9A970), AESx(0x8E8E0789), AESx(0x949433A7),
  AESx(0x9B9B2DB6), AESx(0x1E1E3C22), AESx(0x87871592), AESx(0xE9E9C920),
  AESx(0xCECE8749), AESx(0x5555AAFF), AESx(0x28285078), AESx(0xDFDFA57A),
  AESx(0x8C8C038F), AESx(0xA1A159F8), AESx(0x89890980), AESx(0x0D0D1A17),
  AESx(0xBFBF65DA), AESx(0xE6E6D731), AESx(0x424284C6), AESx(0x6868D0B8),
  AESx(0x414182C3), AESx(0x999929B0), AESx(0x2D2D5A77), AESx(0x0F0F1E11),
  AESx(0xB0B07BCB), AESx(0x5454A8FC), AESx(0xBBBB6DD6), AESx(0x16162C3A)
};

__constant__ static const sph_u32 AES2_C[256] = {
  AESx(0x63C6A563), AESx(0x7CF8847C), AESx(0x77EE9977), AESx(0x7BF68D7B),
  AESx(0xF2FF0DF2), AESx(0x6BD6BD6B), AESx(0x6FDEB16F), AESx(0xC59154C5),
  AESx(0x30605030), AESx(0x01020301), AESx(0x67CEA967), AESx(0x2B567D2B),
  AESx(0xFEE719FE), AESx(0xD7B562D7), AESx(0xAB4DE6AB), AESx(0x76EC9A76),
  AESx(0xCA8F45CA), AESx(0x821F9D82), AESx(0xC98940C9), AESx(0x7DFA877D),
  AESx(0xFAEF15FA), AESx(0x59B2EB59), AESx(0x478EC947), AESx(0xF0FB0BF0),
  AESx(0xAD41ECAD), AESx(0xD4B367D4), AESx(0xA25FFDA2), AESx(0xAF45EAAF),
  AESx(0x9C23BF9C), AESx(0xA453F7A4), AESx(0x72E49672), AESx(0xC09B5BC0),
  AESx(0xB775C2B7), AESx(0xFDE11CFD), AESx(0x933DAE93), AESx(0x264C6A26),
  AESx(0x366C5A36), AESx(0x3F7E413F), AESx(0xF7F502F7), AESx(0xCC834FCC),
  AESx(0x34685C34), AESx(0xA551F4A5), AESx(0xE5D134E5), AESx(0xF1F908F1),
  AESx(0x71E29371), AESx(0xD8AB73D8), AESx(0x31625331), AESx(0x152A3F15),
  AESx(0x04080C04), AESx(0xC79552C7), AESx(0x23466523), AESx(0xC39D5EC3),
  AESx(0x18302818), AESx(0x9637A196), AESx(0x050A0F05), AESx(0x9A2FB59A),
  AESx(0x070E0907), AESx(0x12243612), AESx(0x801B9B80), AESx(0xE2DF3DE2),
  AESx(0xEBCD26EB), AESx(0x274E6927), AESx(0xB27FCDB2), AESx(0x75EA9F75),
  AESx(0x09121B09), AESx(0x831D9E83), AESx(0x2C58742C), AESx(0x1A342E1A),
  AESx(0x1B362D1B), AESx(0x6EDCB26E), AESx(0x5AB4EE5A), AESx(0xA05BFBA0),
  AESx(0x52A4F652), AESx(0x3B764D3B), AESx(0xD6B761D6), AESx(0xB37DCEB3),
  AESx(0x29527B29), AESx(0xE3DD3EE3), AESx(0x2F5E712F), AESx(0x84139784),
  AESx(0x53A6F553), AESx(0xD1B968D1), AESx(0x00000000), AESx(0xEDC12CED),
  AESx(0x20406020), AESx(0xFCE31FFC), AESx(0xB179C8B1), AESx(0x5BB6ED5B),
  AESx(0x6AD4BE6A), AESx(0xCB8D46CB), AESx(0xBE67D9BE), AESx(0x39724B39),
  AESx(0x4A94DE4A), AESx(0x4C98D44C), AESx(0x58B0E858), AESx(0xCF854ACF),
  AESx(0xD0BB6BD0), AESx(0xEFC52AEF), AESx(0xAA4FE5AA), AESx(0xFBED16FB),
  AESx(0x4386C543), AESx(0x4D9AD74D), AESx(0x33665533), AESx(0x85119485),
  AESx(0x458ACF45), AESx(0xF9E910F9), AESx(0x02040602), AESx(0x7FFE817F),
  AESx(0x50A0F050), AESx(0x3C78443C), AESx(0x9F25BA9F), AESx(0xA84BE3A8),
  AESx(0x51A2F351), AESx(0xA35DFEA3), AESx(0x4080C040), AESx(0x8F058A8F),
  AESx(0x923FAD92), AESx(0x9D21BC9D), AESx(0x38704838), AESx(0xF5F104F5),
  AESx(0xBC63DFBC), AESx(0xB677C1B6), AESx(0xDAAF75DA), AESx(0x21426321),
  AESx(0x10203010), AESx(0xFFE51AFF), AESx(0xF3FD0EF3), AESx(0xD2BF6DD2),
  AESx(0xCD814CCD), AESx(0x0C18140C), AESx(0x13263513), AESx(0xECC32FEC),
  AESx(0x5FBEE15F), AESx(0x9735A297), AESx(0x4488CC44), AESx(0x172E3917),
  AESx(0xC49357C4), AESx(0xA755F2A7), AESx(0x7EFC827E), AESx(0x3D7A473D),
  AESx(0x64C8AC64), AESx(0x5DBAE75D), AESx(0x19322B19), AESx(0x73E69573),
  AESx(0x60C0A060), AESx(0x81199881), AESx(0x4F9ED14F), AESx(0xDCA37FDC),
  AESx(0x22446622), AESx(0x2A547E2A), AESx(0x903BAB90), AESx(0x880B8388),
  AESx(0x468CCA46), AESx(0xEEC729EE), AESx(0xB86BD3B8), AESx(0x14283C14),
  AESx(0xDEA779DE), AESx(0x5EBCE25E), AESx(0x0B161D0B), AESx(0xDBAD76DB),
  AESx(0xE0DB3BE0), AESx(0x32645632), AESx(0x3A744E3A), AESx(0x0A141E0A),
  AESx(0x4992DB49), AESx(0x060C0A06), AESx(0x24486C24), AESx(0x5CB8E45C),
  AESx(0xC29F5DC2), AESx(0xD3BD6ED3), AESx(0xAC43EFAC), AESx(0x62C4A662),
  AESx(0x9139A891), AESx(0x9531A495), AESx(0xE4D337E4), AESx(0x79F28B79),
  AESx(0xE7D532E7), AESx(0xC88B43C8), AESx(0x376E5937), AESx(0x6DDAB76D),
  AESx(0x8D018C8D), AESx(0xD5B164D5), AESx(0x4E9CD24E), AESx(0xA949E0A9),
  AESx(0x6CD8B46C), AESx(0x56ACFA56), AESx(0xF4F307F4), AESx(0xEACF25EA),
  AESx(0x65CAAF65), AESx(0x7AF48E7A), AESx(0xAE47E9AE), AESx(0x08101808),
  AESx(0xBA6FD5BA), AESx(0x78F08878), AESx(0x254A6F25), AESx(0x2E5C722E),
  AESx(0x1C38241C), AESx(0xA657F1A6), AESx(0xB473C7B4), AESx(0xC69751C6),
  AESx(0xE8CB23E8), AESx(0xDDA17CDD), AESx(0x74E89C74), AESx(0x1F3E211F),
  AESx(0x4B96DD4B), AESx(0xBD61DCBD), AESx(0x8B0D868B), AESx(0x8A0F858A),
  AESx(0x70E09070), AESx(0x3E7C423E), AESx(0xB571C4B5), AESx(0x66CCAA66),
  AESx(0x4890D848), AESx(0x03060503), AESx(0xF6F701F6), AESx(0x0E1C120E),
  AESx(0x61C2A361), AESx(0x356A5F35), AESx(0x57AEF957), AESx(0xB969D0B9),
  AESx(0x86179186), AESx(0xC19958C1), AESx(0x1D3A271D), AESx(0x9E27B99E),
  AESx(0xE1D938E1), AESx(0xF8EB13F8), AESx(0x982BB398), AESx(0x11223311),
  AESx(0x69D2BB69), AESx(0xD9A970D9), AESx(0x8E07898E), AESx(0x9433A794),
  AESx(0x9B2DB69B), AESx(0x1E3C221E), AESx(0x87159287), AESx(0xE9C920E9),
  AESx(0xCE8749CE), AESx(0x55AAFF55), AESx(0x28507828), AESx(0xDFA57ADF),
  AESx(0x8C038F8C), AESx(0xA159F8A1), AESx(0x89098089), AESx(0x0D1A170D),
  AESx(0xBF65DABF), AESx(0xE6D731E6), AESx(0x4284C642), AESx(0x68D0B868),
  AESx(0x4182C341), AESx(0x9929B099), AESx(0x2D5A772D), AESx(0x0F1E110F),
  AESx(0xB07BCBB0), AESx(0x54A8FC54), AESx(0xBB6DD6BB), AESx(0x162C3A16)
};

__constant__ static const sph_u32 AES3_C[256] = {
  AESx(0xC6A56363), AESx(0xF8847C7C), AESx(0xEE997777), AESx(0xF68D7B7B),
  AESx(0xFF0DF2F2), AESx(0xD6BD6B6B), AESx(0xDEB16F6F), AESx(0x9154C5C5),
  AESx(0x60503030), AESx(0x02030101), AESx(0xCEA96767), AESx(0x567D2B2B),
  AESx(0xE719FEFE), AESx(0xB562D7D7), AESx(0x4DE6ABAB), AESx(0xEC9A7676),
  AESx(0x8F45CACA), AESx(0x1F9D8282), AESx(0x8940C9C9), AESx(0xFA877D7D),
  AESx(0xEF15FAFA), AESx(0xB2EB5959), AESx(0x8EC94747), AESx(0xFB0BF0F0),
  AESx(0x41ECADAD), AESx(0xB367D4D4), AESx(0x5FFDA2A2), AESx(0x45EAAFAF),
  AESx(0x23BF9C9C), AESx(0x53F7A4A4), AESx(0xE4967272), AESx(0x9B5BC0C0),
  AESx(0x75C2B7B7), AESx(0xE11CFDFD), AESx(0x3DAE9393), AESx(0x4C6A2626),
  AESx(0x6C5A3636), AESx(0x7E413F3F), AESx(0xF502F7F7), AESx(0x834FCCCC),
  AESx(0x685C3434), AESx(0x51F4A5A5), AESx(0xD134E5E5), AESx(0xF908F1F1),
  AESx(0xE2937171), AESx(0xAB73D8D8), AESx(0x62533131), AESx(0x2A3F1515),
  AESx(0x080C0404), AESx(0x9552C7C7), AESx(0x46652323), AESx(0x9D5EC3C3),
  AESx(0x30281818), AESx(0x37A19696), AESx(0x0A0F0505), AESx(0x2FB59A9A),
  AESx(0x0E090707), AESx(0x24361212), AESx(0x1B9B8080), AESx(0xDF3DE2E2),
  AESx(0xCD26EBEB), AESx(0x4E692727), AESx(0x7FCDB2B2), AESx(0xEA9F7575),
  AESx(0x121B0909), AESx(0x1D9E8383), AESx(0x58742C2C), AESx(0x342E1A1A),
  AESx(0x362D1B1B), AESx(0xDCB26E6E), AESx(0xB4EE5A5A), AESx(0x5BFBA0A0),
  AESx(0xA4F65252), AESx(0x764D3B3B), AESx(0xB761D6D6), AESx(0x7DCEB3B3),
  AESx(0x527B2929), AESx(0xDD3EE3E3), AESx(0x5E712F2F), AESx(0x13978484),
  AESx(0xA6F55353), AESx(0xB968D1D1), AESx(0x00000000), AESx(0xC12CEDED),
  AESx(0x40602020), AESx(0xE31FFCFC), AESx(0x79C8B1B1), AESx(0xB6ED5B5B),
  AESx(0xD4BE6A6A), AESx(0x8D46CBCB), AESx(0x67D9BEBE), AESx(0x724B3939),
  AESx(0x94DE4A4A), AESx(0x98D44C4C), AESx(0xB0E85858), AESx(0x854ACFCF),
  AESx(0xBB6BD0D0), AESx(0xC52AEFEF), AESx(0x4FE5AAAA), AESx(0xED16FBFB),
  AESx(0x86C54343), AESx(0x9AD74D4D), AESx(0x66553333), AESx(0x11948585),
  AESx(0x8ACF4545), AESx(0xE910F9F9), AESx(0x04060202), AESx(0xFE817F7F),
  AESx(0xA0F05050), AESx(0x78443C3C), AESx(0x25BA9F9F), AESx(0x4BE3A8A8),
  AESx(0xA2F35151), AESx(0x5DFEA3A3), AESx(0x80C04040), AESx(0x058A8F8F),
  AESx(0x3FAD9292), AESx(0x21BC9D9D), AESx(0x70483838), AESx(0xF104F5F5),
  AESx(0x63DFBCBC), AESx(0x77C1B6B6), AESx(0xAF75DADA), AESx(0x42632121),
  AESx(0x20301010), AESx(0xE51AFFFF), AESx(0xFD0EF3F3), AESx(0xBF6DD2D2),
  AESx(0x814CCDCD), AESx(0x18140C0C), AESx(0x26351313), AESx(0xC32FECEC),
  AESx(0xBEE15F5F), AESx(0x35A29797), AESx(0x88CC4444), AESx(0x2E391717),
  AESx(0x9357C4C4), AESx(0x55F2A7A7), AESx(0xFC827E7E), AESx(0x7A473D3D),
  AESx(0xC8AC6464), AESx(0xBAE75D5D), AESx(0x322B1919), AESx(0xE6957373),
  AESx(0xC0A06060), AESx(0x19988181), AESx(0x9ED14F4F), AESx(0xA37FDCDC),
  AESx(0x44662222), AESx(0x547E2A2A), AESx(0x3BAB9090), AESx(0x0B838888),
  AESx(0x8CCA4646), AESx(0xC729EEEE), AESx(0x6BD3B8B8), AESx(0x283C1414),
  AESx(0xA779DEDE), AESx(0xBCE25E5E), AESx(0x161D0B0B), AESx(0xAD76DBDB),
  AESx(0xDB3BE0E0), AESx(0x64563232), AESx(0x744E3A3A), AESx(0x141E0A0A),
  AESx(0x92DB4949), AESx(0x0C0A0606), AESx(0x486C2424), AESx(0xB8E45C5C),
  AESx(0x9F5DC2C2), AESx(0xBD6ED3D3), AESx(0x43EFACAC), AESx(0xC4A66262),
  AESx(0x39A89191), AESx(0x31A49595), AESx(0xD337E4E4), AESx(0xF28B7979),
  AESx(0xD532E7E7), AESx(0x8B43C8C8), AESx(0x6E593737), AESx(0xDAB76D6D),
  AESx(0x018C8D8D), AESx(0xB164D5D5), AESx(0x9CD24E4E), AESx(0x49E0A9A9),
  AESx(0xD8B46C6C), AESx(0xACFA5656), AESx(0xF307F4F4), AESx(0xCF25EAEA),
  AESx(0xCAAF6565), AESx(0xF48E7A7A), AESx(0x47E9AEAE), AESx(0x10180808),
  AESx(0x6FD5BABA), AESx(0xF0887878), AESx(0x4A6F2525), AESx(0x5C722E2E),
  AESx(0x38241C1C), AESx(0x57F1A6A6), AESx(0x73C7B4B4), AESx(0x9751C6C6),
  AESx(0xCB23E8E8), AESx(0xA17CDDDD), AESx(0xE89C7474), AESx(0x3E211F1F),
  AESx(0x96DD4B4B), AESx(0x61DCBDBD), AESx(0x0D868B8B), AESx(0x0F858A8A),
  AESx(0xE0907070), AESx(0x7C423E3E), AESx(0x71C4B5B5), AESx(0xCCAA6666),
  AESx(0x90D84848), AESx(0x06050303), AESx(0xF701F6F6), AESx(0x1C120E0E),
  AESx(0xC2A36161), AESx(0x6A5F3535), AESx(0xAEF95757), AESx(0x69D0B9B9),
  AESx(0x17918686), AESx(0x9958C1C1), AESx(0x3A271D1D), AESx(0x27B99E9E),
  AESx(0xD938E1E1), AESx(0xEB13F8F8), AESx(0x2BB39898), AESx(0x22331111),
  AESx(0xD2BB6969), AESx(0xA970D9D9), AESx(0x07898E8E), AESx(0x33A79494),
  AESx(0x2DB69B9B), AESx(0x3C221E1E), AESx(0x15928787), AESx(0xC920E9E9),
  AESx(0x8749CECE), AESx(0xAAFF5555), AESx(0x50782828), AESx(0xA57ADFDF),
  AESx(0x038F8C8C), AESx(0x59F8A1A1), AESx(0x09808989), AESx(0x1A170D0D),
  AESx(0x65DABFBF), AESx(0xD731E6E6), AESx(0x84C64242), AESx(0xD0B86868),
  AESx(0x82C34141), AESx(0x29B09999), AESx(0x5A772D2D), AESx(0x1E110F0F),
  AESx(0x7BCBB0B0), AESx(0xA8FC5454), AESx(0x6DD6BBBB), AESx(0x2C3A1616)
};



#define AES_ROUND_NOKEY(x0, x1, x2, x3)   do { \
    sph_u32 t0 = (x0); \
    sph_u32 t1 = (x1); \
    sph_u32 t2 = (x2); \
    sph_u32 t3 = (x3); \
    AES_ROUND_NOKEY_LE(t0, t1, t2, t3, x0, x1, x2, x3); \
  } while (0)

#define KEY_EXPAND_ELT(k0, k1, k2, k3)   do { \
    sph_u32 kt; \
    AES_ROUND_NOKEY(k1, k2, k3, k0); \
    kt = (k0); \
    (k0) = (k1); \
    (k1) = (k2); \
    (k2) = (k3); \
    (k3) = kt; \
  } while (0)

/*
 * This function assumes that "msg" is aligned for 32-bit access.
 */
#define c512(msg)  do { \
  sph_u32 p0, p1, p2, p3, p4, p5, p6, p7; \
  sph_u32 p8, p9, pA, pB, pC, pD, pE, pF; \
  sph_u32 x0, x1, x2, x3; \
  int r; \
 \
  p0 = h0; \
  p1 = h1; \
  p2 = h2; \
  p3 = h3; \
  p4 = h4; \
  p5 = h5; \
  p6 = h6; \
  p7 = h7; \
  p8 = h8; \
  p9 = h9; \
  pA = hA; \
  pB = hB; \
  pC = hC; \
  pD = hD; \
  pE = hE; \
  pF = hF; \
  /* round 0 */ \
  x0 = p4 ^ rk00; \
  x1 = p5 ^ rk01; \
  x2 = p6 ^ rk02; \
  x3 = p7 ^ rk03; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  x0 ^= rk04; \
  x1 ^= rk05; \
  x2 ^= rk06; \
  x3 ^= rk07; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  x0 ^= rk08; \
  x1 ^= rk09; \
  x2 ^= rk0A; \
  x3 ^= rk0B; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  x0 ^= rk0C; \
  x1 ^= rk0D; \
  x2 ^= rk0E; \
  x3 ^= rk0F; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  p0 ^= x0; \
  p1 ^= x1; \
  p2 ^= x2; \
  p3 ^= x3; \
  x0 = pC ^ rk10; \
  x1 = pD ^ rk11; \
  x2 = pE ^ rk12; \
  x3 = pF ^ rk13; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  x0 ^= rk14; \
  x1 ^= rk15; \
  x2 ^= rk16; \
  x3 ^= rk17; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  x0 ^= rk18; \
  x1 ^= rk19; \
  x2 ^= rk1A; \
  x3 ^= rk1B; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  x0 ^= rk1C; \
  x1 ^= rk1D; \
  x2 ^= rk1E; \
  x3 ^= rk1F; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  p8 ^= x0; \
  p9 ^= x1; \
  pA ^= x2; \
  pB ^= x3; \
 \
  for (r = 0; r < 3; r ++) { \
    /* round 1, 5, 9 */ \
    KEY_EXPAND_ELT(rk00, rk01, rk02, rk03); \
    rk00 ^= rk1C; \
    rk01 ^= rk1D; \
    rk02 ^= rk1E; \
    rk03 ^= rk1F; \
    if (r == 0) { \
      rk00 ^= sc_count0; \
      rk01 ^= sc_count1; \
      rk02 ^= sc_count2; \
      rk03 ^= SPH_T32(~sc_count3); \
    } \
    x0 = p0 ^ rk00; \
    x1 = p1 ^ rk01; \
    x2 = p2 ^ rk02; \
    x3 = p3 ^ rk03; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk04, rk05, rk06, rk07); \
    rk04 ^= rk00; \
    rk05 ^= rk01; \
    rk06 ^= rk02; \
    rk07 ^= rk03; \
    if (r == 1) { \
      rk04 ^= sc_count3; \
      rk05 ^= sc_count2; \
      rk06 ^= sc_count1; \
      rk07 ^= SPH_T32(~sc_count0); \
    } \
    x0 ^= rk04; \
    x1 ^= rk05; \
    x2 ^= rk06; \
    x3 ^= rk07; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk08, rk09, rk0A, rk0B); \
    rk08 ^= rk04; \
    rk09 ^= rk05; \
    rk0A ^= rk06; \
    rk0B ^= rk07; \
    x0 ^= rk08; \
    x1 ^= rk09; \
    x2 ^= rk0A; \
    x3 ^= rk0B; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk0C, rk0D, rk0E, rk0F); \
    rk0C ^= rk08; \
    rk0D ^= rk09; \
    rk0E ^= rk0A; \
    rk0F ^= rk0B; \
    x0 ^= rk0C; \
    x1 ^= rk0D; \
    x2 ^= rk0E; \
    x3 ^= rk0F; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    pC ^= x0; \
    pD ^= x1; \
    pE ^= x2; \
    pF ^= x3; \
    KEY_EXPAND_ELT(rk10, rk11, rk12, rk13); \
    rk10 ^= rk0C; \
    rk11 ^= rk0D; \
    rk12 ^= rk0E; \
    rk13 ^= rk0F; \
    x0 = p8 ^ rk10; \
    x1 = p9 ^ rk11; \
    x2 = pA ^ rk12; \
    x3 = pB ^ rk13; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk14, rk15, rk16, rk17); \
    rk14 ^= rk10; \
    rk15 ^= rk11; \
    rk16 ^= rk12; \
    rk17 ^= rk13; \
    x0 ^= rk14; \
    x1 ^= rk15; \
    x2 ^= rk16; \
    x3 ^= rk17; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk18, rk19, rk1A, rk1B); \
    rk18 ^= rk14; \
    rk19 ^= rk15; \
    rk1A ^= rk16; \
    rk1B ^= rk17; \
    x0 ^= rk18; \
    x1 ^= rk19; \
    x2 ^= rk1A; \
    x3 ^= rk1B; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk1C, rk1D, rk1E, rk1F); \
    rk1C ^= rk18; \
    rk1D ^= rk19; \
    rk1E ^= rk1A; \
    rk1F ^= rk1B; \
    if (r == 2) { \
      rk1C ^= sc_count2; \
      rk1D ^= sc_count3; \
      rk1E ^= sc_count0; \
      rk1F ^= SPH_T32(~sc_count1); \
    } \
    x0 ^= rk1C; \
    x1 ^= rk1D; \
    x2 ^= rk1E; \
    x3 ^= rk1F; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    p4 ^= x0; \
    p5 ^= x1; \
    p6 ^= x2; \
    p7 ^= x3; \
    /* round 2, 6, 10 */ \
    rk00 ^= rk19; \
    x0 = pC ^ rk00; \
    rk01 ^= rk1A; \
    x1 = pD ^ rk01; \
    rk02 ^= rk1B; \
    x2 = pE ^ rk02; \
    rk03 ^= rk1C; \
    x3 = pF ^ rk03; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk04 ^= rk1D; \
    x0 ^= rk04; \
    rk05 ^= rk1E; \
    x1 ^= rk05; \
    rk06 ^= rk1F; \
    x2 ^= rk06; \
    rk07 ^= rk00; \
    x3 ^= rk07; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk08 ^= rk01; \
    x0 ^= rk08; \
    rk09 ^= rk02; \
    x1 ^= rk09; \
    rk0A ^= rk03; \
    x2 ^= rk0A; \
    rk0B ^= rk04; \
    x3 ^= rk0B; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk0C ^= rk05; \
    x0 ^= rk0C; \
    rk0D ^= rk06; \
    x1 ^= rk0D; \
    rk0E ^= rk07; \
    x2 ^= rk0E; \
    rk0F ^= rk08; \
    x3 ^= rk0F; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    p8 ^= x0; \
    p9 ^= x1; \
    pA ^= x2; \
    pB ^= x3; \
    rk10 ^= rk09; \
    x0 = p4 ^ rk10; \
    rk11 ^= rk0A; \
    x1 = p5 ^ rk11; \
    rk12 ^= rk0B; \
    x2 = p6 ^ rk12; \
    rk13 ^= rk0C; \
    x3 = p7 ^ rk13; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk14 ^= rk0D; \
    x0 ^= rk14; \
    rk15 ^= rk0E; \
    x1 ^= rk15; \
    rk16 ^= rk0F; \
    x2 ^= rk16; \
    rk17 ^= rk10; \
    x3 ^= rk17; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk18 ^= rk11; \
    x0 ^= rk18; \
    rk19 ^= rk12; \
    x1 ^= rk19; \
    rk1A ^= rk13; \
    x2 ^= rk1A; \
    rk1B ^= rk14; \
    x3 ^= rk1B; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk1C ^= rk15; \
    x0 ^= rk1C; \
    rk1D ^= rk16; \
    x1 ^= rk1D; \
    rk1E ^= rk17; \
    x2 ^= rk1E; \
    rk1F ^= rk18; \
    x3 ^= rk1F; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    p0 ^= x0; \
    p1 ^= x1; \
    p2 ^= x2; \
    p3 ^= x3; \
    /* round 3, 7, 11 */ \
    KEY_EXPAND_ELT(rk00, rk01, rk02, rk03); \
    rk00 ^= rk1C; \
    rk01 ^= rk1D; \
    rk02 ^= rk1E; \
    rk03 ^= rk1F; \
    x0 = p8 ^ rk00; \
    x1 = p9 ^ rk01; \
    x2 = pA ^ rk02; \
    x3 = pB ^ rk03; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk04, rk05, rk06, rk07); \
    rk04 ^= rk00; \
    rk05 ^= rk01; \
    rk06 ^= rk02; \
    rk07 ^= rk03; \
    x0 ^= rk04; \
    x1 ^= rk05; \
    x2 ^= rk06; \
    x3 ^= rk07; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk08, rk09, rk0A, rk0B); \
    rk08 ^= rk04; \
    rk09 ^= rk05; \
    rk0A ^= rk06; \
    rk0B ^= rk07; \
    x0 ^= rk08; \
    x1 ^= rk09; \
    x2 ^= rk0A; \
    x3 ^= rk0B; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk0C, rk0D, rk0E, rk0F); \
    rk0C ^= rk08; \
    rk0D ^= rk09; \
    rk0E ^= rk0A; \
    rk0F ^= rk0B; \
    x0 ^= rk0C; \
    x1 ^= rk0D; \
    x2 ^= rk0E; \
    x3 ^= rk0F; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    p4 ^= x0; \
    p5 ^= x1; \
    p6 ^= x2; \
    p7 ^= x3; \
    KEY_EXPAND_ELT(rk10, rk11, rk12, rk13); \
    rk10 ^= rk0C; \
    rk11 ^= rk0D; \
    rk12 ^= rk0E; \
    rk13 ^= rk0F; \
    x0 = p0 ^ rk10; \
    x1 = p1 ^ rk11; \
    x2 = p2 ^ rk12; \
    x3 = p3 ^ rk13; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk14, rk15, rk16, rk17); \
    rk14 ^= rk10; \
    rk15 ^= rk11; \
    rk16 ^= rk12; \
    rk17 ^= rk13; \
    x0 ^= rk14; \
    x1 ^= rk15; \
    x2 ^= rk16; \
    x3 ^= rk17; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk18, rk19, rk1A, rk1B); \
    rk18 ^= rk14; \
    rk19 ^= rk15; \
    rk1A ^= rk16; \
    rk1B ^= rk17; \
    x0 ^= rk18; \
    x1 ^= rk19; \
    x2 ^= rk1A; \
    x3 ^= rk1B; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk1C, rk1D, rk1E, rk1F); \
    rk1C ^= rk18; \
    rk1D ^= rk19; \
    rk1E ^= rk1A; \
    rk1F ^= rk1B; \
    x0 ^= rk1C; \
    x1 ^= rk1D; \
    x2 ^= rk1E; \
    x3 ^= rk1F; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    pC ^= x0; \
    pD ^= x1; \
    pE ^= x2; \
    pF ^= x3; \
    /* round 4, 8, 12 */ \
    rk00 ^= rk19; \
    x0 = p4 ^ rk00; \
    rk01 ^= rk1A; \
    x1 = p5 ^ rk01; \
    rk02 ^= rk1B; \
    x2 = p6 ^ rk02; \
    rk03 ^= rk1C; \
    x3 = p7 ^ rk03; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk04 ^= rk1D; \
    x0 ^= rk04; \
    rk05 ^= rk1E; \
    x1 ^= rk05; \
    rk06 ^= rk1F; \
    x2 ^= rk06; \
    rk07 ^= rk00; \
    x3 ^= rk07; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk08 ^= rk01; \
    x0 ^= rk08; \
    rk09 ^= rk02; \
    x1 ^= rk09; \
    rk0A ^= rk03; \
    x2 ^= rk0A; \
    rk0B ^= rk04; \
    x3 ^= rk0B; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk0C ^= rk05; \
    x0 ^= rk0C; \
    rk0D ^= rk06; \
    x1 ^= rk0D; \
    rk0E ^= rk07; \
    x2 ^= rk0E; \
    rk0F ^= rk08; \
    x3 ^= rk0F; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    p0 ^= x0; \
    p1 ^= x1; \
    p2 ^= x2; \
    p3 ^= x3; \
    rk10 ^= rk09; \
    x0 = pC ^ rk10; \
    rk11 ^= rk0A; \
    x1 = pD ^ rk11; \
    rk12 ^= rk0B; \
    x2 = pE ^ rk12; \
    rk13 ^= rk0C; \
    x3 = pF ^ rk13; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk14 ^= rk0D; \
    x0 ^= rk14; \
    rk15 ^= rk0E; \
    x1 ^= rk15; \
    rk16 ^= rk0F; \
    x2 ^= rk16; \
    rk17 ^= rk10; \
    x3 ^= rk17; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk18 ^= rk11; \
    x0 ^= rk18; \
    rk19 ^= rk12; \
    x1 ^= rk19; \
    rk1A ^= rk13; \
    x2 ^= rk1A; \
    rk1B ^= rk14; \
    x3 ^= rk1B; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk1C ^= rk15; \
    x0 ^= rk1C; \
    rk1D ^= rk16; \
    x1 ^= rk1D; \
    rk1E ^= rk17; \
    x2 ^= rk1E; \
    rk1F ^= rk18; \
    x3 ^= rk1F; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    p8 ^= x0; \
    p9 ^= x1; \
    pA ^= x2; \
    pB ^= x3; \
  } \
  /* round 13 */ \
  KEY_EXPAND_ELT(rk00, rk01, rk02, rk03); \
  rk00 ^= rk1C; \
  rk01 ^= rk1D; \
  rk02 ^= rk1E; \
  rk03 ^= rk1F; \
  x0 = p0 ^ rk00; \
  x1 = p1 ^ rk01; \
  x2 = p2 ^ rk02; \
  x3 = p3 ^ rk03; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  KEY_EXPAND_ELT(rk04, rk05, rk06, rk07); \
  rk04 ^= rk00; \
  rk05 ^= rk01; \
  rk06 ^= rk02; \
  rk07 ^= rk03; \
  x0 ^= rk04; \
  x1 ^= rk05; \
  x2 ^= rk06; \
  x3 ^= rk07; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  KEY_EXPAND_ELT(rk08, rk09, rk0A, rk0B); \
  rk08 ^= rk04; \
  rk09 ^= rk05; \
  rk0A ^= rk06; \
  rk0B ^= rk07; \
  x0 ^= rk08; \
  x1 ^= rk09; \
  x2 ^= rk0A; \
  x3 ^= rk0B; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  KEY_EXPAND_ELT(rk0C, rk0D, rk0E, rk0F); \
  rk0C ^= rk08; \
  rk0D ^= rk09; \
  rk0E ^= rk0A; \
  rk0F ^= rk0B; \
  x0 ^= rk0C; \
  x1 ^= rk0D; \
  x2 ^= rk0E; \
  x3 ^= rk0F; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  pC ^= x0; \
  pD ^= x1; \
  pE ^= x2; \
  pF ^= x3; \
  KEY_EXPAND_ELT(rk10, rk11, rk12, rk13); \
  rk10 ^= rk0C; \
  rk11 ^= rk0D; \
  rk12 ^= rk0E; \
  rk13 ^= rk0F; \
  x0 = p8 ^ rk10; \
  x1 = p9 ^ rk11; \
  x2 = pA ^ rk12; \
  x3 = pB ^ rk13; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  KEY_EXPAND_ELT(rk14, rk15, rk16, rk17); \
  rk14 ^= rk10; \
  rk15 ^= rk11; \
  rk16 ^= rk12; \
  rk17 ^= rk13; \
  x0 ^= rk14; \
  x1 ^= rk15; \
  x2 ^= rk16; \
  x3 ^= rk17; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  KEY_EXPAND_ELT(rk18, rk19, rk1A, rk1B); \
  rk18 ^= rk14 ^ sc_count1; \
  rk19 ^= rk15 ^ sc_count0; \
  rk1A ^= rk16 ^ sc_count3; \
  rk1B ^= rk17 ^ SPH_T32(~sc_count2); \
  x0 ^= rk18; \
  x1 ^= rk19; \
  x2 ^= rk1A; \
  x3 ^= rk1B; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  KEY_EXPAND_ELT(rk1C, rk1D, rk1E, rk1F); \
  rk1C ^= rk18; \
  rk1D ^= rk19; \
  rk1E ^= rk1A; \
  rk1F ^= rk1B; \
  x0 ^= rk1C; \
  x1 ^= rk1D; \
  x2 ^= rk1E; \
  x3 ^= rk1F; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  p4 ^= x0; \
  p5 ^= x1; \
  p6 ^= x2; \
  p7 ^= x3; \
  h0 ^= p8; \
  h1 ^= p9; \
  h2 ^= pA; \
  h3 ^= pB; \
  h4 ^= pC; \
  h5 ^= pD; \
  h6 ^= pE; \
  h7 ^= pF; \
  h8 ^= p0; \
  h9 ^= p1; \
  hA ^= p2; \
  hB ^= p3; \
  hC ^= p4; \
  hD ^= p5; \
  hE ^= p6; \
  hF ^= p7; \
  } while (0)







#define ECHO_DECL_STATE_BIG   \
  sph_u64 W00, W01, W10, W11, W20, W21, W30, W31, W40, W41, W50, W51, W60, W61, W70, W71, W80, W81, W90, W91, WA0, WA1, WB0, WB1, WC0, WC1, WD0, WD1, WE0, WE1, WF0, WF1;

#define AES_2ROUNDS(XX, XY)   do { \
    sph_u32 X0 ,X1, X2, X3; \
    LOHI(X0,X1,XX); LOHI(X2,X3,XY);\
    sph_u32 Y0, Y1, Y2, Y3; \
    AES_ROUND_LE(X0, X1, X2, X3, K0, K1, K2, K3, Y0, Y1, Y2, Y3); \
    AES_ROUND_NOKEY_LE(Y0, Y1, Y2, Y3, X0, X1, X2, X3); \
    XX = MAKE_ULONGLONG(X0,X1);\
    XY = MAKE_ULONGLONG(X2,X3);\
    if ((K0 = T32(K0 + 1)) == 0) { \
      if ((K1 = T32(K1 + 1)) == 0) \
        if ((K2 = T32(K2 + 1)) == 0) \
          K3 = T32(K3 + 1); \
    } \
  } while (0)

#define BIG_SUB_WORDS   do { \
    AES_2ROUNDS(W00, W01); \
    AES_2ROUNDS(W10, W11); \
    AES_2ROUNDS(W20, W21); \
    AES_2ROUNDS(W30, W31); \
    AES_2ROUNDS(W40, W41); \
    AES_2ROUNDS(W50, W51); \
    AES_2ROUNDS(W60, W61); \
    AES_2ROUNDS(W70, W71); \
    AES_2ROUNDS(W80, W81); \
    AES_2ROUNDS(W90, W91); \
    AES_2ROUNDS(WA0, WA1); \
    AES_2ROUNDS(WB0, WB1); \
    AES_2ROUNDS(WC0, WC1); \
    AES_2ROUNDS(WD0, WD1); \
    AES_2ROUNDS(WE0, WE1); \
    AES_2ROUNDS(WF0, WF1); \
  } while (0)

#define SHIFT_ROW1(a, b, c, d)   do { \
    sph_u64 tmp; \
    tmp = W ## a ## 0; \
    W ## a ## 0 = W ## b ## 0; \
    W ## b ## 0 = W ## c ## 0; \
    W ## c ## 0 = W ## d ## 0; \
    W ## d ## 0 = tmp; \
    tmp = W ## a ## 1; \
    W ## a ## 1 = W ## b ## 1; \
    W ## b ## 1 = W ## c ## 1; \
    W ## c ## 1 = W ## d ## 1; \
    W ## d ## 1 = tmp; \
  } while (0)

#define SHIFT_ROW2(a, b, c, d)   do { \
    sph_u64 tmp; \
    tmp = W ## a ## 0; \
    W ## a ## 0 = W ## c ## 0; \
    W ## c ## 0 = tmp; \
    tmp = W ## b ## 0; \
    W ## b ## 0 = W ## d ## 0; \
    W ## d ## 0 = tmp; \
    tmp = W ## a ## 1; \
    W ## a ## 1 = W ## c ## 1; \
    W ## c ## 1 = tmp; \
    tmp = W ## b ## 1; \
    W ## b ## 1 = W ## d ## 1; \
    W ## d ## 1 = tmp; \
  } while (0)

#define SHIFT_ROW3(a, b, c, d)   SHIFT_ROW1(d, c, b, a)

#define BIG_SHIFT_ROWS   do { \
    SHIFT_ROW1(1, 5, 9, D); \
    SHIFT_ROW2(2, 6, A, E); \
    SHIFT_ROW3(3, 7, B, F); \
  } while (0)

#define MIX_COLUMN1(ia, ib, ic, id, n)   do { \
    sph_u64 a = W ## ia ## n; \
    sph_u64 b = W ## ib ## n; \
    sph_u64 c = W ## ic ## n; \
    sph_u64 d = W ## id ## n; \
    sph_u64 ab = a ^ b; \
    sph_u64 bc = b ^ c; \
    sph_u64 cd = c ^ d; \
    sph_u64 abx = ((ab & C64(0x8080808080808080)) >> 7) * 27U \
      ^ ((ab & C64(0x7F7F7F7F7F7F7F7F)) << 1); \
    sph_u64 bcx = ((bc & C64(0x8080808080808080)) >> 7) * 27U \
      ^ ((bc & C64(0x7F7F7F7F7F7F7F7F)) << 1); \
    sph_u64 cdx = ((cd & C64(0x8080808080808080)) >> 7) * 27U \
      ^ ((cd & C64(0x7F7F7F7F7F7F7F7F)) << 1); \
    W ## ia ## n = abx ^ bc ^ d; \
    W ## ib ## n = bcx ^ a ^ cd; \
    W ## ic ## n = cdx ^ ab ^ d; \
    W ## id ## n = abx ^ bcx ^ cdx ^ ab ^ c; \
  } while (0)

#define MIX_COLUMN(a, b, c, d)   do { \
    MIX_COLUMN1(a, b, c, d, 0); \
    MIX_COLUMN1(a, b, c, d, 1); \
  } while (0)

#define BIG_MIX_COLUMNS   do { \
    MIX_COLUMN(0, 1, 2, 3); \
    MIX_COLUMN(4, 5, 6, 7); \
    MIX_COLUMN(8, 9, A, B); \
    MIX_COLUMN(C, D, E, F); \
  } while (0)

#define BIG_ROUND   do { \
    BIG_SUB_WORDS; \
    BIG_SHIFT_ROWS; \
    BIG_MIX_COLUMNS; \
  } while (0)


#define T32   SPH_T32
#define C32   SPH_C32
#define C64   SPH_C64


#define TPBS 256

__global__ __launch_bounds__(TPBS,2) /* 64 registers with 128,8 - 72 regs with 128,7 */
void xevan_shavite512_gpu_hash_64(const uint32_t threads, uint64_t *g_hash)
{
const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t r[16];
		uint64_t *r8=(uint64_t *)r;
		uint64_t *Hash = &g_hash[thread<<3];

__shared__ uint32_t AES0_S[256];
__shared__ uint32_t AES1_S[256];
__shared__ uint32_t AES2_S[256];
//__shared__ uint32_t AES3_S[256];
        if(threadIdx.x < 256){
//	uint32_t temp = __ldg(&AES0_C[threadIdx.x]);
	uint32_t temp = AES0_C[threadIdx.x];
	AES0_S[threadIdx.x] = temp;
	AES1_S[threadIdx.x] = ROL8(temp);
        AES2_S[threadIdx.x] = ROL16(temp);
//        AES3_S[threadIdx.x] = ROR8(temp);
}



		// fülle die Nachricht mit 64-byte (vorheriger Hash)
		*(uint2x4*)&r[ 0] = __ldg4((uint2x4*)&Hash[ 0]);
		*(uint2x4*)&r[ 8] = __ldg4((uint2x4*)&Hash[ 4]);

//__syncthreads();

  sph_u32 h0 = SPH_C32(0x72FCCDD8), h1 = SPH_C32(0x79CA4727), h2 = SPH_C32(0x128A077B), h3 = SPH_C32(0x40D55AEC);
  sph_u32 h4 = SPH_C32(0xD1901A06), h5 = SPH_C32(0x430AE307), h6 = SPH_C32(0xB29F5CD1), h7 = SPH_C32(0xDF07FBFC);
  sph_u32 h8 = SPH_C32(0x8E45D73D), h9 = SPH_C32(0x681AB538), hA = SPH_C32(0xBDE86578), hB = SPH_C32(0xDD577E47);
  sph_u32 hC = SPH_C32(0xE275EADE), hD = SPH_C32(0x502D9FCD), hE = SPH_C32(0xB9357178), hF = SPH_C32(0x022A4B9A);

  // state
  sph_u32 rk00, rk01, rk02, rk03, rk04, rk05, rk06, rk07;
  sph_u32 rk08, rk09, rk0A, rk0B, rk0C, rk0D, rk0E, rk0F;
  sph_u32 rk10, rk11, rk12, rk13, rk14, rk15, rk16, rk17;
  sph_u32 rk18, rk19, rk1A, rk1B, rk1C, rk1D, rk1E, rk1F;

  sph_u32 sc_count0 = 0x400, sc_count1 = 0, sc_count2 = 0, sc_count3 = 0;

  rk00 = r[0];
  rk01 = r[1];
  rk02 = r[2];
  rk03 = r[3];
  rk04 = r[4];
  rk05 = r[5];
  rk06 = r[6];
  rk07 = r[7];
  rk08 = r[8];
  rk09 = r[9];
  rk0A = r[10];
  rk0B = r[11];
  rk0C = r[12];
  rk0D = r[13];
  rk0E = r[14];
  rk0F = r[15];
  rk10 = rk11 = rk12 = rk13 = rk14 = rk15 = rk16 = rk17 = 0;
  rk18 = rk19 = rk1A = rk1B = rk1C = rk1D = rk1E = rk1F = 0;

__syncthreads();

//#pragma unroll 3
  c512(buf);

  rk00 = 0x80;
  rk01 = rk02 = rk03 = rk04 = rk05 = rk06 = rk07 = 0;
  rk08 = rk09 = rk0A = rk0B = rk0C = rk0D = rk0E = rk0F = 0;
  rk10 = rk11 = rk12 = rk13 = rk14 = rk15 = rk16 = rk17 = rk18 = rk19 = rk1A = 0;
  rk1B = 0x4000000;
  rk1C = rk1D = rk1E = 0;
  rk1F = 0x2000000;
  sc_count0 = 0;
//#pragma unroll 3
  c512(buf);

  r[0] = h0;
  r[1] = h1;
  r[2] = h2;
  r[3] = h3;
  r[4] = h4;
  r[5] = h5;
  r[6] = h6;
  r[7] = h7;
  r[8] = h8;
  r[9] = h9;
  r[10] = hA;
  r[11] = hB;
  r[12] = hC;
  r[13] = hD;
  r[14] = hE;
  r[15] = hF;


                *((uint2x4*)&Hash[ 0]) = *(uint2x4*)&r[ 0];
                *((uint2x4*)&Hash[ 4]) = *(uint2x4*)&r[ 8];


	}
}


__host__
void xevan_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
        dim3 grid((threads + TPBS-1)/TPBS);
        dim3 block(TPBS);

        // note: 128 threads minimum are required to init the shared memory array
        xevan_shavite512_gpu_hash_64<<<grid, block>>>(threads, (uint64_t*)d_hash);
}

#define TPBE 256

__global__ __launch_bounds__(TPBE,2) /* 64 registers with 128,8 - 72 regs with 128,7 */
void xevan_echo512_gpu_hash_64(const uint32_t threads, uint64_t *g_hash)
{
const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
        if (thread < threads)
        {
//                uint32_t r[16];
                uint64_t r8[8];
                uint64_t *Hash = &g_hash[thread<<3];

__shared__ uint32_t AES0_S[256];
__shared__ uint32_t AES1_S[256];
__shared__ uint32_t AES2_S[256];
//__shared__ uint32_t AES3_S[256];
        if(threadIdx.x < 256){
  //      uint32_t temp = __ldg(&AES0_C[threadIdx.x]);
uint32_t temp = AES0_C[threadIdx.x];
        AES0_S[threadIdx.x] = temp;
        AES1_S[threadIdx.x] = ROL8(temp);

        AES2_S[threadIdx.x] = ROL16(temp);
//        AES3_S[threadIdx.x] = ROR8(temp);
}

                // fülle die Nachricht mit 64-byte (vorheriger Hash)
                *(uint2x4*)&r8[ 0] = __ldg4((uint2x4*)&Hash[ 0]);
                *(uint2x4*)&r8[ 4] = __ldg4((uint2x4*)&Hash[ 4]);
__syncthreads();
//for(int i=0;i<8;i++)r8[i]=0;
//echo
  sph_u64 W00, W01, W10, W11, W20, W21, W30, W31, W40, W41, W50, W51, W60, W61, W70, W71, W80, W81, W90, W91, WA0, WA1, WB0, WB1, WC0, WC1, WD0, WD1, WE0, WE1, WF0, WF1;
  sph_u64 Vb00, Vb01, Vb10, Vb11, Vb20, Vb21, Vb30, Vb31, Vb40, Vb41, Vb50, Vb51, Vb60, Vb61, Vb70, Vb71;
  Vb00 = Vb10 = Vb20 = Vb30 = Vb40 = Vb50 = Vb60 = Vb70 = 512UL;
  Vb01 = Vb11 = Vb21 = Vb31 = Vb41 = Vb51 = Vb61 = Vb71 = 0;

  sph_u32 K0 = 1024;
  sph_u32 K1 = 0;
  sph_u32 K2 = 0;
  sph_u32 K3 = 0;

  W00 = Vb00;
  W01 = Vb01;
  W10 = Vb10;
  W11 = Vb11;
  W20 = Vb20;
  W21 = Vb21;
  W30 = Vb30;
  W31 = Vb31;
  W40 = Vb40;
  W41 = Vb41;
  W50 = Vb50;
  W51 = Vb51;
  W60 = Vb60;
  W61 = Vb61;
  W70 = Vb70;
  W71 = Vb71;
  W80 = r8[0];
  W81 = r8[1];
  W90 = r8[2];
  W91 = r8[3];
  WA0 = r8[4];
  WA1 = r8[5];
  WB0 = r8[6];
  WB1 = r8[7];
  WC0 = 0;
  WC1 = 0;
  WD0 = 0;
  WD1 = 0;
  WE0 = 0;
  WE1 = 0;
  WF0 = 0;
  WF1 = 0;

//#pragma unroll 10
  for (unsigned u = 0; u < 10; u++) {
    BIG_ROUND;
  }


/*
/////
  r8[0] = W00;
  r8[1] = W01;
  r8[2] = W10;
  r8[3] = W11;
  r8[4] = W20;
  r8[5] = W21;
  r8[6] = W30;
  r8[7] = W31;
                *((uint2x4*)&Hash[ 0]) = *(uint2x4*)&r8[ 0];
                *((uint2x4*)&Hash[ 4]) = *(uint2x4*)&r8[ 4];
return;
/////
*/

  Vb00 ^= r8[0] ^ W00 ^ W80;
  Vb01 ^= r8[1] ^ W01 ^ W81;
  Vb10 ^= r8[2] ^ W10 ^ W90;
  Vb11 ^= r8[3] ^ W11 ^ W91;
  Vb20 ^= r8[4] ^ W20 ^ WA0;
  Vb21 ^= r8[5] ^ W21 ^ WA1;
  Vb30 ^= r8[6] ^ W30 ^ WB0;
  Vb31 ^= r8[7] ^ W31 ^ WB1;
  Vb40 ^= W40 ^ WC0;
  Vb41 ^= W41 ^ WC1;
  Vb50 ^= W50 ^ WD0;
  Vb51 ^= W51 ^ WD1;
  Vb60 ^= W60 ^ WE0;
  Vb61 ^= W61 ^ WE1;
  Vb70 ^= W70 ^ WF0;
  Vb71 ^= W71 ^ WF1;


  W00 = Vb00;
  W01 = Vb01;
  W10 = Vb10;
  W11 = Vb11;
  W20 = Vb20;
  W21 = Vb21;
  W30 = Vb30;
  W31 = Vb31;
  W40 = Vb40;
  W41 = Vb41;
  W50 = Vb50;
  W51 = Vb51;
  W60 = Vb60;
  W61 = Vb61;
  W70 = Vb70;
  W71 = Vb71;
  W80 = 0x80;
  W81 = W90 = W91 = WA0 = WA1 = WB0 = WB1 = WC0 = WC1 = WD0 = WD1 = WE0 = 0;
  WE1 = 0x200000000000000;
  WF0 = 0x400;
  WF1 = 0;
  K0 = K1 = K2 = K3 = 0;
/*
/////
  r8[0] = WC0;
  r8[1] = WC1;
  r8[2] = WD0;
  r8[3] = WD1;
  r8[4] = WE0;
  r8[5] = WE1;
  r8[6] = WF0;
  r8[7] = WF1;
                *((uint2x4*)&Hash[ 0]) = *(uint2x4*)&r8[ 0];
                *((uint2x4*)&Hash[ 4]) = *(uint2x4*)&r8[ 4];
return;
/////
*/
//#pragma unroll 10
  for (unsigned u = 0; u < 10; u++) {
    BIG_ROUND;
  }
/*
/////
  r8[0] = W00;
  r8[1] = W01;
  r8[2] = W10;
  r8[3] = W11;
  r8[4] = W20;
  r8[5] = W21;
  r8[6] = W30;
  r8[7] = W31;
                *((uint2x4*)&Hash[ 0]) = *(uint2x4*)&r8[ 0];
                *((uint2x4*)&Hash[ 4]) = *(uint2x4*)&r8[ 4];
return;
/////
*/

  Vb00 ^= 0x80 ^ W00 ^ W80;
  Vb01 ^= W01 ^ W81;
  Vb10 ^= W10 ^ W90;
  Vb11 ^= W11 ^ W91;
  Vb20 ^= W20 ^ WA0;
  Vb21 ^= W21 ^ WA1;
  Vb30 ^= W30 ^ WB0;
  Vb31 ^= W31 ^ WB1;

  r8[0] = Vb00;
  r8[1] = Vb01;
  r8[2] = Vb10;
  r8[3] = Vb11;
  r8[4] = Vb20;
  r8[5] = Vb21;
  r8[6] = Vb30;
  r8[7] = Vb31;
                *((uint2x4*)&Hash[ 0]) = *(uint2x4*)&r8[ 0];
                *((uint2x4*)&Hash[ 4]) = *(uint2x4*)&r8[ 4];
}
}

__host__
void xevan_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
        dim3 grid((threads + TPBE-1)/TPBE);
        dim3 block(TPBE);

        // note: 128 threads minimum are required to init the shared memory array
        xevan_echo512_gpu_hash_64<<<grid, block>>>(threads, (uint64_t*)d_hash);
}

#undef AES_ROUND_NOKEY
#undef KEY_EXPAND_ELT



__device__ __forceinline__
static void aes_round(const uint32_t sharedMemory[4][256],const uint32_t x0,const uint32_t x1,const uint32_t x2,const uint32_t x3,uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3){

	y0 = sharedMemory[0][__byte_perm(x0, 0, 0x4440)];
	y3 = sharedMemory[1][__byte_perm(x0, 0, 0x4441)];
	y2 = sharedMemory[2][__byte_perm(x0, 0, 0x4442)];
	y1 = __ldg(&AES3_C[__byte_perm(x0, 0, 0x4443)]);
	
	#ifdef INTENSIVE_GMF
	y1^= __ldg(&AES0_C[__byte_perm(x1, 0, 0x4440)]);
	#else
	y1^= sharedMemory[0][__byte_perm(x1, 0, 0x4440)];
	#endif
	y0^= sharedMemory[1][__byte_perm(x1, 0, 0x4441)];
	y3^= sharedMemory[2][__byte_perm(x1, 0, 0x4442)];
	y2^= __ldg(&AES3_C[__byte_perm(x1, 0, 0x4443)]);
	
	y2^= sharedMemory[0][__byte_perm(x2, 0, 0x4440)];
	y1^= sharedMemory[1][__byte_perm(x2, 0, 0x4441)];
	y0^= sharedMemory[2][__byte_perm(x2, 0, 0x4442)];
	y3^= __ldg(&AES3_C[__byte_perm(x2, 0, 0x4443)]);

	y3^= sharedMemory[0][__byte_perm(x3, 0, 0x4440)];
	y2^= sharedMemory[1][__byte_perm(x3, 0, 0x4441)];
	y1^= sharedMemory[2][__byte_perm(x3, 0, 0x4442)];
	y0^= __ldg(&AES3_C[__byte_perm(x3, 0, 0x4443)]);
}

__device__ __forceinline__
static void AES_ROUND_NOKEY(
	const uint32_t sharedMemory[4][256],
	uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3)
{
	uint32_t y0, y1, y2, y3;
	aes_round(sharedMemory,
		x0, x1, x2, x3,
		y0, y1, y2, y3);

	x0 = y0;
	x1 = y1;
	x2 = y2;
	x3 = y3;
}

__device__ __forceinline__
static void KEY_EXPAND_ELT(
	const uint32_t sharedMemory[4][256],
	uint32_t &k0, uint32_t &k1, uint32_t &k2, uint32_t &k3)
{
	uint32_t y0, y1, y2, y3;
	aes_round(sharedMemory,
		k0, k1, k2, k3,
		y0, y1, y2, y3);

	k0 = y1;
	k1 = y2;
	k2 = y3;
	k3 = y0;
}



__device__ __forceinline__
static void c512c(const uint32_t sharedMemory[4][256], uint32_t *state, uint32_t *msg, const uint32_t count)
{
	uint32_t p0, p1, p2, p3, p4, p5, p6, p7;
	uint32_t p8, p9, pA, pB, pC, pD, pE, pF;
	uint32_t x0, x1, x2, x3;
	uint32_t rk00, rk01, rk02, rk03, rk04, rk05, rk06, rk07;
	uint32_t rk08, rk09, rk0A, rk0B, rk0C, rk0D, rk0E, rk0F;
	uint32_t rk10, rk11, rk12, rk13, rk14, rk15, rk16, rk17;
	uint32_t rk18, rk19, rk1A, rk1B, rk1C, rk1D, rk1E, rk1F;
	const uint32_t counter = count;

	p0 = state[0x0];
	p1 = state[0x1];
	p2 = state[0x2];
	p3 = state[0x3];
	p4 = state[0x4];
	p5 = state[0x5];
	p6 = state[0x6];
	p7 = state[0x7];
	p8 = state[0x8];
	p9 = state[0x9];
	pA = state[0xA];
	pB = state[0xB];
	pC = state[0xC];
	pD = state[0xD];
	pE = state[0xE];
	pF = state[0xF];

	/* round 0 */
	rk00 = msg[0];
	x0 = p4 ^ msg[0];
	rk01 = msg[1];
	x1 = p5 ^ msg[1];
	rk02 = msg[2];
	x2 = p6 ^ msg[2];
	rk03 = msg[3];
	x3 = p7 ^ msg[3];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk04 = msg[4];
	x0 ^= msg[4];
	rk05 = msg[5];
	x1 ^= msg[5];
	rk06 = msg[6];
	x2 ^= msg[6];
	rk07 = msg[7];
	x3 ^= msg[7];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk08 = msg[8];
	x0 ^= msg[8];
	rk09 = msg[9];
	x1 ^= msg[9];
	rk0A = msg[10];
	x2 ^= msg[10];
	rk0B = msg[11];
	x3 ^= msg[11];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk0C = msg[12];
	x0 ^= msg[12];
	rk0D = msg[13];
	x1 ^= msg[13];
	rk0E = msg[14];
	x2 ^= msg[14];
	rk0F = msg[15];
	x3 ^= msg[15];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;
	if (count == 512)
	{
		rk10 = 0x80U;
		x0 = pC ^ 0x80U;
		rk11 = 0;
		x1 = pD;
		rk12 = 0;
		x2 = pE;
		rk13 = 0;
		x3 = pF;
		AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
		rk14 = 0;
		rk15 = 0;
		rk16 = 0;
		rk17 = 0;
		AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
		rk18 = 0;
		rk19 = 0;
		rk1A = 0;
		rk1B = 0x02000000U;
		x3 ^= 0x02000000U;
		AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
		rk1C = 0;
		rk1D = 0;
		rk1E = 0;
		rk1F = 0x02000000;
		x3 ^= 0x02000000;
	}
	else
	{
		rk10 = msg[16];
		x0 = pC ^ msg[16];
		rk11 = msg[17];
		x1 = pD ^ msg[17];
		rk12 = msg[18];
		x2 = pE ^ msg[18];
		rk13 = msg[19];
		x3 = pF ^ msg[19];
		AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
		rk14 = msg[20];
		x0 ^= msg[20];
		rk15 = msg[21];
		x1 ^= msg[21];
		rk16 = msg[22];
		x2 ^= msg[22];
		rk17 = msg[23];
		x3 ^= msg[23];
		AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
		rk18 = msg[24];
		x0 ^= msg[24];
		rk19 = msg[25];
		x1 ^= msg[25];
		rk1A = msg[26];
		x2 ^= msg[26];
		rk1B = msg[27];
		x3 ^= msg[27];
		AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
		rk1C = msg[28];
		x0 ^= msg[28];
		rk1D = msg[29];
		x1 ^= msg[29];
		rk1E = msg[30];
		x2 ^= msg[30];
		rk1F = msg[31];
		x3 ^= msg[31];
	}
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;

	// 1
	KEY_EXPAND_ELT(sharedMemory, rk00, rk01, rk02, rk03);
	rk00 ^= rk1C;
	rk01 ^= rk1D;
	rk02 ^= rk1E;
	rk03 ^= rk1F;
	rk00 ^= counter;
	rk03 ^= 0xFFFFFFFF;
	x0 = p0 ^ rk00;
	x1 = p1 ^ rk01;
	x2 = p2 ^ rk02;
	x3 = p3 ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk04, rk05, rk06, rk07);
	rk04 ^= rk00;
	rk05 ^= rk01;
	rk06 ^= rk02;
	rk07 ^= rk03;
	x0 ^= rk04;
	x1 ^= rk05;
	x2 ^= rk06;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk08, rk09, rk0A, rk0B);
	rk08 ^= rk04;
	rk09 ^= rk05;
	rk0A ^= rk06;
	rk0B ^= rk07;
	x0 ^= rk08;
	x1 ^= rk09;
	x2 ^= rk0A;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk0C, rk0D, rk0E, rk0F);
	rk0C ^= rk08;
	rk0D ^= rk09;
	rk0E ^= rk0A;
	rk0F ^= rk0B;
	x0 ^= rk0C;
	x1 ^= rk0D;
	x2 ^= rk0E;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk10, rk11, rk12, rk13);
	rk10 ^= rk0C;
	rk11 ^= rk0D;
	rk12 ^= rk0E;
	rk13 ^= rk0F;
	x0 = p8 ^ rk10;
	x1 = p9 ^ rk11;
	x2 = pA ^ rk12;
	x3 = pB ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk14, rk15, rk16, rk17);
	rk14 ^= rk10;
	rk15 ^= rk11;
	rk16 ^= rk12;
	rk17 ^= rk13;
	x0 ^= rk14;
	x1 ^= rk15;
	x2 ^= rk16;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk18, rk19, rk1A, rk1B);
	rk18 ^= rk14;
	rk19 ^= rk15;
	rk1A ^= rk16;
	rk1B ^= rk17;
	x0 ^= rk18;
	x1 ^= rk19;
	x2 ^= rk1A;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk1C, rk1D, rk1E, rk1F);
	rk1C ^= rk18;
	rk1D ^= rk19;
	rk1E ^= rk1A;
	rk1F ^= rk1B;
	x0 ^= rk1C;
	x1 ^= rk1D;
	x2 ^= rk1E;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;

	rk00 ^= rk19;
	x0 = pC ^ rk00;
	rk01 ^= rk1A;
	x1 = pD ^ rk01;
	rk02 ^= rk1B;
	x2 = pE ^ rk02;
	rk03 ^= rk1C;
	x3 = pF ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk04 ^= rk1D;
	x0 ^= rk04;
	rk05 ^= rk1E;
	x1 ^= rk05;
	rk06 ^= rk1F;
	x2 ^= rk06;
	rk07 ^= rk00;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk08 ^= rk01;
	x0 ^= rk08;
	rk09 ^= rk02;
	x1 ^= rk09;
	rk0A ^= rk03;
	x2 ^= rk0A;
	rk0B ^= rk04;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk0C ^= rk05;
	x0 ^= rk0C;
	rk0D ^= rk06;
	x1 ^= rk0D;
	rk0E ^= rk07;
	x2 ^= rk0E;
	rk0F ^= rk08;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;
	rk10 ^= rk09;
	x0 = p4 ^ rk10;
	rk11 ^= rk0A;
	x1 = p5 ^ rk11;
	rk12 ^= rk0B;
	x2 = p6 ^ rk12;
	rk13 ^= rk0C;
	x3 = p7 ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk14 ^= rk0D;
	x0 ^= rk14;
	rk15 ^= rk0E;
	x1 ^= rk15;
	rk16 ^= rk0F;
	x2 ^= rk16;
	rk17 ^= rk10;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk18 ^= rk11;
	x0 ^= rk18;
	rk19 ^= rk12;
	x1 ^= rk19;
	rk1A ^= rk13;
	x2 ^= rk1A;
	rk1B ^= rk14;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk1C ^= rk15;
	x0 ^= rk1C;
	rk1D ^= rk16;
	x1 ^= rk1D;
	rk1E ^= rk17;
	x2 ^= rk1E;
	rk1F ^= rk18;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;

	/* round 3, 7, 11 */
	KEY_EXPAND_ELT(sharedMemory, rk00, rk01, rk02, rk03);
	rk00 ^= rk1C;
	rk01 ^= rk1D;
	rk02 ^= rk1E;
	rk03 ^= rk1F;
	x0 = p8 ^ rk00;
	x1 = p9 ^ rk01;
	x2 = pA ^ rk02;
	x3 = pB ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk04, rk05, rk06, rk07);
	rk04 ^= rk00;
	rk05 ^= rk01;
	rk06 ^= rk02;
	rk07 ^= rk03;
	x0 ^= rk04;
	x1 ^= rk05;
	x2 ^= rk06;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk08, rk09, rk0A, rk0B);
	rk08 ^= rk04;
	rk09 ^= rk05;
	rk0A ^= rk06;
	rk0B ^= rk07;
	x0 ^= rk08;
	x1 ^= rk09;
	x2 ^= rk0A;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk0C, rk0D, rk0E, rk0F);
	rk0C ^= rk08;
	rk0D ^= rk09;
	rk0E ^= rk0A;
	rk0F ^= rk0B;
	x0 ^= rk0C;
	x1 ^= rk0D;
	x2 ^= rk0E;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk10, rk11, rk12, rk13);
	rk10 ^= rk0C;
	rk11 ^= rk0D;
	rk12 ^= rk0E;
	rk13 ^= rk0F;
	x0 = p0 ^ rk10;
	x1 = p1 ^ rk11;
	x2 = p2 ^ rk12;
	x3 = p3 ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk14, rk15, rk16, rk17);
	rk14 ^= rk10;
	rk15 ^= rk11;
	rk16 ^= rk12;
	rk17 ^= rk13;
	x0 ^= rk14;
	x1 ^= rk15;
	x2 ^= rk16;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk18, rk19, rk1A, rk1B);
	rk18 ^= rk14;
	rk19 ^= rk15;
	rk1A ^= rk16;
	rk1B ^= rk17;
	x0 ^= rk18;
	x1 ^= rk19;
	x2 ^= rk1A;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk1C, rk1D, rk1E, rk1F);
	rk1C ^= rk18;
	rk1D ^= rk19;
	rk1E ^= rk1A;
	rk1F ^= rk1B;
	x0 ^= rk1C;
	x1 ^= rk1D;
	x2 ^= rk1E;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;

	/* round 4, 8, 12 */
	rk00 ^= rk19;
	x0 = p4 ^ rk00;
	rk01 ^= rk1A;
	x1 = p5 ^ rk01;
	rk02 ^= rk1B;
	x2 = p6 ^ rk02;
	rk03 ^= rk1C;
	x3 = p7 ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk04 ^= rk1D;
	x0 ^= rk04;
	rk05 ^= rk1E;
	x1 ^= rk05;
	rk06 ^= rk1F;
	x2 ^= rk06;
	rk07 ^= rk00;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk08 ^= rk01;
	x0 ^= rk08;
	rk09 ^= rk02;
	x1 ^= rk09;
	rk0A ^= rk03;
	x2 ^= rk0A;
	rk0B ^= rk04;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk0C ^= rk05;
	x0 ^= rk0C;
	rk0D ^= rk06;
	x1 ^= rk0D;
	rk0E ^= rk07;
	x2 ^= rk0E;
	rk0F ^= rk08;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;
	rk10 ^= rk09;
	x0 = pC ^ rk10;
	rk11 ^= rk0A;
	x1 = pD ^ rk11;
	rk12 ^= rk0B;
	x2 = pE ^ rk12;
	rk13 ^= rk0C;
	x3 = pF ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk14 ^= rk0D;
	x0 ^= rk14;
	rk15 ^= rk0E;
	x1 ^= rk15;
	rk16 ^= rk0F;
	x2 ^= rk16;
	rk17 ^= rk10;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk18 ^= rk11;
	x0 ^= rk18;
	rk19 ^= rk12;
	x1 ^= rk19;
	rk1A ^= rk13;
	x2 ^= rk1A;
	rk1B ^= rk14;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk1C ^= rk15;
	x0 ^= rk1C;
	rk1D ^= rk16;
	x1 ^= rk1D;
	rk1E ^= rk17;
	x2 ^= rk1E;
	rk1F ^= rk18;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;

	// 2
	KEY_EXPAND_ELT(sharedMemory, rk00, rk01, rk02, rk03);
	rk00 ^= rk1C;
	rk01 ^= rk1D;
	rk02 ^= rk1E;
	rk03 ^= rk1F;
	x0 = p0 ^ rk00;
	x1 = p1 ^ rk01;
	x2 = p2 ^ rk02;
	x3 = p3 ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk04, rk05, rk06, rk07);
	rk04 ^= rk00;
	rk05 ^= rk01;
	rk06 ^= rk02;
	rk07 ^= rk03;
	rk07 ^= SPH_T32(~counter);
	x0 ^= rk04;
	x1 ^= rk05;
	x2 ^= rk06;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk08, rk09, rk0A, rk0B);
	rk08 ^= rk04;
	rk09 ^= rk05;
	rk0A ^= rk06;
	rk0B ^= rk07;
	x0 ^= rk08;
	x1 ^= rk09;
	x2 ^= rk0A;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk0C, rk0D, rk0E, rk0F);
	rk0C ^= rk08;
	rk0D ^= rk09;
	rk0E ^= rk0A;
	rk0F ^= rk0B;
	x0 ^= rk0C;
	x1 ^= rk0D;
	x2 ^= rk0E;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk10, rk11, rk12, rk13);
	rk10 ^= rk0C;
	rk11 ^= rk0D;
	rk12 ^= rk0E;
	rk13 ^= rk0F;
	x0 = p8 ^ rk10;
	x1 = p9 ^ rk11;
	x2 = pA ^ rk12;
	x3 = pB ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk14, rk15, rk16, rk17);
	rk14 ^= rk10;
	rk15 ^= rk11;
	rk16 ^= rk12;
	rk17 ^= rk13;
	x0 ^= rk14;
	x1 ^= rk15;
	x2 ^= rk16;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk18, rk19, rk1A, rk1B);
	rk18 ^= rk14;
	rk19 ^= rk15;
	rk1A ^= rk16;
	rk1B ^= rk17;
	x0 ^= rk18;
	x1 ^= rk19;
	x2 ^= rk1A;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk1C, rk1D, rk1E, rk1F);
	rk1C ^= rk18;
	rk1D ^= rk19;
	rk1E ^= rk1A;
	rk1F ^= rk1B;
	x0 ^= rk1C;
	x1 ^= rk1D;
	x2 ^= rk1E;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;

	rk00 ^= rk19;
	x0 = pC ^ rk00;
	rk01 ^= rk1A;
	x1 = pD ^ rk01;
	rk02 ^= rk1B;
	x2 = pE ^ rk02;
	rk03 ^= rk1C;
	x3 = pF ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk04 ^= rk1D;
	x0 ^= rk04;
	rk05 ^= rk1E;
	x1 ^= rk05;
	rk06 ^= rk1F;
	x2 ^= rk06;
	rk07 ^= rk00;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk08 ^= rk01;
	x0 ^= rk08;
	rk09 ^= rk02;
	x1 ^= rk09;
	rk0A ^= rk03;
	x2 ^= rk0A;
	rk0B ^= rk04;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk0C ^= rk05;
	x0 ^= rk0C;
	rk0D ^= rk06;
	x1 ^= rk0D;
	rk0E ^= rk07;
	x2 ^= rk0E;
	rk0F ^= rk08;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;
	rk10 ^= rk09;
	x0 = p4 ^ rk10;
	rk11 ^= rk0A;
	x1 = p5 ^ rk11;
	rk12 ^= rk0B;
	x2 = p6 ^ rk12;
	rk13 ^= rk0C;
	x3 = p7 ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk14 ^= rk0D;
	x0 ^= rk14;
	rk15 ^= rk0E;
	x1 ^= rk15;
	rk16 ^= rk0F;
	x2 ^= rk16;
	rk17 ^= rk10;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk18 ^= rk11;
	x0 ^= rk18;
	rk19 ^= rk12;
	x1 ^= rk19;
	rk1A ^= rk13;
	x2 ^= rk1A;
	rk1B ^= rk14;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk1C ^= rk15;
	x0 ^= rk1C;
	rk1D ^= rk16;
	x1 ^= rk1D;
	rk1E ^= rk17;
	x2 ^= rk1E;
	rk1F ^= rk18;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;

	/* round 3, 7, 11 */
	KEY_EXPAND_ELT(sharedMemory, rk00, rk01, rk02, rk03);
	rk00 ^= rk1C;
	rk01 ^= rk1D;
	rk02 ^= rk1E;
	rk03 ^= rk1F;
	x0 = p8 ^ rk00;
	x1 = p9 ^ rk01;
	x2 = pA ^ rk02;
	x3 = pB ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk04, rk05, rk06, rk07);
	rk04 ^= rk00;
	rk05 ^= rk01;
	rk06 ^= rk02;
	rk07 ^= rk03;
	x0 ^= rk04;
	x1 ^= rk05;
	x2 ^= rk06;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk08, rk09, rk0A, rk0B);
	rk08 ^= rk04;
	rk09 ^= rk05;
	rk0A ^= rk06;
	rk0B ^= rk07;
	x0 ^= rk08;
	x1 ^= rk09;
	x2 ^= rk0A;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk0C, rk0D, rk0E, rk0F);
	rk0C ^= rk08;
	rk0D ^= rk09;
	rk0E ^= rk0A;
	rk0F ^= rk0B;
	x0 ^= rk0C;
	x1 ^= rk0D;
	x2 ^= rk0E;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk10, rk11, rk12, rk13);
	rk10 ^= rk0C;
	rk11 ^= rk0D;
	rk12 ^= rk0E;
	rk13 ^= rk0F;
	x0 = p0 ^ rk10;
	x1 = p1 ^ rk11;
	x2 = p2 ^ rk12;
	x3 = p3 ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk14, rk15, rk16, rk17);
	rk14 ^= rk10;
	rk15 ^= rk11;
	rk16 ^= rk12;
	rk17 ^= rk13;
	x0 ^= rk14;
	x1 ^= rk15;
	x2 ^= rk16;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk18, rk19, rk1A, rk1B);
	rk18 ^= rk14;
	rk19 ^= rk15;
	rk1A ^= rk16;
	rk1B ^= rk17;
	x0 ^= rk18;
	x1 ^= rk19;
	x2 ^= rk1A;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk1C, rk1D, rk1E, rk1F);
	rk1C ^= rk18;
	rk1D ^= rk19;
	rk1E ^= rk1A;
	rk1F ^= rk1B;
	x0 ^= rk1C;
	x1 ^= rk1D;
	x2 ^= rk1E;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;

	/* round 4, 8, 12 */
	rk00 ^= rk19;
	x0 = p4 ^ rk00;
	rk01 ^= rk1A;
	x1 = p5 ^ rk01;
	rk02 ^= rk1B;
	x2 = p6 ^ rk02;
	rk03 ^= rk1C;
	x3 = p7 ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk04 ^= rk1D;
	x0 ^= rk04;
	rk05 ^= rk1E;
	x1 ^= rk05;
	rk06 ^= rk1F;
	x2 ^= rk06;
	rk07 ^= rk00;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk08 ^= rk01;
	x0 ^= rk08;
	rk09 ^= rk02;
	x1 ^= rk09;
	rk0A ^= rk03;
	x2 ^= rk0A;
	rk0B ^= rk04;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk0C ^= rk05;
	x0 ^= rk0C;
	rk0D ^= rk06;
	x1 ^= rk0D;
	rk0E ^= rk07;
	x2 ^= rk0E;
	rk0F ^= rk08;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;
	rk10 ^= rk09;
	x0 = pC ^ rk10;
	rk11 ^= rk0A;
	x1 = pD ^ rk11;
	rk12 ^= rk0B;
	x2 = pE ^ rk12;
	rk13 ^= rk0C;
	x3 = pF ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk14 ^= rk0D;
	x0 ^= rk14;
	rk15 ^= rk0E;
	x1 ^= rk15;
	rk16 ^= rk0F;
	x2 ^= rk16;
	rk17 ^= rk10;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk18 ^= rk11;
	x0 ^= rk18;
	rk19 ^= rk12;
	x1 ^= rk19;
	rk1A ^= rk13;
	x2 ^= rk1A;
	rk1B ^= rk14;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk1C ^= rk15;
	x0 ^= rk1C;
	rk1D ^= rk16;
	x1 ^= rk1D;
	rk1E ^= rk17;
	x2 ^= rk1E;
	rk1F ^= rk18;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;

	// 3
	KEY_EXPAND_ELT(sharedMemory, rk00, rk01, rk02, rk03);
	rk00 ^= rk1C;
	rk01 ^= rk1D;
	rk02 ^= rk1E;
	rk03 ^= rk1F;
	x0 = p0 ^ rk00;
	x1 = p1 ^ rk01;
	x2 = p2 ^ rk02;
	x3 = p3 ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk04, rk05, rk06, rk07);
	rk04 ^= rk00;
	rk05 ^= rk01;
	rk06 ^= rk02;
	rk07 ^= rk03;
	x0 ^= rk04;
	x1 ^= rk05;
	x2 ^= rk06;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk08, rk09, rk0A, rk0B);
	rk08 ^= rk04;
	rk09 ^= rk05;
	rk0A ^= rk06;
	rk0B ^= rk07;
	x0 ^= rk08;
	x1 ^= rk09;
	x2 ^= rk0A;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk0C, rk0D, rk0E, rk0F);
	rk0C ^= rk08;
	rk0D ^= rk09;
	rk0E ^= rk0A;
	rk0F ^= rk0B;
	x0 ^= rk0C;
	x1 ^= rk0D;
	x2 ^= rk0E;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk10, rk11, rk12, rk13);
	rk10 ^= rk0C;
	rk11 ^= rk0D;
	rk12 ^= rk0E;
	rk13 ^= rk0F;
	x0 = p8 ^ rk10;
	x1 = p9 ^ rk11;
	x2 = pA ^ rk12;
	x3 = pB ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk14, rk15, rk16, rk17);
	rk14 ^= rk10;
	rk15 ^= rk11;
	rk16 ^= rk12;
	rk17 ^= rk13;
	x0 ^= rk14;
	x1 ^= rk15;
	x2 ^= rk16;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk18, rk19, rk1A, rk1B);
	rk18 ^= rk14;
	rk19 ^= rk15;
	rk1A ^= rk16;
	rk1B ^= rk17;
	x0 ^= rk18;
	x1 ^= rk19;
	x2 ^= rk1A;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk1C, rk1D, rk1E, rk1F);
	rk1C ^= rk18;
	rk1D ^= rk19;
	rk1E ^= rk1A;
	rk1F ^= rk1B;
	rk1E ^= counter;
	rk1F ^= 0xFFFFFFFF;
	x0 ^= rk1C;
	x1 ^= rk1D;
	x2 ^= rk1E;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;

	rk00 ^= rk19;
	x0 = pC ^ rk00;
	rk01 ^= rk1A;
	x1 = pD ^ rk01;
	rk02 ^= rk1B;
	x2 = pE ^ rk02;
	rk03 ^= rk1C;
	x3 = pF ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk04 ^= rk1D;
	x0 ^= rk04;
	rk05 ^= rk1E;
	x1 ^= rk05;
	rk06 ^= rk1F;
	x2 ^= rk06;
	rk07 ^= rk00;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk08 ^= rk01;
	x0 ^= rk08;
	rk09 ^= rk02;
	x1 ^= rk09;
	rk0A ^= rk03;
	x2 ^= rk0A;
	rk0B ^= rk04;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk0C ^= rk05;
	x0 ^= rk0C;
	rk0D ^= rk06;
	x1 ^= rk0D;
	rk0E ^= rk07;
	x2 ^= rk0E;
	rk0F ^= rk08;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;
	rk10 ^= rk09;
	x0 = p4 ^ rk10;
	rk11 ^= rk0A;
	x1 = p5 ^ rk11;
	rk12 ^= rk0B;
	x2 = p6 ^ rk12;
	rk13 ^= rk0C;
	x3 = p7 ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk14 ^= rk0D;
	x0 ^= rk14;
	rk15 ^= rk0E;
	x1 ^= rk15;
	rk16 ^= rk0F;
	x2 ^= rk16;
	rk17 ^= rk10;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk18 ^= rk11;
	x0 ^= rk18;
	rk19 ^= rk12;
	x1 ^= rk19;
	rk1A ^= rk13;
	x2 ^= rk1A;
	rk1B ^= rk14;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk1C ^= rk15;
	x0 ^= rk1C;
	rk1D ^= rk16;
	x1 ^= rk1D;
	rk1E ^= rk17;
	x2 ^= rk1E;
	rk1F ^= rk18;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;

	/* round 3, 7, 11 */
	KEY_EXPAND_ELT(sharedMemory, rk00, rk01, rk02, rk03);
	rk00 ^= rk1C;
	rk01 ^= rk1D;
	rk02 ^= rk1E;
	rk03 ^= rk1F;
	x0 = p8 ^ rk00;
	x1 = p9 ^ rk01;
	x2 = pA ^ rk02;
	x3 = pB ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk04, rk05, rk06, rk07);
	rk04 ^= rk00;
	rk05 ^= rk01;
	rk06 ^= rk02;
	rk07 ^= rk03;
	x0 ^= rk04;
	x1 ^= rk05;
	x2 ^= rk06;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk08, rk09, rk0A, rk0B);
	rk08 ^= rk04;
	rk09 ^= rk05;
	rk0A ^= rk06;
	rk0B ^= rk07;
	x0 ^= rk08;
	x1 ^= rk09;
	x2 ^= rk0A;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk0C, rk0D, rk0E, rk0F);
	rk0C ^= rk08;
	rk0D ^= rk09;
	rk0E ^= rk0A;
	rk0F ^= rk0B;
	x0 ^= rk0C;
	x1 ^= rk0D;
	x2 ^= rk0E;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk10, rk11, rk12, rk13);
	rk10 ^= rk0C;
	rk11 ^= rk0D;
	rk12 ^= rk0E;
	rk13 ^= rk0F;
	x0 = p0 ^ rk10;
	x1 = p1 ^ rk11;
	x2 = p2 ^ rk12;
	x3 = p3 ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk14, rk15, rk16, rk17);
	rk14 ^= rk10;
	rk15 ^= rk11;
	rk16 ^= rk12;
	rk17 ^= rk13;
	x0 ^= rk14;
	x1 ^= rk15;
	x2 ^= rk16;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk18, rk19, rk1A, rk1B);
	rk18 ^= rk14;
	rk19 ^= rk15;
	rk1A ^= rk16;
	rk1B ^= rk17;
	x0 ^= rk18;
	x1 ^= rk19;
	x2 ^= rk1A;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk1C, rk1D, rk1E, rk1F);
	rk1C ^= rk18;
	rk1D ^= rk19;
	rk1E ^= rk1A;
	rk1F ^= rk1B;
	x0 ^= rk1C;
	x1 ^= rk1D;
	x2 ^= rk1E;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	/* round 4, 8, 12 */
	rk00 ^= rk19;
	x0 = p4 ^ rk00;
	rk01 ^= rk1A;
	x1 = p5 ^ rk01;
	rk02 ^= rk1B;
	x2 = p6 ^ rk02;
	rk03 ^= rk1C;
	x3 = p7 ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk04 ^= rk1D;
	x0 ^= rk04;
	rk05 ^= rk1E;
	x1 ^= rk05;
	rk06 ^= rk1F;
	x2 ^= rk06;
	rk07 ^= rk00;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk08 ^= rk01;
	x0 ^= rk08;
	rk09 ^= rk02;
	x1 ^= rk09;
	rk0A ^= rk03;
	x2 ^= rk0A;
	rk0B ^= rk04;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk0C ^= rk05;
	x0 ^= rk0C;
	rk0D ^= rk06;
	x1 ^= rk0D;
	rk0E ^= rk07;
	x2 ^= rk0E;
	rk0F ^= rk08;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;
	rk10 ^= rk09;
	x0 = pC ^ rk10;
	rk11 ^= rk0A;
	x1 = pD ^ rk11;
	rk12 ^= rk0B;
	x2 = pE ^ rk12;
	rk13 ^= rk0C;
	x3 = pF ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk14 ^= rk0D;
	x0 ^= rk14;
	rk15 ^= rk0E;
	x1 ^= rk15;
	rk16 ^= rk0F;
	x2 ^= rk16;
	rk17 ^= rk10;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk18 ^= rk11;
	x0 ^= rk18;
	rk19 ^= rk12;
	x1 ^= rk19;
	rk1A ^= rk13;
	x2 ^= rk1A;
	rk1B ^= rk14;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk1C ^= rk15;
	x0 ^= rk1C;
	rk1D ^= rk16;
	x1 ^= rk1D;
	rk1E ^= rk17;
	x2 ^= rk1E;
	rk1F ^= rk18;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;

	/* round 13 */
	KEY_EXPAND_ELT(sharedMemory, rk00, rk01, rk02, rk03);
	rk00 ^= rk1C;
	rk01 ^= rk1D;
	rk02 ^= rk1E;
	rk03 ^= rk1F;
	x0 = p0 ^ rk00;
	x1 = p1 ^ rk01;
	x2 = p2 ^ rk02;
	x3 = p3 ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk04, rk05, rk06, rk07);
	rk04 ^= rk00;
	rk05 ^= rk01;
	rk06 ^= rk02;
	rk07 ^= rk03;
	x0 ^= rk04;
	x1 ^= rk05;
	x2 ^= rk06;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk08, rk09, rk0A, rk0B);
	rk08 ^= rk04;
	rk09 ^= rk05;
	rk0A ^= rk06;
	rk0B ^= rk07;
	x0 ^= rk08;
	x1 ^= rk09;
	x2 ^= rk0A;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk0C, rk0D, rk0E, rk0F);
	rk0C ^= rk08;
	rk0D ^= rk09;
	rk0E ^= rk0A;
	rk0F ^= rk0B;
	x0 ^= rk0C;
	x1 ^= rk0D;
	x2 ^= rk0E;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk10, rk11, rk12, rk13);
	rk10 ^= rk0C;
	rk11 ^= rk0D;
	rk12 ^= rk0E;
	rk13 ^= rk0F;
	x0 = p8 ^ rk10;
	x1 = p9 ^ rk11;
	x2 = pA ^ rk12;
	x3 = pB ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk14, rk15, rk16, rk17);
	rk14 ^= rk10;
	rk15 ^= rk11;
	rk16 ^= rk12;
	rk17 ^= rk13;
	x0 ^= rk14;
	x1 ^= rk15;
	x2 ^= rk16;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk18, rk19, rk1A, rk1B);
	rk18 ^= rk14;
	rk19 ^= rk15 ^ counter;
	rk1A ^= rk16;
	rk1B ^= rk17 ^ 0xFFFFFFFF;
	x0 ^= rk18;
	x1 ^= rk19;
	x2 ^= rk1A;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk1C, rk1D, rk1E, rk1F);
	rk1C ^= rk18;
	rk1D ^= rk19;
	rk1E ^= rk1A;
	rk1F ^= rk1B;
	x0 ^= rk1C;
	x1 ^= rk1D;
	x2 ^= rk1E;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;
	state[0x0] ^= p8;
	state[0x1] ^= p9;
	state[0x2] ^= pA;
	state[0x3] ^= pB;
	state[0x4] ^= pC;
	state[0x5] ^= pD;
	state[0x6] ^= pE;
	state[0x7] ^= pF;
	state[0x8] ^= p0;
	state[0x9] ^= p1;
	state[0xA] ^= p2;
	state[0xB] ^= p3;
	state[0xC] ^= p4;
	state[0xD] ^= p5;
	state[0xE] ^= p6;
	state[0xF] ^= p7;
}

/*
__global__ __launch_bounds__(TPBS,2)
void xevan_shavite512_gpu_hash_128(const uint32_t threads, uint64_t *g_hash)
{
const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
        if (thread < threads)
        {
                uint32_t msg[16];
                uint64_t *r8=(uint64_t *)r;
                uint64_t *Hash = &g_hash[thread<<3];

__shared__ uint32_t AES0_S[256];
__shared__ uint32_t AES1_S[256];
__shared__ uint32_t AES2_S[256];
//__shared__ uint32_t AES3_S[256];
        if(threadIdx.x < 256){
//      uint32_t temp = __ldg(&AES0_C[threadIdx.x]);
        uint32_t temp = AES0_C[threadIdx.x];
        AES0_S[threadIdx.x] = temp;
        AES1_S[threadIdx.x] = ROL8(temp);
        AES2_S[threadIdx.x] = ROL16(temp);
//        AES3_S[threadIdx.x] = ROR8(temp);
}
                *(uint2x4*)&r[ 0] = __ldg4((uint2x4*)&Hash[ 0]);
                *(uint2x4*)&r[ 8] = __ldg4((uint2x4*)&Hash[ 4]);


	}
}

*/
#endif


