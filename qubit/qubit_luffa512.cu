/*******************************************************************************
 * luffa512 for 80-bytes input (with midstate precalc based on the work of klausT and SP)
 */

#include <miner.h>
#include "cuda_helper.h"
#include "cuda_vectors.h"

static unsigned char PaddedMessage[128];
__constant__ uint64_t c_PaddedMessage80[10]; // padded message (80 bytes + padding)
__constant__ uint32_t _ALIGN(8) statebufferpre[8];
__constant__ uint32_t _ALIGN(8) statechainvpre[40];



#define MULT0(a) {\
	tmp = a[7]; \
	a[7] = a[6]; \
	a[6] = a[5]; \
	a[5] = a[4]; \
	a[4] = a[3] ^ tmp; \
	a[3] = a[2] ^ tmp; \
	a[2] = a[1]; \
	a[1] = a[0] ^ tmp; \
	a[0] = tmp; \
}

#define MULT2(a,j) {\
	tmp = a[7+(8*j)];\
	a[7+(8*j)] = a[6+(8*j)];\
	a[6+(8*j)] = a[5+(8*j)];\
	a[5+(8*j)] = a[4+(8*j)];\
	a[4+(8*j)] = a[3+(8*j)] ^ tmp;\
	a[3+(8*j)] = a[2+(8*j)] ^ tmp;\
	a[2+(8*j)] = a[1+(8*j)];\
	a[1+(8*j)] = a[0+(8*j)] ^ tmp;\
	a[0+(8*j)] = tmp;\
}

#define TWEAK(a0,a1,a2,a3,j)\
	a0 = ROTL32(a0,j);\
	a1 = ROTL32(a1,j);\
	a2 = ROTL32(a2,j);\
	a3 = ROTL32(a3,j);

#define STEP(c0,c1) {\
\
	uint32_t temp[ 2];\
	temp[ 0]  = chainv[0];\
	temp[ 1]  = chainv[ 5];\
	chainv[ 2] ^= chainv[ 3];\
	chainv[ 7] ^= chainv[ 4];\
	chainv[ 0] |= chainv[ 1];\
	chainv[ 5] |= chainv[ 6];\
	chainv[ 1]  = ~chainv[ 1];\
	chainv[ 6]  = ~chainv[ 6];\
	chainv[ 0] ^= chainv[ 3];\
	chainv[ 5] ^= chainv[ 4];\
	chainv[ 3] &= temp[ 0];\
	chainv[ 4] &= temp[ 1];\
	chainv[ 1] ^= chainv[ 3];\
	chainv[ 6] ^= chainv[ 4];\
	chainv[ 3] ^= chainv[ 2];\
	chainv[ 4] ^= chainv[ 7];\
	chainv[ 2] &= chainv[ 0];\
	chainv[ 7] &= chainv[ 5];\
	chainv[ 0]  = ~chainv[ 0];\
	chainv[ 5]  = ~chainv[ 5];\
	chainv[ 2] ^= chainv[ 1];\
	chainv[ 7] ^= chainv[ 6];\
	chainv[ 1] |= chainv[ 3];\
	chainv[ 6] |= chainv[ 4];\
	temp[ 0] ^= chainv[ 1];\
	temp[ 1] ^= chainv[ 6];\
	chainv[ 3] ^= chainv[ 2];\
	chainv[ 4] ^= chainv[ 7] ^ temp[ 0];\
	chainv[ 2] &= chainv[ 1];\
	chainv[ 7]  = (chainv[ 7] & chainv[ 6]) ^ chainv[ 3];\
	chainv[ 1] ^= chainv[ 0];\
	chainv[ 6] ^= chainv[ 5] ^ chainv[ 2];\
	chainv[ 5]  = chainv[ 1] ^ temp[ 1];\
	chainv[ 0]  = chainv[ 4] ^ ROTL32(temp[ 0],2); \
	chainv[ 1]  = chainv[ 5] ^ ROTL32(chainv[ 1],2); \
	chainv[ 2]  = chainv[ 6] ^ ROTL32(chainv[ 2],2); \
	chainv[ 3]  = chainv[ 7] ^ ROTL32(chainv[ 3],2); \
	chainv[ 4]  = chainv[ 0] ^ ROTL32(chainv[ 4],14); \
	chainv[ 5]  = chainv[ 1] ^ ROTL32(chainv[ 5],14); \
	chainv[ 6]  = chainv[ 2] ^ ROTL32(chainv[ 6],14); \
	chainv[ 7]  = chainv[ 3] ^ ROTL32(chainv[ 7],14); \
	chainv[ 0]  = chainv[ 4] ^ ROTL32(chainv[ 0],10) ^ c0; \
	chainv[ 1]  = chainv[ 5] ^ ROTL32(chainv[ 1],10); \
	chainv[ 2]  = chainv[ 6] ^ ROTL32(chainv[ 2],10); \
	chainv[ 3]  = chainv[ 7] ^ ROTL32(chainv[ 3],10); \
	chainv[ 4]  = ROTL32(chainv[ 4],1) ^ c1; \
	chainv[ 5]  = ROTL32(chainv[ 5],1); \
	chainv[ 6]  = ROTL32(chainv[ 6],1); \
	chainv[ 7]  = ROTL32(chainv[ 7],1); \
}

__device__ __forceinline__
void STEP2(uint32_t *t, const uint2 c0, const uint2 c1){
	uint32_t temp[ 4];
	temp[ 0] = t[ 0];
	temp[ 1] = t[ 5];	
	temp[ 2] = t[0+8];
	temp[ 3] = t[8+5];
	t[ 2] ^= t[ 3];
	t[ 7] ^= t[ 4];		
	t[8+2] ^= t[8+3];
	t[8+7] ^= t[8+4];
	t[ 0] |= t[ 1];	
	t[ 5] |= t[ 6];
	t[8+0]|= t[8+1];
	t[8+5]|= t[8+6];
	t[ 1]  = ~t[ 1];
	t[ 6]  = ~t[ 6];
	t[8+1] = ~t[8+1];
	t[8+6] = ~t[8+6];
	t[ 0] ^= t[ 3];
	t[ 5] ^= t[ 4];
	t[8+0]^= t[8+3];	
	t[8+5]^= t[8+4];
	t[ 3] &= temp[ 0];
	t[ 4] &= temp[ 1];
	t[8+3]&= temp[ 2];
	t[8+4]&= temp[ 3];
	t[ 1] ^= t[ 3];	
	t[ 6] ^= t[ 4];
	t[8+1]^= t[8+3];
	t[8+6]^= t[8+4];
	t[ 3] ^= t[ 2];	
	t[ 4] ^= t[ 7];	
	t[8+3]^= t[8+2];
	t[8+4]^= t[8+7];
	t[ 2] &= t[ 0];	
	t[ 7] &= t[ 5];	
	t[8+2]&= t[8+0];
	t[8+7]&= t[8+5];
	t[ 0]  = ~t[ 0];
	t[ 5]  = ~t[ 5];
	t[8+0] = ~t[8+0];
	t[8+5] = ~t[8+5];
	t[ 2] ^= t[ 1];
	t[ 7] ^= t[ 6];
	t[8+2]^= t[8+1];
	t[8+7]^= t[8+6];
	t[ 1] |= t[ 3];	
	t[ 6] |= t[ 4];	
	t[8+1]|= t[8+3];
	t[8+6]|= t[8+4];
	
	temp[ 0] ^= t[ 1];
	temp[ 1] ^= t[ 6];
	temp[ 2] ^= t[8+1];
	temp[ 3] ^= t[8+6];
	
	t[ 3] ^= t[ 2];
	t[ 4] ^= t[ 7] ^ temp[ 0];
	t[8+3]^= t[8+2];	
	t[8+4]^= t[8+7] ^ temp[ 2];
	t[ 2] &= t[ 1];
	t[ 7]  = (t[ 7] & t[ 6]) ^ t[ 3];
	t[8+2]&= t[8+1];
	t[ 1] ^= t[ 0];
	t[8+7] = (t[8+6] & t[8+7]) ^ t[8+3];
	t[ 6] ^= t[ 5] ^ t[ 2];		
	t[8+1]^= t[8+0];	
	t[8+6]^= t[8+2]^ t[8+5];
	t[ 5]  = t[ 1] ^ temp[ 1];
	t[ 0]  = t[ 4] ^ ROTL32(temp[ 0],2);
	t[8+5] = t[8+1]^ temp[ 3];	
	t[8+0] = t[8+4]^ ROTL32(temp[ 2],2);
	t[ 1]  = t[ 5] ^ ROTL32(t[ 1],2);
	t[ 2]  = t[ 6] ^ ROTL32(t[ 2],2);
	t[8+1] = t[8+5]^ ROTL32(t[8+1],2);
	t[8+2] = t[8+6]^ ROTL32(t[8+2],2);
	t[ 3]  = t[ 7] ^ ROTL32(t[ 3],2);
	t[ 4]  = t[ 0] ^ ROTL32(t[ 4],14);
	t[8+3] = t[8+7] ^ ROTL32(t[8+3],2);
	t[8+4] = t[8+0] ^ ROTL32(t[8+4],14);
	t[ 5]  = t[ 1] ^ ROTL32(t[ 5],14);
	t[ 6]  = t[ 2] ^ ROTL32(t[ 6],14);
	t[8+5] = t[8+1] ^ ROTL32(t[8+5],14);
	t[8+6] = t[8+2] ^ ROTL32(t[8+6],14);
	t[ 7]  = t[ 3] ^ ROTL32(t[ 7],14);
	t[ 0]  = t[ 4] ^ ROTL32(t[ 0],10) ^ c0.x;
	t[8+7] = t[8+3]^ ROTL32(t[8+7],14);
	t[8+0] = t[8+4]^ ROTL32(t[8+0],10) ^ c1.x;
	t[ 1]  = t[ 5] ^ ROTL32(t[ 1],10);
	t[ 2]  = t[ 6] ^ ROTL32(t[ 2],10);
	t[8+1] = t[8+5]^ ROTL32(t[8+1],10);
	t[8+2] = t[8+6]^ ROTL32(t[8+2],10);
	t[ 3]  = t[ 7] ^ ROTL32(t[ 3],10);
	t[ 4]  = ROTL32(t[ 4],1) ^ c0.y;
	t[8+3] = t[8+7] ^ ROTL32(t[8+3],10);
	t[8+4] = ROTL32(t[8+4],1) ^ c1.y;
	t[ 5]  = ROTL32(t[ 5],1);
	t[ 6]  = ROTL32(t[ 6],1);
	t[8+5] = ROTL32(t[8+5],1);
	t[8+6] = ROTL32(t[8+6],1);
	t[ 7]  = ROTL32(t[ 7],1);
	t[8+7] = ROTL32(t[8+7],1);
}

__device__ __forceinline__
void STEP1(uint32_t *t, const uint2 c){
	uint32_t temp[ 2];
	temp[ 0] = t[ 0];			temp[ 1] = t[ 5];
	t[ 2] ^= t[ 3];				t[ 7] ^= t[ 4];
	t[ 0] |= t[ 1];				t[ 5] |= t[ 6];
	t[ 1]  = ~t[ 1];			t[ 6]  = ~t[ 6];
	t[ 0] ^= t[ 3];				t[ 5] ^= t[ 4];
	t[ 3] &= temp[ 0];			t[ 4] &= temp[ 1];
	t[ 1] ^= t[ 3];				t[ 6] ^= t[ 4];
	t[ 3] ^= t[ 2];				t[ 4] ^= t[ 7];
	t[ 2] &= t[ 0];				t[ 7] &= t[ 5];
	t[ 0]  = ~t[ 0];			t[ 5]  = ~t[ 5];
	t[ 2] ^= t[ 1];				t[ 7] ^= t[ 6];
	t[ 1] |= t[ 3];				t[ 6] |= t[ 4];
	temp[ 0] ^= t[ 1];			temp[ 1] ^= t[ 6];
	t[ 3] ^= t[ 2];				t[ 4] ^= t[ 7] ^ temp[ 0];
	t[ 2] &= t[ 1];				t[ 7]  = (t[ 7] & t[ 6]) ^ t[ 3];
	t[ 1] ^= t[ 0];				t[ 6] ^= t[ 5] ^ t[ 2];
	t[ 5]  = t[ 1] ^ temp[ 1];		t[ 0]  = t[ 4] ^ ROTL32(temp[ 0],2);
	t[ 1]  = t[ 5] ^ ROTL32(t[ 1],2);	t[ 2]  = t[ 6] ^ ROTL32(t[ 2],2);
	t[ 3]  = t[ 7] ^ ROTL32(t[ 3],2);	t[ 4]  = t[ 0] ^ ROTL32(t[ 4],14);
	t[ 5]  = t[ 1] ^ ROTL32(t[ 5],14);	t[ 6]  = t[ 2] ^ ROTL32(t[ 6],14);
	t[ 7]  = t[ 3] ^ ROTL32(t[ 7],14);	t[ 0]  = t[ 4] ^ ROTL32(t[ 0],10) ^ c.x;
	t[ 1]  = t[ 5] ^ ROTL32(t[ 1],10);	t[ 2]  = t[ 6] ^ ROTL32(t[ 2],10);
	t[ 3]  = t[ 7] ^ ROTL32(t[ 3],10);	t[ 4]  = ROTL32(t[ 4],1) ^ c.y;
	t[ 5]  = ROTL32(t[ 5],1);		t[ 6]  = ROTL32(t[ 6],1);
						t[ 7]  = ROTL32(t[ 7],1);
}

__constant__ const uint32_t c_CNS[80] = {
		0x303994a6,0xe0337818,0xc0e65299,0x441ba90d, 0x6cc33a12,0x7f34d442,0xdc56983e,0x9389217f, 0x1e00108f,0xe5a8bce6,0x7800423d,0x5274baf4, 0x8f5b7882,0x26889ba7,0x96e1db12,0x9a226e9d,
		0xb6de10ed,0x01685f3d,0x70f47aae,0x05a17cf4, 0x0707a3d4,0xbd09caca,0x1c1e8f51,0xf4272b28, 0x707a3d45,0x144ae5cc,0xaeb28562,0xfaa7ae2b, 0xbaca1589,0x2e48f1c1,0x40a46f3e,0xb923c704,
		0xfc20d9d2,0xe25e72c1,0x34552e25,0xe623bb72, 0x7ad8818f,0x5c58a4a4,0x8438764a,0x1e38e2e7, 0xbb6de032,0x78e38b9d,0xedb780c8,0x27586719, 0xd9847356,0x36eda57f,0xa2c78434,0x703aace7,
		0xb213afa5,0xe028c9bf,0xc84ebe95,0x44756f91, 0x4e608a22,0x7e8fce32,0x56d858fe,0x956548be, 0x343b138f,0xfe191be2,0xd0ec4e3d,0x3cb226e5, 0x2ceb4882,0x5944a28e,0xb3ad2208,0xa1c4c355,
		0xf0d2e9e3,0x5090d577,0xac11d7fa,0x2d1925ab, 0x1bcb66f2,0xb46496ac,0x6f2d9bc9,0xd1925ab0, 0x78602649,0x29131ab6,0x8edae952,0x0fc053c3, 0x3b6ba548,0x3f014f0c,0xedae9520,0xfc053c31
	};

static uint32_t h_CNS[80] = {
		0x303994a6,0xe0337818,0xc0e65299,0x441ba90d, 0x6cc33a12,0x7f34d442,0xdc56983e,0x9389217f, 0x1e00108f,0xe5a8bce6,0x7800423d,0x5274baf4, 0x8f5b7882,0x26889ba7,0x96e1db12,0x9a226e9d,
		0xb6de10ed,0x01685f3d,0x70f47aae,0x05a17cf4, 0x0707a3d4,0xbd09caca,0x1c1e8f51,0xf4272b28, 0x707a3d45,0x144ae5cc,0xaeb28562,0xfaa7ae2b, 0xbaca1589,0x2e48f1c1,0x40a46f3e,0xb923c704,
		0xfc20d9d2,0xe25e72c1,0x34552e25,0xe623bb72, 0x7ad8818f,0x5c58a4a4,0x8438764a,0x1e38e2e7, 0xbb6de032,0x78e38b9d,0xedb780c8,0x27586719, 0xd9847356,0x36eda57f,0xa2c78434,0x703aace7,
		0xb213afa5,0xe028c9bf,0xc84ebe95,0x44756f91, 0x4e608a22,0x7e8fce32,0x56d858fe,0x956548be, 0x343b138f,0xfe191be2,0xd0ec4e3d,0x3cb226e5, 0x2ceb4882,0x5944a28e,0xb3ad2208,0xa1c4c355,
		0xf0d2e9e3,0x5090d577,0xac11d7fa,0x2d1925ab, 0x1bcb66f2,0xb46496ac,0x6f2d9bc9,0xd1925ab0, 0x78602649,0x29131ab6,0x8edae952,0x0fc053c3, 0x3b6ba548,0x3f014f0c,0xedae9520,0xfc053c31
	};


__device__
static void rnd512(uint32_t *const __restrict__ statebuffer, uint32_t *const __restrict__ statechainv){
	uint32_t t[40];
	uint32_t tmp;

	tmp = statechainv[ 7] ^ statechainv[7 + 8] ^ statechainv[7 +16] ^ statechainv[7 +24] ^ statechainv[7 +32];
	t[7] = statechainv[ 6] ^ statechainv[6 + 8] ^ statechainv[6 +16] ^ statechainv[6 +24] ^ statechainv[6 +32];
	t[6] = statechainv[ 5] ^ statechainv[5 + 8] ^ statechainv[5 +16] ^ statechainv[5 +24] ^ statechainv[5 +32];
	t[5] = statechainv[ 4] ^ statechainv[4 + 8] ^ statechainv[4 +16] ^ statechainv[4 +24] ^ statechainv[4 +32];
	t[4] = statechainv[ 3] ^ statechainv[3 + 8] ^ statechainv[3 +16] ^ statechainv[3 +24] ^ statechainv[3 +32] ^ tmp;
	t[3] = statechainv[ 2] ^ statechainv[2 + 8] ^ statechainv[2 +16] ^ statechainv[2 +24] ^ statechainv[2 +32] ^ tmp;
	t[2] = statechainv[ 1] ^ statechainv[1 + 8] ^ statechainv[1 +16] ^ statechainv[1 +24] ^ statechainv[1 +32];
	t[1] = statechainv[ 0] ^ statechainv[0 + 8] ^ statechainv[0 +16] ^ statechainv[0 +24] ^ statechainv[0 +32] ^ tmp;
	t[0] = tmp;

//	*(uint2x4*)statechainv ^= *(uint2x4*)t;
	#pragma unroll 8
	for(int i=0;i<8;i++)
		statechainv[i] ^= t[i];

	#pragma unroll 4
	for (int j=1;j<5;j++) {
		#pragma unroll 8
		for(int i=0;i<8;i++)
			statechainv[i+(j<<3)] ^= t[i];
//		*(uint2x4*)&statechainv[8*j] ^= *(uint2x4*)t;
		#pragma unroll 8
		for(int i=0;i<8;i++)
			t[i+(j<<3)] = statechainv[i+(j<<3)];
//		*(uint2x4*)&t[8*j] = *(uint2x4*)&statechainv[8*j];
	}

//	*(uint2x4*)t = *(uint2x4*)statechainv;
	#pragma unroll 8
	for(int i=0;i<8;i++)
		t[i] = statechainv[i];

	MULT0(statechainv);

	#pragma unroll 4
	for (int j=1;j<5;j++)
		MULT2(statechainv, j);

	#pragma unroll 5
	for (int j=0;j<5;j++)
		#pragma unroll 8
		for(int i=0;i<8;i++)
			statechainv[i+8*j] ^= t[i+(8*((j+1)%5))];
//		*(uint2x4*)&statechainv[8*j] ^= *(uint2x4*)&t[8*((j+1)%5)];

	#pragma unroll 5
	for (int j=0;j<5;j++)
		*(uint2x4*)&t[8*j] = *(uint2x4*)&statechainv[8*j];

	MULT0(statechainv);
	#pragma unroll 4
	for (int j=1;j<5;j++)
		MULT2(statechainv, j);

	#pragma unroll 5
	for (int j=0;j<5;j++)
		*(uint2x4*)&statechainv[8*j] ^= *(uint2x4*)&t[8*((j+4)%5)];

	#pragma unroll 5
	for (int j=0;j<5;j++) {
		*(uint2x4*)&statechainv[8*j] ^= *(uint2x4*)statebuffer;
		MULT0(statebuffer);
	}

	TWEAK(statechainv[12], statechainv[13], statechainv[14], statechainv[15], 1);
	TWEAK(statechainv[20], statechainv[21], statechainv[22], statechainv[23], 2);
	TWEAK(statechainv[28], statechainv[29], statechainv[30], statechainv[31], 3);
	TWEAK(statechainv[36], statechainv[37], statechainv[38], statechainv[39], 4);

	for (int i = 0; i<8; i++){
		STEP2( statechainv    ,*(uint2*)&c_CNS[(2 * i) +  0], *(uint2*)&c_CNS[(2 * i) + 16]);
		STEP2(&statechainv[16],*(uint2*)&c_CNS[(2 * i) + 32], *(uint2*)&c_CNS[(2 * i) + 48]);
		STEP1(&statechainv[32],*(uint2*)&c_CNS[(2 * i) + 64]);
	}
}

__device__
static void rnd512_first(uint32_t *const __restrict__ state, uint32_t *const __restrict__ buffer)
{
	#pragma unroll 5
	for (int j = 0; j<5; j++) {
		uint32_t tmp;
		#pragma unroll 8
		for(int i=0;i<8;i++)
			state[i+(j<<3)] ^= buffer[i];
		MULT0(buffer);
	}
	TWEAK(state[12], state[13], state[14], state[15], 1);
	TWEAK(state[20], state[21], state[22], state[23], 2);
	TWEAK(state[28], state[29], state[30], state[31], 3);
	TWEAK(state[36], state[37], state[38], state[39], 4);
	
	for (int i = 0; i<8; i++) {
		STEP2(&state[ 0],*(uint2*)&c_CNS[(2 * i) +  0],*(uint2*)&c_CNS[(2 * i) + 16]);
		STEP2(&state[16],*(uint2*)&c_CNS[(2 * i) + 32],*(uint2*)&c_CNS[(2 * i) + 48]);
		STEP1(&state[32],*(uint2*)&c_CNS[(2 * i) + 64]);
	}
}

__device__
static void qubit_rnd512_first(uint32_t *const __restrict__ statebuffer, uint32_t *const __restrict__ statechainv){

	*(uint4*)&statechainv[ 0] ^= *(uint4*)&statebuffer[ 0];
	statechainv[ 4] ^= statebuffer[4];

	*(uint4*)&statechainv[ 9] ^= *(uint4*)&statebuffer[ 0];
	statechainv[13] ^= statebuffer[4];

	*(uint4*)&statechainv[18] ^= *(uint4*)&statebuffer[ 0];
	statechainv[22] ^= statebuffer[4];

	*(uint4*)&statechainv[27] ^= *(uint4*)&statebuffer[ 0];
	statechainv[31] ^= statebuffer[4];

	statechainv[0 + 8 * 4] ^= statebuffer[4];
	statechainv[1 + 8 * 4] ^= statebuffer[4];
	statechainv[3 + 8 * 4] ^= statebuffer[4];
	statechainv[4 + 8 * 4] ^= statebuffer[4];
	*(uint4*)&statechainv[4 + 8*4] ^= *(uint4*)&statebuffer[ 0];

	TWEAK(statechainv[12], statechainv[13], statechainv[14], statechainv[15], 1);
	TWEAK(statechainv[20], statechainv[21], statechainv[22], statechainv[23], 2);
	TWEAK(statechainv[28], statechainv[29], statechainv[30], statechainv[31], 3);
	TWEAK(statechainv[36], statechainv[37], statechainv[38], statechainv[39], 4);

	#pragma unroll 8
	for (uint32_t i = 0; i<8; i++){
		STEP2(&statechainv[ 0],*(uint2*)&c_CNS[(2 * i) +  0],*(uint2*)&c_CNS[(2 * i) + 16]);
		STEP2(&statechainv[16],*(uint2*)&c_CNS[(2 * i) + 32],*(uint2*)&c_CNS[(2 * i) + 48]);
		STEP1(&statechainv[32],*(uint2*)&c_CNS[(2 * i) + 64]);
	}
}



void rnd512cpu(uint32_t *statebuffer, uint32_t *statechainv)
{
	int i, j;
	uint32_t t[40];
	uint32_t chainv[8];
	uint32_t tmp;

	for (i = 0; i<8; i++)
	{
		t[i] = statechainv[i];
		for (j = 1; j<5; j++)
		{
			t[i] ^= statechainv[i + 8 * j];
		}
	}

	MULT2(t, 0);

	for (j = 0; j<5; j++)
	{
		for (i = 0; i<8; i++)
		{
			statechainv[i + 8 * j] ^= t[i];
		}
	}

	for (j = 0; j<5; j++)
	{
		for (i = 0; i<8; i++)
		{
			t[i + 8 * j] = statechainv[i + 8 * j];
		}
	}

	for (j = 0; j<5; j++)
	{
		MULT2(statechainv, j);
	}

	for (j = 0; j<5; j++)
	{
		for (i = 0; i<8; i++)
		{
			statechainv[8 * j + i] ^= t[8 * ((j + 1) % 5) + i];
		}
	}

	for (j = 0; j<5; j++)
	{
		for (i = 0; i<8; i++)
		{
			t[i + 8 * j] = statechainv[i + 8 * j];
		}
	}

	for (j = 0; j<5; j++)
	{
		MULT2(statechainv, j);
	}

	for (j = 0; j<5; j++)
	{
		for (i = 0; i<8; i++)
		{
			statechainv[8 * j + i] ^= t[8 * ((j + 4) % 5) + i];
		}
	}

	for (j = 0; j<5; j++)
	{
		for (i = 0; i<8; i++)
		{
			statechainv[i + 8 * j] ^= statebuffer[i];
		}
		MULT2(statebuffer, 0);
	}

	for (i = 0; i<8; i++)
	{
		chainv[i] = statechainv[i];
	}

	for (i = 0; i<8; i++)
	{
		STEP(h_CNS[(2 * i)], h_CNS[(2 * i) + 1]);
	}

	for (i = 0; i<8; i++)
	{
		statechainv[i] = chainv[i];
		chainv[i] = statechainv[i + 8];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 1);


	for (i = 0; i<8; i++)
	{
		STEP(h_CNS[(2 * i) + 16], h_CNS[(2 * i) + 16 + 1]);
	}

	for (i = 0; i<8; i++)
	{
		statechainv[i + 8] = chainv[i];
		chainv[i] = statechainv[i + 16];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 2);

	for (i = 0; i<8; i++)
	{
		STEP(h_CNS[(2 * i) + 32], h_CNS[(2 * i) + 32 + 1]);
	}

	for (i = 0; i<8; i++)
	{
		statechainv[i + 16] = chainv[i];
		chainv[i] = statechainv[i + 24];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 3);

	for (i = 0; i<8; i++)
	{
		STEP(h_CNS[(2 * i) + 48], h_CNS[(2 * i) + 48 + 1]);
	}

	for (i = 0; i<8; i++)
	{
		statechainv[i + 24] = chainv[i];
		chainv[i] = statechainv[i + 32];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 4);

	for (i = 0; i<8; i++)
	{
		STEP(h_CNS[(2 * i) + 64], h_CNS[(2 * i) + 64 + 1]);
	}

	for (i = 0; i<8; i++)
	{
		statechainv[i + 32] = chainv[i];
	}
}

/***************************************************/
__device__ __forceinline__
static void rnd512_nullhash(uint32_t *const __restrict__ state){

	uint32_t t[40];
	uint32_t tmp;

	tmp = state[ 7] ^ state[7 + 8] ^ state[7 +16] ^ state[7 +24] ^ state[7 +32];
	t[7] = state[ 6] ^ state[6 + 8] ^ state[6 +16] ^ state[6 +24] ^ state[6 +32];
	t[6] = state[ 5] ^ state[5 + 8] ^ state[5 +16] ^ state[5 +24] ^ state[5 +32];
	t[5] = state[ 4] ^ state[4 + 8] ^ state[4 +16] ^ state[4 +24] ^ state[4 +32];
	t[4] = state[ 3] ^ state[3 + 8] ^ state[3 +16] ^ state[3 +24] ^ state[3 +32] ^ tmp;
	t[3] = state[ 2] ^ state[2 + 8] ^ state[2 +16] ^ state[2 +24] ^ state[2 +32] ^ tmp;
	t[2] = state[ 1] ^ state[1 + 8] ^ state[1 +16] ^ state[1 +24] ^ state[1 +32];
	t[1] = state[ 0] ^ state[0 + 8] ^ state[0 +16] ^ state[0 +24] ^ state[0 +32] ^ tmp;
	t[0] = tmp;
	
	#pragma unroll 5
	for (int j = 0; j<5; j++){
//		#pragma unroll 8
//		for(int i=0;i<8;i++)
//			state[i+(j<<3)] ^= t[i];
		*(uint2x4*)&state[8*j] ^= *(uint2x4*)t;
	}
	
	#pragma unroll 5
	for (int j = 0; j<5; j++){
///		#pragma unroll 8
///		for(int i=0;i<8;i++)
//			t[i+(j<<3)] = state[i+(j<<3)];
		*(uint2x4*)&t[8*j] = *(uint2x4*)&state[8*j];
	}
	#pragma unroll 5
	for (int j = 0; j<5; j++) {
		MULT2(state, j);
	}

	#pragma unroll 5
	for (int j = 0; j<5; j++) {
//		#pragma unroll 8
//		for(int i=0;i<8;i++)
//			state[i+(j<<3)] ^= t[i + (((j + 1) % 5)<<3)];
		*(uint2x4*)&state[8*j] ^= *(uint2x4*)&t[8 * ((j + 1) % 5)];
	}

	#pragma unroll 5
	for (int j = 0; j<5; j++) {
//		#pragma unroll 8
//		for(int i=0;i<8;i++)
//			t[i+8*j] = state[i+8*j];
		*(uint2x4*)&t[8*j] = *(uint2x4*)&state[8*j];
	}

	#pragma unroll 5
	for (int j = 0; j<5; j++) {
		MULT2(state, j);
	}

	#pragma unroll 5
	for (int j = 0; j<5; j++) {
		#pragma unroll 8
		for(int i=0;i<8;i++)
			state[i+8*j] ^= t[i+(8 * ((j + 4) % 5))];
//		*(uint2x4*)&state[8*j] ^= *(uint2x4*)&t[8 * ((j + 4) % 5)];
	}

	TWEAK(state[12], state[13], state[14], state[15], 1);
	TWEAK(state[20], state[21], state[22], state[23], 2);
	TWEAK(state[28], state[29], state[30], state[31], 3);
	TWEAK(state[36], state[37], state[38], state[39], 4);
	
//	#pragma unroll 8
	for (int i = 0; i<8; i++) {
		STEP2(&state[ 0],*(uint2*)&c_CNS[(2 * i) +  0],*(uint2*)&c_CNS[(2 * i) + 16]);
		STEP2(&state[16],*(uint2*)&c_CNS[(2 * i) + 32],*(uint2*)&c_CNS[(2 * i) + 48]);
		STEP1(&state[32],*(uint2*)&c_CNS[(2 * i) + 64]);
	}
}


/***************************************************/
// Die Hash-Funktion
//#if __CUDA_ARCH__ == 500
//__launch_bounds__(256, 4)
//#endif


__global__ __launch_bounds__(256, 4)
void qubit_luffa512_gpu_hash_80(const uint32_t threads,const uint32_t startNounce, uint32_t *outputHash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nounce = startNounce + thread;
		uint64_t buff[16] = {0};

		buff[ 8] = c_PaddedMessage80[8];
		buff[ 9] = c_PaddedMessage80[9];
		buff[10] = 0x80;
		buff[11] = 0x0100;
		buff[15] = 0x8002000000000000;

		// die Nounce durch die thread-spezifische ersetzen
		buff[9] = REPLACE_HIDWORD(buff[9], cuda_swab32(nounce));

		uint32_t statebuffer[8];
		uint32_t statechainv[40];

		#pragma unroll 4
		for (int i = 0; i<4; i++)
			statebuffer[i] = cuda_swab32(((uint32_t*)buff)[i + 16]);

		*(uint4*)&statebuffer[ 4] = *(uint4*)&statebufferpre[ 4];

		#pragma unroll 40
		for (int i = 0; i<40; i++)
			statechainv[i] = statechainvpre[i];

		statebuffer[4] = 0x80000000;

		qubit_rnd512_first(statebuffer, statechainv);

		uint32_t *outHash = outputHash + (thread<<4);

		rnd512_nullhash(statechainv);
		*(uint2x4*)&outHash[ 0] = swapvec(*(uint2x4*)&statechainv[ 0] ^ *(uint2x4*)&statechainv[ 8] ^ *(uint2x4*)&statechainv[16] ^ *(uint2x4*)&statechainv[24] ^ *(uint2x4*)&statechainv[32]);
			
		rnd512_nullhash(statechainv);
		*(uint2x4*)&outHash[ 8] = swapvec(*(uint2x4*)&statechainv[ 0] ^ *(uint2x4*)&statechainv[ 8] ^ *(uint2x4*)&statechainv[16] ^ *(uint2x4*)&statechainv[24] ^ *(uint2x4*)&statechainv[32]);


	}
}

__host__
void qubit_luffa512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash){

	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	qubit_luffa512_gpu_hash_80 <<<grid, block>>> (threads, startNounce, d_outputHash);
}

//#if __CUDA_ARCH__ == 500
//	#define __ldg(x) (*x)
//#endif

#define sph_u32 uint32_t

__constant__ static const sph_u32 RC00[8] = {
  SPH_C32(0x303994a6), SPH_C32(0xc0e65299),
  SPH_C32(0x6cc33a12), SPH_C32(0xdc56983e),
  SPH_C32(0x1e00108f), SPH_C32(0x7800423d),
  SPH_C32(0x8f5b7882), SPH_C32(0x96e1db12)
};

__constant__ static const sph_u32 RC04[8] = {
  SPH_C32(0xe0337818), SPH_C32(0x441ba90d),
  SPH_C32(0x7f34d442), SPH_C32(0x9389217f),
  SPH_C32(0xe5a8bce6), SPH_C32(0x5274baf4),
  SPH_C32(0x26889ba7), SPH_C32(0x9a226e9d)
};

__constant__ static const sph_u32 RC10[8] = {
  SPH_C32(0xb6de10ed), SPH_C32(0x70f47aae),
  SPH_C32(0x0707a3d4), SPH_C32(0x1c1e8f51),
  SPH_C32(0x707a3d45), SPH_C32(0xaeb28562),
  SPH_C32(0xbaca1589), SPH_C32(0x40a46f3e)
};

__constant__ static const sph_u32 RC14[8] = {
  SPH_C32(0x01685f3d), SPH_C32(0x05a17cf4),
  SPH_C32(0xbd09caca), SPH_C32(0xf4272b28),
  SPH_C32(0x144ae5cc), SPH_C32(0xfaa7ae2b),
  SPH_C32(0x2e48f1c1), SPH_C32(0xb923c704)
};

__constant__ static const sph_u32 RC20[8] = {
  SPH_C32(0xfc20d9d2), SPH_C32(0x34552e25),
  SPH_C32(0x7ad8818f), SPH_C32(0x8438764a),
  SPH_C32(0xbb6de032), SPH_C32(0xedb780c8),
  SPH_C32(0xd9847356), SPH_C32(0xa2c78434)
};

__constant__ static const sph_u32 RC24[8] = {
  SPH_C32(0xe25e72c1), SPH_C32(0xe623bb72),
  SPH_C32(0x5c58a4a4), SPH_C32(0x1e38e2e7),
  SPH_C32(0x78e38b9d), SPH_C32(0x27586719),
  SPH_C32(0x36eda57f), SPH_C32(0x703aace7)
};

__constant__ static const sph_u32 RC30[8] = {
  SPH_C32(0xb213afa5), SPH_C32(0xc84ebe95),
  SPH_C32(0x4e608a22), SPH_C32(0x56d858fe),
  SPH_C32(0x343b138f), SPH_C32(0xd0ec4e3d),
  SPH_C32(0x2ceb4882), SPH_C32(0xb3ad2208)
};

__constant__ static const sph_u32 RC34[8] = {
  SPH_C32(0xe028c9bf), SPH_C32(0x44756f91),
  SPH_C32(0x7e8fce32), SPH_C32(0x956548be),
  SPH_C32(0xfe191be2), SPH_C32(0x3cb226e5),
  SPH_C32(0x5944a28e), SPH_C32(0xa1c4c355)
};


__constant__ static const sph_u32 RC40[8] = {
  SPH_C32(0xf0d2e9e3), SPH_C32(0xac11d7fa),
  SPH_C32(0x1bcb66f2), SPH_C32(0x6f2d9bc9),
  SPH_C32(0x78602649), SPH_C32(0x8edae952),
  SPH_C32(0x3b6ba548), SPH_C32(0xedae9520)
};

__constant__ static const sph_u32 RC44[8] = {
  SPH_C32(0x5090d577), SPH_C32(0x2d1925ab),
  SPH_C32(0xb46496ac), SPH_C32(0xd1925ab0),
  SPH_C32(0x29131ab6), SPH_C32(0x0fc053c3),
  SPH_C32(0x3f014f0c), SPH_C32(0xfc053c31)
};

#define SPH_ROTL32 ROTL32

#define DECL_TMP8(w) \
  sph_u32 w ## 0, w ## 1, w ## 2, w ## 3, w ## 4, w ## 5, w ## 6, w ## 7;

#define M2(d, s)   do { \
    sph_u32 tmp = s ## 7; \
    d ## 7 = s ## 6; \
    d ## 6 = s ## 5; \
    d ## 5 = s ## 4; \
    d ## 4 = s ## 3 ^ tmp; \
    d ## 3 = s ## 2 ^ tmp; \
    d ## 2 = s ## 1; \
    d ## 1 = s ## 0 ^ tmp; \
    d ## 0 = tmp; \
  } while (0)

#define XOR(d, s1, s2)   do { \
    d ## 0 = s1 ## 0 ^ s2 ## 0; \
    d ## 1 = s1 ## 1 ^ s2 ## 1; \
    d ## 2 = s1 ## 2 ^ s2 ## 2; \
    d ## 3 = s1 ## 3 ^ s2 ## 3; \
    d ## 4 = s1 ## 4 ^ s2 ## 4; \
    d ## 5 = s1 ## 5 ^ s2 ## 5; \
    d ## 6 = s1 ## 6 ^ s2 ## 6; \
    d ## 7 = s1 ## 7 ^ s2 ## 7; \
  } while (0)

#define SUB_CRUMB(a0, a1, a2, a3)   do { \
    sph_u32 tmp; \
    tmp = (a0); \
    (a0) |= (a1); \
    (a2) ^= (a3); \
    (a1) = SPH_T32(~(a1)); \
    (a0) ^= (a3); \
    (a3) &= tmp; \
    (a1) ^= (a3); \
    (a3) ^= (a2); \
    (a2) &= (a0); \
    (a0) = SPH_T32(~(a0)); \
    (a2) ^= (a1); \
    (a1) |= (a3); \
    tmp ^= (a1); \
    (a3) ^= (a2); \
    (a2) &= (a1); \
    (a1) ^= (a0); \
    (a0) = tmp; \
  } while (0)


#define MIX_WORD(u, v)   do { \
    (v) ^= (u); \
    (u) = SPH_ROTL32((u), 2) ^ (v); \
    (v) = SPH_ROTL32((v), 14) ^ (u); \
    (u) = SPH_ROTL32((u), 10) ^ (v); \
    (v) = SPH_ROTL32((v), 1); \
  } while (0)

#define MI5   do { \
    DECL_TMP8(a) \
    DECL_TMP8(b) \
    XOR(a, V0, V1); \
    XOR(b, V2, V3); \
    XOR(a, a, b); \
    XOR(a, a, V4); \
    M2(a, a); \
    XOR(V0, a, V0); \
    XOR(V1, a, V1); \
    XOR(V2, a, V2); \
    XOR(V3, a, V3); \
    XOR(V4, a, V4); \
    M2(b, V0); \
    XOR(b, b, V1); \
    M2(V1, V1); \
    XOR(V1, V1, V2); \
    M2(V2, V2); \
    XOR(V2, V2, V3); \
    M2(V3, V3); \
    XOR(V3, V3, V4); \
    M2(V4, V4); \
    XOR(V4, V4, V0); \
    M2(V0, b); \
    XOR(V0, V0, V4); \
    M2(V4, V4); \
    XOR(V4, V4, V3); \
    M2(V3, V3); \
    XOR(V3, V3, V2); \
    M2(V2, V2); \
    XOR(V2, V2, V1); \
    M2(V1, V1); \
    XOR(V1, V1, b); \
    XOR(V0, V0, M); \
    M2(M, M); \
    XOR(V1, V1, M); \
    M2(M, M); \
    XOR(V2, V2, M); \
    M2(M, M); \
    XOR(V3, V3, M); \
    M2(M, M); \
    XOR(V4, V4, M); \
  } while (0)

#define TWEAK5   do { \
    V14 = SPH_ROTL32(V14, 1); \
    V15 = SPH_ROTL32(V15, 1); \
    V16 = SPH_ROTL32(V16, 1); \
    V17 = SPH_ROTL32(V17, 1); \
    V24 = SPH_ROTL32(V24, 2); \
    V25 = SPH_ROTL32(V25, 2); \
    V26 = SPH_ROTL32(V26, 2); \
    V27 = SPH_ROTL32(V27, 2); \
    V34 = SPH_ROTL32(V34, 3); \
    V35 = SPH_ROTL32(V35, 3); \
    V36 = SPH_ROTL32(V36, 3); \
    V37 = SPH_ROTL32(V37, 3); \
    V44 = SPH_ROTL32(V44, 4); \
    V45 = SPH_ROTL32(V45, 4); \
    V46 = SPH_ROTL32(V46, 4); \
    V47 = SPH_ROTL32(V47, 4); \
  } while (0)



#define LUFFA_P5   do { \
    int r; \
    TWEAK5; \
    for (r = 0; r < 8; r ++) { \
      SUB_CRUMB(V00, V01, V02, V03); \
      SUB_CRUMB(V05, V06, V07, V04); \
      MIX_WORD(V00, V04); \
      MIX_WORD(V01, V05); \
      MIX_WORD(V02, V06); \
      MIX_WORD(V03, V07); \
      V00 ^= RC00[r]; \
      V04 ^= RC04[r]; \
    } \
    for (r = 0; r < 8; r ++) { \
      SUB_CRUMB(V10, V11, V12, V13); \
      SUB_CRUMB(V15, V16, V17, V14); \
      MIX_WORD(V10, V14); \
      MIX_WORD(V11, V15); \
      MIX_WORD(V12, V16); \
      MIX_WORD(V13, V17); \
      V10 ^= RC10[r]; \
      V14 ^= RC14[r]; \
    } \
    for (r = 0; r < 8; r ++) { \
      SUB_CRUMB(V20, V21, V22, V23); \
      SUB_CRUMB(V25, V26, V27, V24); \
      MIX_WORD(V20, V24); \
      MIX_WORD(V21, V25); \
      MIX_WORD(V22, V26); \
      MIX_WORD(V23, V27); \
      V20 ^= RC20[r]; \
      V24 ^= RC24[r]; \
    } \
    for (r = 0; r < 8; r ++) { \
      SUB_CRUMB(V30, V31, V32, V33); \
      SUB_CRUMB(V35, V36, V37, V34); \
      MIX_WORD(V30, V34); \
      MIX_WORD(V31, V35); \
      MIX_WORD(V32, V36); \
      MIX_WORD(V33, V37); \
      V30 ^= RC30[r]; \
      V34 ^= RC34[r]; \
    } \
    for (r = 0; r < 8; r ++) { \
      SUB_CRUMB(V40, V41, V42, V43); \
      SUB_CRUMB(V45, V46, V47, V44); \
      MIX_WORD(V40, V44); \
      MIX_WORD(V41, V45); \
      MIX_WORD(V42, V46); \
      MIX_WORD(V43, V47); \
      V40 ^= RC40[r]; \
      V44 ^= RC44[r]; \
    } \
  } while (0)


#define LUFFA_P5_LDG   do { \
    int r; \
    TWEAK5; \
    for (r = 0; r < 8; r ++) { \
      SUB_CRUMB(V00, V01, V02, V03); \
      SUB_CRUMB(V05, V06, V07, V04); \
      MIX_WORD(V00, V04); \
      MIX_WORD(V01, V05); \
      MIX_WORD(V02, V06); \
      MIX_WORD(V03, V07); \
      V00 ^= (RC00[r]); \
      V04 ^= (RC04[r]); \
    } \
    for (r = 0; r < 8; r ++) { \
      SUB_CRUMB(V10, V11, V12, V13); \
      SUB_CRUMB(V15, V16, V17, V14); \
      MIX_WORD(V10, V14); \
      MIX_WORD(V11, V15); \
      MIX_WORD(V12, V16); \
      MIX_WORD(V13, V17); \
      V10 ^= __ldg(&RC10[r]); \
      V14 ^= __ldg(&RC14[r]); \
    } \
    for (r = 0; r < 8; r ++) { \
      SUB_CRUMB(V20, V21, V22, V23); \
      SUB_CRUMB(V25, V26, V27, V24); \
      MIX_WORD(V20, V24); \
      MIX_WORD(V21, V25); \
      MIX_WORD(V22, V26); \
      MIX_WORD(V23, V27); \
      V20 ^= __ldg(&RC20[r]); \
      V24 ^= __ldg(&RC24[r]); \
    } \
    for (r = 0; r < 8; r ++) { \
      SUB_CRUMB(V30, V31, V32, V33); \
      SUB_CRUMB(V35, V36, V37, V34); \
      MIX_WORD(V30, V34); \
      MIX_WORD(V31, V35); \
      MIX_WORD(V32, V36); \
      MIX_WORD(V33, V37); \
      V30 ^= __ldg(&RC30[r]); \
      V34 ^= __ldg(&RC34[r]); \
    } \
    for (r = 0; r < 8; r ++) { \
      SUB_CRUMB(V40, V41, V42, V43); \
      SUB_CRUMB(V45, V46, V47, V44); \
      MIX_WORD(V40, V44); \
      MIX_WORD(V41, V45); \
      MIX_WORD(V42, V46); \
      MIX_WORD(V43, V47); \
      V40 ^= __ldg(&RC40[r]); \
      V44 ^= __ldg(&RC44[r]); \
    } \
  } while (0)


#define TPB_L 256
__global__
__launch_bounds__(TPB_L,2)
void x11_luffa512_gpu_hash_64(uint32_t threads, uint32_t *g_hash){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint32_t statebuffer[8];
	
	if (thread < threads)
	{

		uint32_t statechainv[40] = {
			0x8bb0a761, 0xc2e4aa8b, 0x2d539bc9, 0x381408f8,	0x478f6633, 0x255a46ff, 0x581c37f7, 0x601c2e8e,
			0x266c5f9d, 0xc34715d8, 0x8900670e, 0x51a540be,	0xe4ce69fb, 0x5089f4d4, 0x3cc0a506, 0x609bcb02,
			0xa4e3cd82, 0xd24fd6ca, 0xc0f196dc, 0xcf41eafe,	0x0ff2e673, 0x303804f2, 0xa7b3cd48, 0x677addd4,
			0x66e66a8a, 0x2303208f, 0x486dafb4, 0xc0d37dc6,	0x634d15af, 0xe5af6747, 0x10af7e38, 0xee7e6428,
			0x01262e5d, 0xc92c2e64, 0x82fee966, 0xcea738d3,	0x867de2b0, 0xe0714818, 0xda6e831f, 0xa7062529
		};

/*
	uint32_t statechainv[40] =
	{
		0x6d251e69, 0x44b051e0, 0x4eaa6fb4, 0xdbf78465,	0x6e292011, 0x90152df4, 0xee058139, 0xdef610bb,
		0xc3b44b95, 0xd9d2f256, 0x70eee9a0, 0xde099fa3,	0x5d9b0557, 0x8fc944b3, 0xcf1ccf0e, 0x746cd581,
		0xf7efc89d, 0x5dba5781, 0x04016ce5, 0xad659c05,	0x0306194f, 0x666d1836, 0x24aa230a, 0x8b264ae7,
		0x858075d5, 0x36d79cce, 0xe571f7d7, 0x204b1f67,	0x35870c6a, 0x57e9e923, 0x14bcb808, 0x7cde72ce,
		0x6c68e9be, 0x5ec41e22, 0xc825b7c7, 0xaffb4363,	0xf5df3999, 0x0fc688f1, 0xb07224cc, 0x03e86cea
	};
*/
		uint2x4* Hash = (uint2x4*)&g_hash[thread<<4];

		uint32_t hash[16];

		*(uint2x4*)&hash[0] = __ldg4(&Hash[0]);
		*(uint2x4*)&hash[8] = __ldg4(&Hash[1]);
		
//		for(int i=0;i<16;i++)hash[i]=0;

  sph_u32 V00 = SPH_C32(0x6d251e69), V01 = SPH_C32(0x44b051e0), V02 = SPH_C32(0x4eaa6fb4), V03 = SPH_C32(0xdbf78465), V04 = SPH_C32(0x6e292011), V05 = SPH_C32(0x90152df4), V06 = SPH_C32(0xee058139), V07 = SPH_C32(0xdef610bb);
  sph_u32 V10 = SPH_C32(0xc3b44b95), V11 = SPH_C32(0xd9d2f256), V12 = SPH_C32(0x70eee9a0), V13 = SPH_C32(0xde099fa3), V14 = SPH_C32(0x5d9b0557), V15 = SPH_C32(0x8fc944b3), V16 = SPH_C32(0xcf1ccf0e), V17 = SPH_C32(0x746cd581);
  sph_u32 V20 = SPH_C32(0xf7efc89d), V21 = SPH_C32(0x5dba5781), V22 = SPH_C32(0x04016ce5), V23 = SPH_C32(0xad659c05), V24 = SPH_C32(0x0306194f), V25 = SPH_C32(0x666d1836), V26 = SPH_C32(0x24aa230a), V27 = SPH_C32(0x8b264ae7);
  sph_u32 V30 = SPH_C32(0x858075d5), V31 = SPH_C32(0x36d79cce), V32 = SPH_C32(0xe571f7d7), V33 = SPH_C32(0x204b1f67), V34 = SPH_C32(0x35870c6a), V35 = SPH_C32(0x57e9e923), V36 = SPH_C32(0x14bcb808), V37 = SPH_C32(0x7cde72ce);
  sph_u32 V40 = SPH_C32(0x6c68e9be), V41 = SPH_C32(0x5ec41e22), V42 = SPH_C32(0xc825b7c7), V43 = SPH_C32(0xaffb4363), V44 = SPH_C32(0xf5df3999), V45 = SPH_C32(0x0fc688f1), V46 = SPH_C32(0xb07224cc), V47 = SPH_C32(0x03e86cea);

  DECL_TMP8(M);

  M0 = cuda_swab32(hash[0]);
  M1 = cuda_swab32(hash[1]);
  M2 = cuda_swab32(hash[2]);
  M3 = cuda_swab32(hash[3]);
  M4 = cuda_swab32(hash[4]);
  M5 = cuda_swab32(hash[5]);
  M6 = cuda_swab32(hash[6]);
  M7 = cuda_swab32(hash[7]);
//#pragma unroll 7
  for(int i = 0; i < 7; i++)
  {
    MI5;
//#pragma unroll 8
    LUFFA_P5;


    if(i == 0)
    {
      M0 = cuda_swab32(hash[8]);
      M1 = cuda_swab32(hash[9]);
      M2 = cuda_swab32(hash[10]);
      M3 = cuda_swab32(hash[11]);
      M4 = cuda_swab32(hash[12]);
      M5 = cuda_swab32(hash[13]);
      M6 = cuda_swab32(hash[14]);
      M7 = cuda_swab32(hash[15]);
    }
    else if(i == 1)
    {
      M0 = M1 = M2 = M3 = M4 = M5 = M6 = M7 = 0;
    }
    else if(i == 2)
    {
      M0 = M1 = M2 = M3 = M4 = M5 = M6 = M7 = 0;
    }
    else if(i == 3)
    {
      M0 = 0x80000000;
      M1 = M2 = M3 = M4 = M5 = M6 = M7 = 0;
    }
    else if(i == 4)
      M0 = M1 = M2 = M3 = M4 = M5 = M6 = M7 = 0;
    else if(i == 5)
    {
      hash[0] = cuda_swab32(V00 ^ V10 ^ V20 ^ V30 ^ V40);
      hash[1] = cuda_swab32(V01 ^ V11 ^ V21 ^ V31 ^ V41);
      hash[2] = cuda_swab32(V02 ^ V12 ^ V22 ^ V32 ^ V42);
      hash[3] = cuda_swab32(V03 ^ V13 ^ V23 ^ V33 ^ V43);
      hash[4] = cuda_swab32(V04 ^ V14 ^ V24 ^ V34 ^ V44);
      hash[5] = cuda_swab32(V05 ^ V15 ^ V25 ^ V35 ^ V45);
      hash[6] = cuda_swab32(V06 ^ V16 ^ V26 ^ V36 ^ V46);
      hash[7] = cuda_swab32(V07 ^ V17 ^ V27 ^ V37 ^ V47);
    }
  }

  hash[8] = cuda_swab32(V00 ^ V10 ^ V20 ^ V30 ^ V40);
  hash[9] = cuda_swab32(V01 ^ V11 ^ V21 ^ V31 ^ V41);
  hash[10] = cuda_swab32(V02 ^ V12 ^ V22 ^ V32 ^ V42);
  hash[11] = cuda_swab32(V03 ^ V13 ^ V23 ^ V33 ^ V43);
  hash[12] = cuda_swab32(V04 ^ V14 ^ V24 ^ V34 ^ V44);
  hash[13] = cuda_swab32(V05 ^ V15 ^ V25 ^ V35 ^ V45);
  hash[14] = cuda_swab32(V06 ^ V16 ^ V26 ^ V36 ^ V46);
  hash[15] = cuda_swab32(V07 ^ V17 ^ V27 ^ V37 ^ V47);

		Hash[ 0] = *(uint2x4*)&hash[ 0];
		Hash[ 1] = *(uint2x4*)&hash[ 8];
	}
}

__host__ void qubit_cpu_precalc()
{
	uint32_t tmp,i,j;
	uint32_t statebuffer[8];
	uint32_t t[40];


	uint32_t statechainv[40] =
	{
		0x6d251e69, 0x44b051e0, 0x4eaa6fb4, 0xdbf78465,	0x6e292011, 0x90152df4, 0xee058139, 0xdef610bb,
		0xc3b44b95, 0xd9d2f256, 0x70eee9a0, 0xde099fa3,	0x5d9b0557, 0x8fc944b3, 0xcf1ccf0e, 0x746cd581,
		0xf7efc89d, 0x5dba5781, 0x04016ce5, 0xad659c05,	0x0306194f, 0x666d1836, 0x24aa230a, 0x8b264ae7,
		0x858075d5, 0x36d79cce, 0xe571f7d7, 0x204b1f67,	0x35870c6a, 0x57e9e923, 0x14bcb808, 0x7cde72ce,
		0x6c68e9be, 0x5ec41e22, 0xc825b7c7, 0xaffb4363,	0xf5df3999, 0x0fc688f1, 0xb07224cc, 0x03e86cea
	};
	for (int i = 0; i<8; i++)
		statebuffer[i] = cuda_swab32(*(((uint32_t*)PaddedMessage) + i));
	rnd512cpu(statebuffer, statechainv);

	for (int i = 0; i<8; i++)
		statebuffer[i] = cuda_swab32(*(((uint32_t*)PaddedMessage) + i + 8));

	rnd512cpu(statebuffer, statechainv);


	for (int i = 0; i<8; i++)
	{
		t[i] = statechainv[i];
		for (int j = 1; j<5; j++)
		{
			t[i] ^= statechainv[i + 8 * j];
		}
	}

	MULT2(t, 0);

	for (int j = 0; j<5; j++) {
		for (int i = 0; i<8; i++) {
			statechainv[i + 8 * j] ^= t[i];
		}
	}
	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++) {
			t[i + 8 * j] = statechainv[i + 8 * j];
		}
	}

	for (j = 0; j<5; j++) {
		MULT2(statechainv, j);
	}

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++) {
			statechainv[8 * j + i] ^= t[8 * ((j + 1) % 5) + i];
		}
	}

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++) {
			t[i + 8 * j] = statechainv[i + 8 * j];
		}
	}

	for (j = 0; j<5; j++) {
		MULT2(statechainv, j);
	}

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++) {
			statechainv[8 * j + i] ^= t[8 * ((j + 4) % 5) + i];
		}
	}
	cudaMemcpyToSymbol(statebufferpre, statebuffer, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(statechainvpre, statechainv, 40 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

__host__
void qubit_luffa512_cpu_setBlock_80(void *pdata)
{
	memcpy(PaddedMessage, pdata, 80);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol( c_PaddedMessage80, PaddedMessage, 10*sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
	qubit_cpu_precalc();
}

__host__
void x11_luffa512_cpu_hash_64(int thr_id, uint32_t threads,uint32_t *d_hash)
{
    const uint32_t threadsperblock = TPB_L;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    x11_luffa512_gpu_hash_64<<<grid, block>>>(threads,d_hash);
}

////////


////////

__device__ __forceinline__
static void Update512(uint32_t *statebuffer, uint32_t *statechainv, const uint32_t *data)
{
	#pragma unroll
	for (int i = 0; i < 8; i++) statebuffer[i] = cuda_swab32(data[i]);
	rnd512_first(statechainv, statebuffer);

	#pragma unroll
	for (int i = 0; i < 8; i++) statebuffer[i] = cuda_swab32(data[i + 8]);
	rnd512(statebuffer, statechainv);
}

/***************************************************/
__device__ __forceinline__
static void finalization512(uint32_t *statebuffer, uint32_t *statechainv, uint32_t *b)
{
	int i,j;

	statebuffer[0] = 0x80000000;
	#pragma unroll 7
	for(int i=1;i<8;i++) statebuffer[i] = 0;
	rnd512(statebuffer, statechainv);

	/*---- blank round with m=0 ----*/
	rnd512_nullhash(statechainv);

	#pragma unroll
	for(i=0;i<8;i++) {
		b[i] = statechainv[i];
		#pragma unroll 4
		for(j=1;j<5;j++) {
			b[i] ^= statechainv[i+8*j];
		}
		b[i] = cuda_swab32((b[i]));
	}

	rnd512_nullhash(statechainv);

	#pragma unroll
	for(i=0;i<8;i++) {
		b[8 + i] = statechainv[i];
		#pragma unroll 4
		for(j=1;j<5;j++) {
			b[8+i] ^= statechainv[i+8*j];
		}
		b[8 + i] = cuda_swab32((b[8 + i]));
	}
}


/*
__global__
__launch_bounds__(TPB_L,2)
void x11_luffa512_gpu_hash_128(uint32_t threads, uint32_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t statechainv[40] = {
			0x8bb0a761, 0xc2e4aa8b, 0x2d539bc9, 0x381408f8,
			0x478f6633, 0x255a46ff, 0x581c37f7, 0x601c2e8e,
			0x266c5f9d, 0xc34715d8, 0x8900670e, 0x51a540be,
			0xe4ce69fb, 0x5089f4d4, 0x3cc0a506, 0x609bcb02,
			0xa4e3cd82, 0xd24fd6ca, 0xc0f196dc, 0xcf41eafe,
			0x0ff2e673, 0x303804f2, 0xa7b3cd48, 0x677addd4,
			0x66e66a8a, 0x2303208f, 0x486dafb4, 0xc0d37dc6,
			0x634d15af, 0xe5af6747, 0x10af7e38, 0xee7e6428,
			0x01262e5d, 0xc92c2e64, 0x82fee966, 0xcea738d3,
			0x867de2b0, 0xe0714818, 0xda6e831f, 0xa7062529
		};

		uint32_t statebuffer[8];
		uint32_t *const Hash = &g_hash[thread * 16U];

		Update512(statebuffer, statechainv, Hash);
		finalization512(statebuffer, statechainv, Hash);

		//Cubehash

		uint32_t x0 = 0x2AEA2A61, x1 = 0x50F494D4, x2 = 0x2D538B8B, x3 = 0x4167D83E;
		uint32_t x4 = 0x3FEE2313, x5 = 0xC701CF8C, x6 = 0xCC39968E, x7 = 0x50AC5695;
		uint32_t x8 = 0x4D42C787, x9 = 0xA647A8B3, xa = 0x97CF0BEF, xb = 0x825B4537;
		uint32_t xc = 0xEEF864D2, xd = 0xF22090C4, xe = 0xD0E5CD33, xf = 0xA23911AE;
		uint32_t xg = 0xFCD398D9, xh = 0x148FE485, xi = 0x1B017BEF, xj = 0xB6444532;
		uint32_t xk = 0x6A536159, xl = 0x2FF5781C, xm = 0x91FA7934, xn = 0x0DBADEA9;
		uint32_t xo = 0xD65C8A2B, xp = 0xA5A70E75, xq = 0xB1C62456, xr = 0xBC796576;
		uint32_t xs = 0x1921C8F7, xt = 0xE7989AF1, xu = 0x7795D246, xv = 0xD43E3B44;

		x0 ^= Hash[0];
		x1 ^= Hash[1];
		x2 ^= Hash[2];
		x3 ^= Hash[3];
		x4 ^= Hash[4];
		x5 ^= Hash[5];
		x6 ^= Hash[6];
		x7 ^= Hash[7];

		SIXTEEN_ROUNDS;

		x0 ^= Hash[8];
		x1 ^= Hash[9];
		x2 ^= Hash[10];
		x3 ^= Hash[11];
		x4 ^= Hash[12];
		x5 ^= Hash[13];
		x6 ^= Hash[14];
		x7 ^= Hash[15];

		SIXTEEN_ROUNDS;
		x0 ^= 0x80;

		SIXTEEN_ROUNDS;
		xv ^= 1;

		for (int i = 3; i < 13; i++) {
			SIXTEEN_ROUNDS;
		}

		Hash[0] = x0;
		Hash[1] = x1;
		Hash[2] = x2;
		Hash[3] = x3;
		Hash[4] = x4;
		Hash[5] = x5;
		Hash[6] = x6;
		Hash[7] = x7;
		Hash[8] = x8;
		Hash[9] = x9;
		Hash[10] = xa;
		Hash[11] = xb;
		Hash[12] = xc;
		Hash[13] = xd;
		Hash[14] = xe;
		Hash[15] = xf;

	}
}

__host__
void x11_luffa512_cpu_hash_128(int thr_id, uint32_t threads,uint32_t *d_hash)
{
    const uint32_t threadsperblock = TPB_L;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    x11_luffa512_gpu_hash_128<<<grid, block>>>(threads,d_hash);
}
*/
