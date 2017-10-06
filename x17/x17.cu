/**
 * X17 algorithm (X15 + sha512 + haval256)
 */

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"

#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"

#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"

#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"

#include "sph/sph_sha2.h"
#include "sph/sph_haval.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "x11/cuda_x11.h"

#define NBN 2

// Memory for the hash functions
static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

extern void x13_hamsi_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int mode);
extern void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x15_whirlpool_cpu_free(int thr_id);

extern void x17_sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void x17_haval256_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t* resNonce, uint64_t target);
extern void bmw256_cpu_hash_32_full(int thr_id, uint32_t threads, uint32_t *g_hash);
extern void quark_bmw512_cpu_hash_64x(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void quark_groestl512(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void groestl512_cpu_init(int thr_id, uint32_t threads);
extern void groestl512_cpu_hash(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void xevan_skein512(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void keccak_xevan_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void qubit_luffa512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash);
extern void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x11_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void xevan_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x11_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void xevan_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x11_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x11_simd512_cpu_init(int thr_id, uint32_t threads);
extern void xevan_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x13_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void xevan_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void xevan_sha512_cpu_hash_64(int thr_id, int threads, uint32_t *d_hash);
extern void xevan_haval512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_outputHash);
extern void xevan_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void xevan_haval512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *resNonce, uint64_t target);
extern void xevan_groestl512_cpu_hash(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void keccak_xevan_cpu_hash_64_A(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_outputHash);
extern void quark_blake512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_outputHash);
extern void quark_groestl512_cpu_hash_128(int thr_id, uint32_t threads,  uint32_t *d_hash);
extern void x11_luffa512_cpu_hash_128(int thr_id, uint32_t threads,uint32_t *d_hash);



// X17 CPU Hash (Validation)
extern "C" void x17hash(void *output, const void *input)
{
	uint32_t _ALIGN(64) hash[32]; // 128 bytes required
	const int dataLen = 128;
//return;
	sph_blake512_context     ctx_blake;
	sph_bmw512_context       ctx_bmw;
	sph_groestl512_context   ctx_groestl;
	sph_skein512_context     ctx_skein;
	sph_jh512_context        ctx_jh;
	sph_keccak512_context    ctx_keccak;
	sph_luffa512_context     ctx_luffa;
	sph_cubehash512_context  ctx_cubehash;
	sph_shavite512_context   ctx_shavite;
	sph_simd512_context      ctx_simd;
	sph_echo512_context      ctx_echo;
	sph_hamsi512_context     ctx_hamsi;
	sph_fugue512_context     ctx_fugue;
	sph_shabal512_context    ctx_shabal;
	sph_whirlpool_context    ctx_whirlpool;
	sph_sha512_context       ctx_sha512;
	sph_haval256_5_context   ctx_haval;

//print_hash(input,20);
	sph_blake512_init(&ctx_blake);
	sph_blake512(&ctx_blake, input, 80);
	sph_blake512_close(&ctx_blake, hash);
//print_hash(hash,32);
	memset(&hash[16], 0, 64);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512(&ctx_bmw, hash, dataLen);
	sph_bmw512_close(&ctx_bmw, hash);
//print_hash(hash,32);
//for(int i=0;i<32;i++)hash[i]=0;
	sph_groestl512_init(&ctx_groestl);
	sph_groestl512(&ctx_groestl, hash, dataLen);
	sph_groestl512_close(&ctx_groestl, hash);
//print_hash(hash,32);
//for(int i=0;i<32;i++)hash[i]=0;

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, hash, dataLen);
	sph_skein512_close(&ctx_skein, hash);

//print_hash(hash,32);
//for(int i=0;i<32;i++)hash[i]=0;
	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, hash, dataLen);
	sph_jh512_close(&ctx_jh, hash);
//print_hash(hash,32);

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, hash, dataLen);
	sph_keccak512_close(&ctx_keccak, hash);
//print_hash(hash,32);
//for(int i=0;i<32;i++)hash[i]=0;
	sph_luffa512_init(&ctx_luffa);
	sph_luffa512(&ctx_luffa, hash, dataLen);
	sph_luffa512_close(&ctx_luffa, hash);
//print_hash(hash,32);
//for(int i=0;i<32;i++)hash[i]=0;
	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, hash, dataLen);
	sph_cubehash512_close(&ctx_cubehash, hash);
//print_hash(hash,32);
	sph_shavite512_init(&ctx_shavite);
	sph_shavite512(&ctx_shavite, hash, dataLen);
	sph_shavite512_close(&ctx_shavite, hash);
//print_hash(hash,32);
	sph_simd512_init(&ctx_simd);
	sph_simd512(&ctx_simd, hash, dataLen);
	sph_simd512_close(&ctx_simd, hash);
//print_hash(hash,32);
	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, hash, dataLen);
	sph_echo512_close(&ctx_echo, hash);
//print_hash(hash,32);
//for(int i=0;i<32;i++)hash[i]=0;
	sph_hamsi512_init(&ctx_hamsi);
	sph_hamsi512(&ctx_hamsi, hash, dataLen);
	sph_hamsi512_close(&ctx_hamsi, hash);
//print_hash(hash,32);
//for(int i=0;i<32;i++)hash[i]=0;
	sph_fugue512_init(&ctx_fugue);
	sph_fugue512(&ctx_fugue, hash, dataLen);
	sph_fugue512_close(&ctx_fugue, hash);
//print_hash(hash,32);
	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, hash, dataLen);
	sph_shabal512_close(&ctx_shabal, hash);
//print_hash(hash,32);
	sph_whirlpool_init(&ctx_whirlpool);
	sph_whirlpool(&ctx_whirlpool, hash, dataLen);
	sph_whirlpool_close(&ctx_whirlpool, hash);
//print_hash(hash,32);
//for(int i=0;i<32;i++)hash[i]=0;
	sph_sha512_init(&ctx_sha512);
	sph_sha512(&ctx_sha512,(const void*) hash, dataLen);
	sph_sha512_close(&ctx_sha512,(void*) hash);
//print_hash(hash,32);
	sph_haval256_5_init(&ctx_haval);
	sph_haval256_5(&ctx_haval,(const void*) hash, dataLen);
	sph_haval256_5_close(&ctx_haval, hash);
//print_hash(hash,32);

	memset(&hash[8], 0, dataLen - 32);

	sph_blake512_init(&ctx_blake);
	sph_blake512(&ctx_blake, hash, dataLen);
	sph_blake512_close(&ctx_blake, hash);

//print_hash(hash,32);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512(&ctx_bmw, hash, dataLen);
	sph_bmw512_close(&ctx_bmw, hash);

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512(&ctx_groestl, hash, dataLen);
	sph_groestl512_close(&ctx_groestl, hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, hash, dataLen);
	sph_skein512_close(&ctx_skein, hash);

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, hash, dataLen);
	sph_jh512_close(&ctx_jh, hash);

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, hash, dataLen);
	sph_keccak512_close(&ctx_keccak, hash);

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512(&ctx_luffa, hash, dataLen);
	sph_luffa512_close(&ctx_luffa, hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, hash, dataLen);
	sph_cubehash512_close(&ctx_cubehash, hash);

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512(&ctx_shavite, hash, dataLen);
	sph_shavite512_close(&ctx_shavite, hash);

	sph_simd512_init(&ctx_simd);
	sph_simd512(&ctx_simd, hash, dataLen);
	sph_simd512_close(&ctx_simd, hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, hash, dataLen);
	sph_echo512_close(&ctx_echo, hash);

	sph_hamsi512_init(&ctx_hamsi);
	sph_hamsi512(&ctx_hamsi, hash, dataLen);
	sph_hamsi512_close(&ctx_hamsi, hash);

	sph_fugue512_init(&ctx_fugue);
	sph_fugue512(&ctx_fugue, hash, dataLen);
	sph_fugue512_close(&ctx_fugue, hash);

	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, hash, dataLen);
	sph_shabal512_close(&ctx_shabal, hash);

	sph_whirlpool_init(&ctx_whirlpool);
	sph_whirlpool(&ctx_whirlpool, hash, dataLen);
	sph_whirlpool_close(&ctx_whirlpool, hash);

	sph_sha512_init(&ctx_sha512);
	sph_sha512(&ctx_sha512,(const void*) hash, dataLen);
	sph_sha512_close(&ctx_sha512,(void*) hash);

//print_hash(hash,32);
	sph_haval256_5_init(&ctx_haval);
	sph_haval256_5(&ctx_haval,(const void*) hash, dataLen);
	sph_haval256_5_close(&ctx_haval, hash);
//print_hash(hash,8);
	memcpy(output, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };


void print_hash(unsigned int *data,int size){
for(int i=0;i<size;i++)
        gpulog(LOG_WARNING, 0,"%x ",data[i]);
gpulog(LOG_WARNING, 0,"-------------");
}


extern "C" int scanhash_x17(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done){

	int dev_id = device_map[thr_id];

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
/*
	uint32_t default_throughput = 1<<20;
	
	if (strstr(device_name[dev_id], "GTX 970")) default_throughput+=256*256*6;
	if (strstr(device_name[dev_id], "GTX 980")) default_throughput =1<<22;
	
	uint32_t throughput = cuda_default_throughput(thr_id, default_throughput); // 19=256*256*8;
*/
	uint32_t default_throughput;
	if(device_sm[dev_id]<=500) default_throughput = 1<<20;
	else if(device_sm[dev_id]<=520) default_throughput = 1<<21;
	else if(device_sm[dev_id]>520) default_throughput = (1<<22) + (1<<21);
	default_throughput = 1<<20;
	if((strstr(device_name[dev_id], "1070")))default_throughput = 1<<20;
	if((strstr(device_name[dev_id], "1080")))default_throughput = 1<<20;
	
	uint32_t throughput = cuda_default_throughput(thr_id, default_throughput); // 19=256*256*8;
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	throughput&=0xFFFFFF70; //multiples of 128 due to simd_echo kernel

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0xff;

	gpulog(LOG_INFO,thr_id,"target %x %x %x",ptarget[5], ptarget[6], ptarget[7]);
        gpulog(LOG_INFO,thr_id,"target %llx",*(uint64_t*)&ptarget[6]);

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
//			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		}
		gpulog(LOG_INFO,thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

//		x11_simd_echo_512_cpu_init(thr_id, throughput);
		x15_whirlpool_cpu_init(thr_id, throughput, 0);
		groestl512_cpu_init(thr_id, throughput);
		x11_simd512_cpu_init(thr_id, throughput);
//for(;;);
		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 8 * sizeof(uint64_t) * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
		h_resNonce[thr_id] = (uint32_t*) malloc(NBN  * 8 * sizeof(uint32_t));
		if(h_resNonce[thr_id] == NULL){
			gpulog(LOG_ERR,thr_id,"Host memory allocation failed");
			exit(EXIT_FAILURE);
		}		
		init[thr_id] = true;
	}

	uint32_t _ALIGN(64) endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
//		endiandata[k]=0;
//	print_hash(endiandata,20);
	quark_blake512_cpu_setBlock_80(thr_id, endiandata);
	cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));
//	x11_simd512_cpu_init(thr_id, throughput);
//	for(;;);
	do {
		// Hash with CUDA


		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);//A
		quark_groestl512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);

		quark_skein512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_jh512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);//A //fast
//		keccak_xevan_cpu_hash_64_A(thr_id, throughput,  d_hash[thr_id]);//A

//cudaMemset(d_hash[thr_id], 0x00, 16*sizeof(uint32_t));
//		x11_luffa512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); //P
//cudaMemcpy(h_resNonce[thr_id], &d_hash[thr_id][0], 16*sizeof(uint32_t), cudaMemcpyDeviceToHost);
//print_hash(h_resNonce[thr_id],16);
//cudaMemset(d_hash[thr_id], 0x00, 16*sizeof(uint32_t));

		x11_luffa512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);//A
//cudaMemcpy(h_resNonce[thr_id], &d_hash[thr_id][0], 16*sizeof(uint32_t), cudaMemcpyDeviceToHost);
//print_hash(h_resNonce[thr_id],16);
//for(;;);

		x11_cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); //A 256
		xevan_shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);//P slow r2
                x11_simd512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);  //A slow r3

//                cudaMemset(d_hash[thr_id], 0x00, 16*sizeof(uint32_t));


//		xevan_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); //slow r1

//                cudaMemcpy(h_resNonce[thr_id], &d_hash[thr_id][0], 16*sizeof(uint32_t), cudaMemcpyDeviceToHost);
//		print_hash(h_resNonce[thr_id],16);


  //              cudaMemset(d_hash[thr_id], 0x00, 16*sizeof(uint32_t));

		x11_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);//A


//                cudaMemcpy(h_resNonce[thr_id], &d_hash[thr_id][0], 16*sizeof(uint32_t), cudaMemcpyDeviceToHost);
 //               print_hash(h_resNonce[thr_id],16);

//for(;;);

                x13_hamsi512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); //fast
		x13_fugue512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); //fast ++
		x14_shabal512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); //fast
		xevan_whirlpool_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); //opt2
		xevan_sha512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); //fast
		xevan_haval512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); //fast


//		xevan_blake512_cpu_hash_64(thr_id, throughput,  d_hash[thr_id]);//BAD
quark_blake512_cpu_hash_128(thr_id, throughput,  d_hash[thr_id]);//BAD

//
                quark_bmw512_cpu_hash_64x(thr_id, throughput, NULL, d_hash[thr_id]);
//                xevan_groestl512_cpu_hash(thr_id, throughput, d_hash[thr_id]);
quark_groestl512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);

//                xevan_skein512(thr_id, throughput, d_hash[thr_id]);
                quark_skein512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);

                quark_jh512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
//                keccak_xevan_cpu_hash_64_A(thr_id, throughput,  d_hash[thr_id]);
//                x11_luffa512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
                x11_luffa512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);//A

                x11_cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
                xevan_shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);//move to shared
                x11_simd512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); 

//                xevan_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
                x11_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);

                x13_hamsi512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
                x13_fugue512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
                x14_shabal512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
                xevan_whirlpool_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
                xevan_sha512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);







/*
for(int i = 10000;i< 10016;i++){
                cudaMemcpy(h_resNonce[thr_id], &d_hash[thr_id][16*i], 16*sizeof(uint32_t), cudaMemcpyDeviceToHost);
print_hash(h_resNonce[thr_id],8);
}
		for(;;);

*/
		xevan_haval512_cpu_hash_64_final(thr_id, throughput, d_hash[thr_id],d_resNonce[thr_id],*(uint64_t*)&ptarget[6]);

		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);

		if (h_resNonce[thr_id][0] != UINT32_MAX){
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNounce = pdata[19];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], startNounce + h_resNonce[thr_id][0]);
			x17hash(vhash64, endiandata);
//			*hashes_done = pdata[19] - first_nonce + throughput + 1;
//			pdata[19] = startNounce + h_resNonce[thr_id][0];
			gpulog(LOG_WARNING, 0,"NONCE FOUND ");
//			return 1;
			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
				int res = 1;
				*hashes_done = pdata[19] - first_nonce + throughput + 1;
				work_set_target_ratio(work, vhash64);
				pdata[19] = startNounce + h_resNonce[thr_id][0];
				if (h_resNonce[thr_id][1] != UINT32_MAX) {
					pdata[21] = startNounce+h_resNonce[thr_id][1];
					if(!opt_quiet)
						gpulog(LOG_BLUE,dev_id,"Found 2nd nonce: %08x", pdata[21]);
					be32enc(&endiandata[19], pdata[21]);
					x17hash(vhash64, endiandata);
					if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio[0]){
						work_set_target_ratio(work, vhash64);
						xchg(pdata[19],pdata[21]);
					}
					res++;
				}
				return res;
			}
			else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", h_resNonce[thr_id][0]);
				cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));				
			}
		}

		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > (uint64_t)throughput + pdata[19]));

	*hashes_done = pdata[19] - first_nonce + 1;

	return 0;
}

// cleanup
extern "C" void free_x17(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	free(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	cudaFree(d_hash[thr_id]);

	x11_simd_echo_512_cpu_free(thr_id);
	x15_whirlpool_cpu_free(thr_id);
	cudaDeviceSynchronize();
	init[thr_id] = false;
}
