/*
 * Quick Hamsi-512 for X13
 * by tsiv - 2014
 *
 * Provos Alexis - 2016
 */

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_vectors.h"
/*
static __constant__ const uint32_t d_alpha_n[] = {
	0xff00f0f0, 0xccccaaaa, 0xf0f0cccc, 0xff00aaaa, 0xccccaaaa, 0xf0f0ff00, 0xaaaacccc, 0xf0f0ff00,	0xf0f0cccc, 0xaaaaff00, 0xccccff00, 0xaaaaf0f0, 0xaaaaf0f0, 0xff00cccc, 0xccccf0f0, 0xff00aaaa,
	0xccccaaaa, 0xff00f0f0, 0xff00aaaa, 0xf0f0cccc, 0xf0f0ff00, 0xccccaaaa, 0xf0f0ff00, 0xaaaacccc,	0xaaaaff00, 0xf0f0cccc, 0xaaaaf0f0, 0xccccff00, 0xff00cccc, 0xaaaaf0f0, 0xff00aaaa, 0xccccf0f0
};

static __constant__ const uint32_t d_alpha_f[] = {
	0xcaf9639c, 0x0ff0f9c0, 0x639c0ff0, 0xcaf9f9c0, 0x0ff0f9c0, 0x639ccaf9, 0xf9c00ff0, 0x639ccaf9,	0x639c0ff0, 0xf9c0caf9, 0x0ff0caf9, 0xf9c0639c, 0xf9c0639c, 0xcaf90ff0, 0x0ff0639c, 0xcaf9f9c0,
	0x0ff0f9c0, 0xcaf9639c, 0xcaf9f9c0, 0x639c0ff0, 0x639ccaf9, 0x0ff0f9c0, 0x639ccaf9, 0xf9c00ff0,	0xf9c0caf9, 0x639c0ff0, 0xf9c0639c, 0x0ff0caf9, 0xcaf90ff0, 0xf9c0639c, 0xcaf9f9c0, 0x0ff0639c
};

static __constant__ const uint32_t c_c[] = {
		0x73746565, 0x6c706172, 0x6b204172, 0x656e6265, 0x72672031, 0x302c2062, 0x75732032, 0x3434362c,
		0x20422d33, 0x30303120, 0x4c657576, 0x656e2d48, 0x65766572, 0x6c65652c, 0x2042656c, 0x6769756d
};

static __constant__ const uint32_t d_T512[1024] = {
	0xef0b0270, 0x3afd0000, 0x5dae0000, 0x69490000, 0x9b0f3c06, 0x4405b5f9, 0x66140a51, 0x924f5d0a, 0xc96b0030, 0xe7250000, 0x2f840000, 0x264f0000, 0x08695bf9, 0x6dfcf137, 0x509f6984, 0x9e69af68,
	0xc96b0030, 0xe7250000, 0x2f840000, 0x264f0000, 0x08695bf9, 0x6dfcf137, 0x509f6984, 0x9e69af68, 0x26600240, 0xddd80000, 0x722a0000, 0x4f060000, 0x936667ff, 0x29f944ce, 0x368b63d5, 0x0c26f262,
	0x145a3c00, 0xb9e90000, 0x61270000, 0xf1610000, 0xce613d6c, 0xb0493d78, 0x47a96720, 0xe18e24c5, 0x23671400, 0xc8b90000, 0xf4c70000, 0xfb750000, 0x73cd2465, 0xf8a6a549, 0x02c40a3f, 0xdc24e61f,
	0x23671400, 0xc8b90000, 0xf4c70000, 0xfb750000, 0x73cd2465, 0xf8a6a549, 0x02c40a3f, 0xdc24e61f, 0x373d2800, 0x71500000, 0x95e00000, 0x0a140000, 0xbdac1909, 0x48ef9831, 0x456d6d1f, 0x3daac2da,
	0x54285c00, 0xeaed0000, 0xc5d60000, 0xa1c50000, 0xb3a26770, 0x94a5c4e1, 0x6bb0419d, 0x551b3782, 0x9cbb1800, 0xb0d30000, 0x92510000, 0xed930000, 0x593a4345, 0xe114d5f4, 0x430633da, 0x78cace29,
	0x9cbb1800, 0xb0d30000, 0x92510000, 0xed930000, 0x593a4345, 0xe114d5f4, 0x430633da, 0x78cace29, 0xc8934400, 0x5a3e0000, 0x57870000, 0x4c560000, 0xea982435, 0x75b11115, 0x28b67247, 0x2dd1f9ab,
	0x29449c00, 0x64e70000, 0xf24b0000, 0xc2f30000, 0x0ede4e8f, 0x56c23745, 0xf3e04259, 0x8d0d9ec4, 0x466d0c00, 0x08620000, 0xdd5d0000, 0xbadd0000, 0x6a927942, 0x441f2b93, 0x218ace6f, 0xbf2c0be2,
	0x466d0c00, 0x08620000, 0xdd5d0000, 0xbadd0000, 0x6a927942, 0x441f2b93, 0x218ace6f, 0xbf2c0be2, 0x6f299000, 0x6c850000, 0x2f160000, 0x782e0000, 0x644c37cd, 0x12dd1cd6, 0xd26a8c36, 0x32219526,
	0xf6800005, 0x3443c000, 0x24070000, 0x8f3d0000, 0x21373bfb, 0x0ab8d5ae, 0xcdc58b19, 0xd795ba31, 0xa67f0001, 0x71378000, 0x19fc0000, 0x96db0000, 0x3a8b6dfd, 0xebcaaef3, 0x2c6d478f, 0xac8e6c88,
	0xa67f0001, 0x71378000, 0x19fc0000, 0x96db0000, 0x3a8b6dfd, 0xebcaaef3, 0x2c6d478f, 0xac8e6c88, 0x50ff0004, 0x45744000, 0x3dfb0000, 0x19e60000, 0x1bbc5606, 0xe1727b5d, 0xe1a8cc96, 0x7b1bd6b9,
	0xf7750009, 0xcf3cc000, 0xc3d60000, 0x04920000, 0x029519a9, 0xf8e836ba, 0x7a87f14e, 0x9e16981a, 0xd46a0000, 0x8dc8c000, 0xa5af0000, 0x4a290000, 0xfc4e427a, 0xc9b4866c, 0x98369604, 0xf746c320,
	0xd46a0000, 0x8dc8c000, 0xa5af0000, 0x4a290000, 0xfc4e427a, 0xc9b4866c, 0x98369604, 0xf746c320, 0x231f0009, 0x42f40000, 0x66790000, 0x4ebb0000, 0xfedb5bd3, 0x315cb0d6, 0xe2b1674a, 0x69505b3a,
	0x774400f0, 0xf15a0000, 0xf5b20000, 0x34140000, 0x89377e8c, 0x5a8bec25, 0x0bc3cd1e, 0xcf3775cb, 0xf46c0050, 0x96180000, 0x14a50000, 0x031f0000, 0x42947eb8, 0x66bf7e19, 0x9ca470d2, 0x8a341574,
	0xf46c0050, 0x96180000, 0x14a50000, 0x031f0000, 0x42947eb8, 0x66bf7e19, 0x9ca470d2, 0x8a341574, 0x832800a0, 0x67420000, 0xe1170000, 0x370b0000, 0xcba30034, 0x3c34923c, 0x9767bdcc, 0x450360bf,
	0xe8870170, 0x9d720000, 0x12db0000, 0xd4220000, 0xf2886b27, 0xa921e543, 0x4ef8b518, 0x618813b1, 0xb4370060, 0x0c4c0000, 0x56c20000, 0x5cae0000, 0x94541f3f, 0x3b3ef825, 0x1b365f3d, 0xf3d45758,
	0xb4370060, 0x0c4c0000, 0x56c20000, 0x5cae0000, 0x94541f3f, 0x3b3ef825, 0x1b365f3d, 0xf3d45758, 0x5cb00110, 0x913e0000, 0x44190000, 0x888c0000, 0x66dc7418, 0x921f1d66, 0x55ceea25, 0x925c44e9,
	0x0c720000, 0x49e50f00, 0x42790000, 0x5cea0000, 0x33aa301a, 0x15822514, 0x95a34b7b, 0xb44b0090, 0xfe220000, 0xa7580500, 0x25d10000, 0xf7600000, 0x893178da, 0x1fd4f860, 0x4ed0a315, 0xa123ff9f,
	0xfe220000, 0xa7580500, 0x25d10000, 0xf7600000, 0x893178da, 0x1fd4f860, 0x4ed0a315, 0xa123ff9f, 0xf2500000, 0xeebd0a00, 0x67a80000, 0xab8a0000, 0xba9b48c0, 0x0a56dd74, 0xdb73e86e, 0x1568ff0f,
	0x45180000, 0xa5b51700, 0xf96a0000, 0x3b480000, 0x1ecc142c, 0x231395d6, 0x16bca6b0, 0xdf33f4df, 0xb83d0000, 0x16710600, 0x379a0000, 0xf5b10000, 0x228161ac, 0xae48f145, 0x66241616, 0xc5c1eb3e,
	0xb83d0000, 0x16710600, 0x379a0000, 0xf5b10000, 0x228161ac, 0xae48f145, 0x66241616, 0xc5c1eb3e, 0xfd250000, 0xb3c41100, 0xcef00000, 0xcef90000, 0x3c4d7580, 0x8d5b6493, 0x7098b0a6, 0x1af21fe1,
	0x75a40000, 0xc28b2700, 0x94a40000, 0x90f50000, 0xfb7857e0, 0x49ce0bae, 0x1767c483, 0xaedf667e, 0xd1660000, 0x1bbc0300, 0x9eec0000, 0xf6940000, 0x03024527, 0xcf70fcf2, 0xb4431b17, 0x857f3c2b,
	0xd1660000, 0x1bbc0300, 0x9eec0000, 0xf6940000, 0x03024527, 0xcf70fcf2, 0xb4431b17, 0x857f3c2b, 0xa4c20000, 0xd9372400, 0x0a480000, 0x66610000, 0xf87a12c7, 0x86bef75c, 0xa324df94, 0x2ba05a55,
	0x75c90003, 0x0e10c000, 0xd1200000, 0xbaea0000, 0x8bc42f3e, 0x8758b757, 0xbb28761d, 0x00b72e2b, 0xeecf0001, 0x6f564000, 0xf33e0000, 0xa79e0000, 0xbdb57219, 0xb711ebc5, 0x4a3b40ba, 0xfeabf254,
	0xeecf0001, 0x6f564000, 0xf33e0000, 0xa79e0000, 0xbdb57219, 0xb711ebc5, 0x4a3b40ba, 0xfeabf254, 0x9b060002, 0x61468000, 0x221e0000, 0x1d740000, 0x36715d27, 0x30495c92, 0xf11336a7, 0xfe1cdc7f,
	0x86790000, 0x3f390002, 0xe19ae000, 0x98560000, 0x9565670e, 0x4e88c8ea, 0xd3dd4944, 0x161ddab9, 0x30b70000, 0xe5d00000, 0xf4f46000, 0x42c40000, 0x63b83d6a, 0x78ba9460, 0x21afa1ea, 0xb0a51834,
	0x30b70000, 0xe5d00000, 0xf4f46000, 0x42c40000, 0x63b83d6a, 0x78ba9460, 0x21afa1ea, 0xb0a51834, 0xb6ce0000, 0xdae90002, 0x156e8000, 0xda920000, 0xf6dd5a64, 0x36325c8a, 0xf272e8ae, 0xa6b8c28d,
	0x14190000, 0x23ca003c, 0x50df0000, 0x44b60000, 0x1b6c67b0, 0x3cf3ac75, 0x61e610b0, 0xdbcadb80, 0xe3430000, 0x3a4e0014, 0xf2c60000, 0xaa4e0000, 0xdb1e42a6, 0x256bbe15, 0x123db156, 0x3a4e99d7,
	0xe3430000, 0x3a4e0014, 0xf2c60000, 0xaa4e0000, 0xdb1e42a6, 0x256bbe15, 0x123db156, 0x3a4e99d7, 0xf75a0000, 0x19840028, 0xa2190000, 0xeef80000, 0xc0722516, 0x19981260, 0x73dba1e6, 0xe1844257,
	0x54500000, 0x0671005c, 0x25ae0000, 0x6a1e0000, 0x2ea54edf, 0x664e8512, 0xbfba18c3, 0x7e715d17, 0xbc8d0000, 0xfc3b0018, 0x19830000, 0xd10b0000, 0xae1878c4, 0x42a69856, 0x0012da37, 0x2c3b504e,
	0xbc8d0000, 0xfc3b0018, 0x19830000, 0xd10b0000, 0xae1878c4, 0x42a69856, 0x0012da37, 0x2c3b504e, 0xe8dd0000, 0xfa4a0044, 0x3c2d0000, 0xbb150000, 0x80bd361b, 0x24e81d44, 0xbfa8c2f4, 0x524a0d59,
	0x69510000, 0xd4e1009c, 0xc3230000, 0xac2f0000, 0xe4950bae, 0xcea415dc, 0x87ec287c, 0xbce1a3ce, 0xc6730000, 0xaf8d000c, 0xa4c10000, 0x218d0000, 0x23111587, 0x7913512f, 0x1d28ac88, 0x378dd173,
	0xc6730000, 0xaf8d000c, 0xa4c10000, 0x218d0000, 0x23111587, 0x7913512f, 0x1d28ac88, 0x378dd173, 0xaf220000, 0x7b6c0090, 0x67e20000, 0x8da20000, 0xc7841e29, 0xb7b744f3, 0x9ac484f4, 0x8b6c72bd,
	0xcc140000, 0xa5630000, 0x5ab90780, 0x3b500000, 0x4bd013ff, 0x879b3418, 0x694348c1, 0xca5a87fe, 0x819e0000, 0xec570000, 0x66320280, 0x95f30000, 0x5da92802, 0x48f43cbc, 0xe65aa22d, 0x8e67b7fa,
	0x819e0000, 0xec570000, 0x66320280, 0x95f30000, 0x5da92802, 0x48f43cbc, 0xe65aa22d, 0x8e67b7fa, 0x4d8a0000, 0x49340000, 0x3c8b0500, 0xaea30000, 0x16793bfd, 0xcf6f08a4, 0x8f19eaec, 0x443d3004,
	0x78230000, 0x12fc0000, 0xa93a0b80, 0x90a50000, 0x713e2879, 0x7ee98924, 0xf08ca062, 0x636f8bab, 0x02af0000, 0xb7280000, 0xba1c0300, 0x56980000, 0xba8d45d3, 0x8048c667, 0xa95c149a, 0xf4f6ea7b,
	0x02af0000, 0xb7280000, 0xba1c0300, 0x56980000, 0xba8d45d3, 0x8048c667, 0xa95c149a, 0xf4f6ea7b, 0x7a8c0000, 0xa5d40000, 0x13260880, 0xc63d0000, 0xcbb36daa, 0xfea14f43, 0x59d0b4f8, 0x979961d0,
	0xac480000, 0x1ba60000, 0x45fb1380, 0x03430000, 0x5a85316a, 0x1fb250b6, 0xfe72c7fe, 0x91e478f6, 0x1e4e0000, 0xdecf0000, 0x6df80180, 0x77240000, 0xec47079e, 0xf4a0694e, 0xcda31812, 0x98aa496e,
	0x1e4e0000, 0xdecf0000, 0x6df80180, 0x77240000, 0xec47079e, 0xf4a0694e, 0xcda31812, 0x98aa496e, 0xb2060000, 0xc5690000, 0x28031200, 0x74670000, 0xb6c236f4, 0xeb1239f8, 0x33d1dfec, 0x094e3198,
	0xaec30000, 0x9c4f0001, 0x79d1e000, 0x2c150000, 0x45cc75b3, 0x6650b736, 0xab92f78f, 0xa312567b, 0xdb250000, 0x09290000, 0x49aac000, 0x81e10000, 0xcafe6b59, 0x42793431, 0x43566b76, 0xe86cba2e,
	0xdb250000, 0x09290000, 0x49aac000, 0x81e10000, 0xcafe6b59, 0x42793431, 0x43566b76, 0xe86cba2e, 0x75e60000, 0x95660001, 0x307b2000, 0xadf40000, 0x8f321eea, 0x24298307, 0xe8c49cf9, 0x4b7eec55,
	0x58430000, 0x807e0000, 0x78330001, 0xc66b3800, 0xe7375cdc, 0x79ad3fdd, 0xac73fe6f, 0x3a4479b1, 0x1d5a0000, 0x2b720000, 0x488d0000, 0xaf611800, 0x25cb2ec5, 0xc879bfd0, 0x81a20429, 0x1e7536a6,
	0x1d5a0000, 0x2b720000, 0x488d0000, 0xaf611800, 0x25cb2ec5, 0xc879bfd0, 0x81a20429, 0x1e7536a6, 0x45190000, 0xab0c0000, 0x30be0001, 0x690a2000, 0xc2fc7219, 0xb1d4800d, 0x2dd1fa46, 0x24314f17,
	0xa53b0000, 0x14260000, 0x4e30001e, 0x7cae0000, 0x8f9e0dd5, 0x78dfaa3d, 0xf73168d8, 0x0b1b4946, 0x07ed0000, 0xb2500000, 0x8774000a, 0x970d0000, 0x437223ae, 0x48c76ea4, 0xf4786222, 0x9075b1ce,
	0x07ed0000, 0xb2500000, 0x8774000a, 0x970d0000, 0x437223ae, 0x48c76ea4, 0xf4786222, 0x9075b1ce, 0xa2d60000, 0xa6760000, 0xc9440014, 0xeba30000, 0xccec2e7b, 0x3018c499, 0x03490afa, 0x9b6ef888,
	0x88980000, 0x1f940000, 0x7fcf002e, 0xfb4e0000, 0xf158079a, 0x61ae9167, 0xa895706c, 0xe6107494, 0x0bc20000, 0xdb630000, 0x7e88000c, 0x15860000, 0x91fd48f3, 0x7581bb43, 0xf460449e, 0xd8b61463,
	0x0bc20000, 0xdb630000, 0x7e88000c, 0x15860000, 0x91fd48f3, 0x7581bb43, 0xf460449e, 0xd8b61463, 0x835a0000, 0xc4f70000, 0x01470022, 0xeec80000, 0x60a54f69, 0x142f2a24, 0x5cf534f2, 0x3ea660f7,
	0x52500000, 0x29540000, 0x6a61004e, 0xf0ff0000, 0x9a317eec, 0x452341ce, 0xcf568fe5, 0x5303130f, 0x538d0000, 0xa9fc0000, 0x9ef70006, 0x56ff0000, 0x0ae4004e, 0x92c5cdf9, 0xa9444018, 0x7f975691,
	0x538d0000, 0xa9fc0000, 0x9ef70006, 0x56ff0000, 0x0ae4004e, 0x92c5cdf9, 0xa9444018, 0x7f975691, 0x01dd0000, 0x80a80000, 0xf4960048, 0xa6000000, 0x90d57ea2, 0xd7e68c37, 0x6612cffd, 0x2c94459e,
	0xe6280000, 0x4c4b0000, 0xa8550000, 0xd3d002e0, 0xd86130b8, 0x98a7b0da, 0x289506b4, 0xd75a4897, 0xf0c50000, 0x59230000, 0x45820000, 0xe18d00c0, 0x3b6d0631, 0xc2ed5699, 0xcbe0fe1c, 0x56a7b19f,
	0xf0c50000, 0x59230000, 0x45820000, 0xe18d00c0, 0x3b6d0631, 0xc2ed5699, 0xcbe0fe1c, 0x56a7b19f, 0x16ed0000, 0x15680000, 0xedd70000, 0x325d0220, 0xe30c3689, 0x5a4ae643, 0xe375f8a8, 0x81fdf908,
	0xb4310000, 0x77330000, 0xb15d0000, 0x7fd004e0, 0x78a26138, 0xd116c35d, 0xd256d489, 0x4e6f74de, 0xe3060000, 0xbdc10000, 0x87130000, 0xbff20060, 0x2eba0a1a, 0x8db53751, 0x73c5ab06, 0x5bd61539,
	0xe3060000, 0xbdc10000, 0x87130000, 0xbff20060, 0x2eba0a1a, 0x8db53751, 0x73c5ab06, 0x5bd61539, 0x57370000, 0xcaf20000, 0x364e0000, 0xc0220480, 0x56186b22, 0x5ca3f40c, 0xa1937f8f, 0x15b961e7,
	0x02f20000, 0xa2810000, 0x873f0000, 0xe36c7800, 0x1e1d74ef, 0x073d2bd6, 0xc4c23237, 0x7f32259e, 0xbadd0000, 0x13ad0000, 0xb7e70000, 0xf7282800, 0xdf45144d, 0x361ac33a, 0xea5a8d14, 0x2a2c18f0,
	0xbadd0000, 0x13ad0000, 0xb7e70000, 0xf7282800, 0xdf45144d, 0x361ac33a, 0xea5a8d14, 0x2a2c18f0, 0xb82f0000, 0xb12c0000, 0x30d80000, 0x14445000, 0xc15860a2, 0x3127e8ec, 0x2e98bf23, 0x551e3d6e,
	0x1e6c0000, 0xc4420000, 0x8a2e0000, 0xbcb6b800, 0x2c4413b6, 0x8bfdd3da, 0x6a0c1bc8, 0xb99dc2eb, 0x92560000, 0x1eda0000, 0xea510000, 0xe8b13000, 0xa93556a5, 0xebfb6199, 0xb15c2254, 0x33c5244f,
	0x92560000, 0x1eda0000, 0xea510000, 0xe8b13000, 0xa93556a5, 0xebfb6199, 0xb15c2254, 0x33c5244f, 0x8c3a0000, 0xda980000, 0x607f0000, 0x54078800, 0x85714513, 0x6006b243, 0xdb50399c, 0x8a58e6a4,
	0x033d0000, 0x08b30000, 0xf33a0000, 0x3ac20007, 0x51298a50, 0x6b6e661f, 0x0ea5cfe3, 0xe6da7ffe, 0xa8da0000, 0x96be0000, 0x5c1d0000, 0x07da0002, 0x7d669583, 0x1f98708a, 0xbb668808, 0xda878000,
	0xa8da0000, 0x96be0000, 0x5c1d0000, 0x07da0002, 0x7d669583, 0x1f98708a, 0xbb668808, 0xda878000, 0xabe70000, 0x9e0d0000, 0xaf270000, 0x3d180005, 0x2c4f1fd3, 0x74f61695, 0xb5c347eb, 0x3c5dfffe,
	0x01930000, 0xe7820000, 0xedfb0000, 0xcf0c000b, 0x8dd08d58, 0xbca3b42e, 0x063661e1, 0x536f9e7b, 0x92280000, 0xdc850000, 0x57fa0000, 0x56dc0003, 0xbae92316, 0x5aefa30c, 0x90cef752, 0x7b1675d7,
	0x92280000, 0xdc850000, 0x57fa0000, 0x56dc0003, 0xbae92316, 0x5aefa30c, 0x90cef752, 0x7b1675d7, 0x93bb0000, 0x3b070000, 0xba010000, 0x99d00008, 0x3739ae4e, 0xe64c1722, 0x96f896b3, 0x2879ebac,
	0x5fa80000, 0x56030000, 0x43ae0000, 0x64f30013, 0x257e86bf, 0x1311944e, 0x541e95bf, 0x8ea4db69, 0x00440000, 0x7f480000, 0xda7c0000, 0x2a230001, 0x3badc9cc, 0xa9b69c87, 0x030a9e60, 0xbe0a679e,
	0x00440000, 0x7f480000, 0xda7c0000, 0x2a230001, 0x3badc9cc, 0xa9b69c87, 0x030a9e60, 0xbe0a679e, 0x5fec0000, 0x294b0000, 0x99d20000, 0x4ed00012, 0x1ed34f73, 0xbaa708c9, 0x57140bdf, 0x30aebcf7,
	0xee930000, 0xd6070000, 0x92c10000, 0x2b9801e0, 0x9451287c, 0x3b6cfb57, 0x45312374, 0x201f6a64, 0x7b280000, 0x57420000, 0xa9e50000, 0x634300a0, 0x9edb442f, 0x6d9995bb, 0x27f83b03, 0xc7ff60f0,
	0x7b280000, 0x57420000, 0xa9e50000, 0x634300a0, 0x9edb442f, 0x6d9995bb, 0x27f83b03, 0xc7ff60f0, 0x95bb0000, 0x81450000, 0x3b240000, 0x48db0140, 0x0a8a6c53, 0x56f56eec, 0x62c91877, 0xe7e00a94
};

#define SBOX(a, b, c, d) { \
		uint32_t t; \
		t =(a); \
		a =(a & c) ^ d; \
		c =(c ^ b) ^ a; \
		d =(d | t) ^ b; \
		b = d; \
		d =((d | (t ^ c)) ^ a); \
		a&= b; \
		t^=(c ^ a); \
		b = b ^ d ^ t; \
		(a) = (c); \
		(c) = (b); \
		(b) = (d); \
		(d) = (~t); \
	}

#define HAMSI_L(a, b, c, d) { \
		(a) = ROTL32(a, 13); \
		(c) = ROTL32(c, 3); \
		(b) ^= (a) ^ (c); \
		(d) ^= (c) ^ ((a) << 3); \
		(b) = ROTL32(b, 1); \
		(d) = ROTL32(d, 7); \
		(a) = ROTL32(a ^ b ^ d, 5); \
		(c) = ROTL32(c ^ d ^ (b<<7), 22); \
	}

#define ROUND_BIG(rc, alpha) { \
		m[ 0] ^= alpha[ 0]; \
		c[ 4] ^= alpha[ 8]; \
		m[ 8] ^= alpha[16]; \
		c[12] ^= alpha[24]; \
		m[ 1] ^= alpha[ 1] ^ (rc); \
		c[ 5] ^= alpha[ 9]; \
		m[ 9] ^= alpha[17]; \
		c[13] ^= alpha[25]; \
		c[ 0] ^= alpha[ 2]; \
		m[ 4] ^= alpha[10]; \
		c[ 8] ^= alpha[18]; \
		m[12] ^= alpha[26]; \
		c[ 1] ^= alpha[ 3]; \
		m[ 5] ^= alpha[11]; \
		c[ 9] ^= alpha[19]; \
		m[13] ^= alpha[27]; \
		m[ 2] ^= alpha[ 4]; \
		c[ 6] ^= alpha[12]; \
		m[10] ^= alpha[20]; \
		c[14] ^= alpha[28]; \
		m[ 3] ^= alpha[ 5]; \
		c[ 7] ^= alpha[13]; \
		m[11] ^= alpha[21]; \
		c[15] ^= alpha[29]; \
		c[ 2] ^= alpha[ 6]; \
		m[ 6] ^= alpha[14]; \
		c[10] ^= alpha[22]; \
		m[14] ^= alpha[30]; \
		c[ 3] ^= alpha[ 7]; \
		m[ 7] ^= alpha[15]; \
		c[11] ^= alpha[23]; \
		m[15] ^= alpha[31]; \
		SBOX(m[ 0], c[ 4], m[ 8], c[12]); \
		SBOX(m[ 1], c[ 5], m[ 9], c[13]); \
		SBOX(c[ 0], m[ 4], c[ 8], m[12]); \
		SBOX(c[ 1], m[ 5], c[ 9], m[13]); \
		HAMSI_L(m[ 0], c[ 5], c[ 8], m[13]); \
		SBOX(m[ 2], c[ 6], m[10], c[14]); \
		HAMSI_L(m[ 1], m[ 4], c[ 9], c[14]); \
		SBOX(m[ 3], c[ 7], m[11], c[15]); \
		HAMSI_L(c[ 0], m[ 5], m[10], c[15]); \
		SBOX(c[ 2], m[ 6], c[10], m[14]); \
		HAMSI_L(c[ 1], c[ 6], m[11], m[14]); \
		SBOX(c[ 3], m[ 7], c[11], m[15]); \
		HAMSI_L(m[ 2], c[ 7], c[10], m[15]); \
		HAMSI_L(m[ 3], m[ 6], c[11], c[12]); \
		HAMSI_L(c[ 2], m[ 7], m[ 8], c[13]); \
		HAMSI_L(c[ 3], c[ 4], m[ 9], m[12]); \
		HAMSI_L(m[ 0], c[ 0], m[ 3], c[ 3]); \
		HAMSI_L(m[ 8], c[ 9], m[11], c[10]); \
		HAMSI_L(c[ 5], m[ 5], c[ 6], m[ 6]); \
		HAMSI_L(c[13], m[12], c[14], m[15]); \
	}

*/
#define sph_u32 uint32_t
__constant__ static const sph_u32 HAMSI_IV512[] = {
  SPH_C32(0x73746565), SPH_C32(0x6c706172), SPH_C32(0x6b204172),
  SPH_C32(0x656e6265), SPH_C32(0x72672031), SPH_C32(0x302c2062),
  SPH_C32(0x75732032), SPH_C32(0x3434362c), SPH_C32(0x20422d33),
  SPH_C32(0x30303120), SPH_C32(0x4c657576), SPH_C32(0x656e2d48),
  SPH_C32(0x65766572), SPH_C32(0x6c65652c), SPH_C32(0x2042656c),
  SPH_C32(0x6769756d)
};


__constant__ static const sph_u32 alpha_n[] = {
  SPH_C32(0xff00f0f0), SPH_C32(0xccccaaaa), SPH_C32(0xf0f0cccc),
  SPH_C32(0xff00aaaa), SPH_C32(0xccccaaaa), SPH_C32(0xf0f0ff00),
  SPH_C32(0xaaaacccc), SPH_C32(0xf0f0ff00), SPH_C32(0xf0f0cccc),
  SPH_C32(0xaaaaff00), SPH_C32(0xccccff00), SPH_C32(0xaaaaf0f0),
  SPH_C32(0xaaaaf0f0), SPH_C32(0xff00cccc), SPH_C32(0xccccf0f0),
  SPH_C32(0xff00aaaa), SPH_C32(0xccccaaaa), SPH_C32(0xff00f0f0),
  SPH_C32(0xff00aaaa), SPH_C32(0xf0f0cccc), SPH_C32(0xf0f0ff00),
  SPH_C32(0xccccaaaa), SPH_C32(0xf0f0ff00), SPH_C32(0xaaaacccc),
  SPH_C32(0xaaaaff00), SPH_C32(0xf0f0cccc), SPH_C32(0xaaaaf0f0),
  SPH_C32(0xccccff00), SPH_C32(0xff00cccc), SPH_C32(0xaaaaf0f0),
  SPH_C32(0xff00aaaa), SPH_C32(0xccccf0f0)
};

__constant__ static const sph_u32 alpha_f[] = {
  SPH_C32(0xcaf9639c), SPH_C32(0x0ff0f9c0), SPH_C32(0x639c0ff0),
  SPH_C32(0xcaf9f9c0), SPH_C32(0x0ff0f9c0), SPH_C32(0x639ccaf9),
  SPH_C32(0xf9c00ff0), SPH_C32(0x639ccaf9), SPH_C32(0x639c0ff0),
  SPH_C32(0xf9c0caf9), SPH_C32(0x0ff0caf9), SPH_C32(0xf9c0639c),
  SPH_C32(0xf9c0639c), SPH_C32(0xcaf90ff0), SPH_C32(0x0ff0639c),
  SPH_C32(0xcaf9f9c0), SPH_C32(0x0ff0f9c0), SPH_C32(0xcaf9639c),
  SPH_C32(0xcaf9f9c0), SPH_C32(0x639c0ff0), SPH_C32(0x639ccaf9),
  SPH_C32(0x0ff0f9c0), SPH_C32(0x639ccaf9), SPH_C32(0xf9c00ff0),
  SPH_C32(0xf9c0caf9), SPH_C32(0x639c0ff0), SPH_C32(0xf9c0639c),
  SPH_C32(0x0ff0caf9), SPH_C32(0xcaf90ff0), SPH_C32(0xf9c0639c),
  SPH_C32(0xcaf9f9c0), SPH_C32(0x0ff0639c)
};

#define HAMSI_DECL_STATE_SMALL \
  sph_u32 c0, c1, c2, c3, c4, c5, c6, c7;

#define HAMSI_READ_STATE_SMALL(sc)   do { \
    c0 = h[0x0]; \
    c1 = h[0x1]; \
    c2 = h[0x2]; \
    c3 = h[0x3]; \
    c4 = h[0x4]; \
    c5 = h[0x5]; \
    c6 = h[0x6]; \
    c7 = h[0x7]; \
  } while (0)

#define HAMSI_WRITE_STATE_SMALL(sc)   do { \
    h[0x0] = c0; \
    h[0x1] = c1; \
    h[0x2] = c2; \
    h[0x3] = c3; \
    h[0x4] = c4; \
    h[0x5] = c5; \
    h[0x6] = c6; \
    h[0x7] = c7; \
  } while (0)

#define hamsi_s0   m0
#define hamsi_s1   m1
#define hamsi_s2   c0
#define hamsi_s3   c1
#define hamsi_s4   c2
#define hamsi_s5   c3
#define hamsi_s6   m2
#define hamsi_s7   m3
#define hamsi_s8   m4
#define hamsi_s9   m5
#define hamsi_sA   c4
#define hamsi_sB   c5
#define hamsi_sC   c6
#define hamsi_sD   c7
#define hamsi_sE   m6
#define hamsi_sF   m7

#define SBOX(a, b, c, d)   do { \
    sph_u32 t; \
    t = (a); \
    (a) &= (c); \
    (a) ^= (d); \
    (c) ^= (b); \
    (c) ^= (a); \
    (d) |= t; \
    (d) ^= (b); \
    t ^= (c); \
    (b) = (d); \
    (d) |= t; \
    (d) ^= (a); \
    (a) &= (b); \
    t ^= (a); \
    (b) ^= (d); \
    (b) ^= t; \
    (a) = (c); \
    (c) = (b); \
    (b) = (d); \
    (d) = SPH_T32(~t); \
  } while (0)

#define HAMSI_L(a, b, c, d)   do { \
    (a) = SPH_ROTL32(a, 13); \
    (c) = SPH_ROTL32(c, 3); \
    (b) ^= (a) ^ (c); \
    (d) ^= (c) ^ SPH_T32((a) << 3); \
    (b) = SPH_ROTL32(b, 1); \
    (d) = SPH_ROTL32(d, 7); \
    (a) ^= (b) ^ (d); \
    (c) ^= (d) ^ SPH_T32((b) << 7); \
    (a) = SPH_ROTL32(a, 5); \
    (c) = SPH_ROTL32(c, 22); \
  } while (0)

#define ROUND_SMALL(rc, alpha)   do { \
    hamsi_s0 ^= alpha[0x00]; \
    hamsi_s1 ^= alpha[0x01] ^ (sph_u32)(rc); \
    hamsi_s2 ^= alpha[0x02]; \
    hamsi_s3 ^= alpha[0x03]; \
    hamsi_s4 ^= alpha[0x08]; \
    hamsi_s5 ^= alpha[0x09]; \
    hamsi_s6 ^= alpha[0x0A]; \
    hamsi_s7 ^= alpha[0x0B]; \
    hamsi_s8 ^= alpha[0x10]; \
    hamsi_s9 ^= alpha[0x11]; \
    hamsi_sA ^= alpha[0x12]; \
    hamsi_sB ^= alpha[0x13]; \
    hamsi_sC ^= alpha[0x18]; \
    hamsi_sD ^= alpha[0x19]; \
    hamsi_sE ^= alpha[0x1A]; \
    hamsi_sF ^= alpha[0x1B]; \
    SBOX(hamsi_s0, hamsi_s4, hamsi_s8, hamsi_sC); \
    SBOX(hamsi_s1, hamsi_s5, hamsi_s9, hamsi_sD); \
    SBOX(hamsi_s2, hamsi_s6, hamsi_sA, hamsi_sE); \
    SBOX(hamsi_s3, hamsi_s7, hamsi_sB, hamsi_sF); \
    HAMSI_L(hamsi_s0, hamsi_s5, hamsi_sA, hamsi_sF); \
    HAMSI_L(hamsi_s1, hamsi_s6, hamsi_sB, hamsi_sC); \
    HAMSI_L(hamsi_s2, hamsi_s7, hamsi_s8, hamsi_sD); \
    HAMSI_L(hamsi_s3, hamsi_s4, hamsi_s9, hamsi_sE); \
  } while (0)

#define P_SMALL   do { \
    ROUND_SMALL(0, alpha_n); \
    ROUND_SMALL(1, alpha_n); \
    ROUND_SMALL(2, alpha_n); \
  } while (0)

#define PF_SMALL   do { \
    ROUND_SMALL(0, alpha_f); \
    ROUND_SMALL(1, alpha_f); \
    ROUND_SMALL(2, alpha_f); \
    ROUND_SMALL(3, alpha_f); \
    ROUND_SMALL(4, alpha_f); \
    ROUND_SMALL(5, alpha_f); \
  } while (0)

#define T_SMALL   do { \
    /* order is important */ \
    c7 = (h[7] ^= hamsi_sB); \
    c6 = (h[6] ^= hamsi_sA); \
    c5 = (h[5] ^= hamsi_s9); \
    c4 = (h[4] ^= hamsi_s8); \
    c3 = (h[3] ^= hamsi_s3); \
    c2 = (h[2] ^= hamsi_s2); \
    c1 = (h[1] ^= hamsi_s1); \
    c0 = (h[0] ^= hamsi_s0); \
  } while (0)

#define hamsi_s00   m0
#define hamsi_s01   m1
#define hamsi_s02   c0
#define hamsi_s03   c1
#define hamsi_s04   m2
#define hamsi_s05   m3
#define hamsi_s06   c2
#define hamsi_s07   c3
#define hamsi_s08   c4
#define hamsi_s09   c5
#define hamsi_s0A   m4
#define hamsi_s0B   m5
#define hamsi_s0C   c6
#define hamsi_s0D   c7
#define hamsi_s0E   m6
#define hamsi_s0F   m7
#define hamsi_s10   m8
#define hamsi_s11   m9
#define hamsi_s12   c8
#define hamsi_s13   c9
#define hamsi_s14   mA
#define hamsi_s15   mB
#define hamsi_s16   cA
#define hamsi_s17   cB
#define hamsi_s18   cC
#define hamsi_s19   cD
#define hamsi_s1A   mC
#define hamsi_s1B   mD
#define hamsi_s1C   cE
#define hamsi_s1D   cF
#define hamsi_s1E   mE
#define hamsi_s1F   mF

#define ROUND_BIG(rc, alpha)   do { \
    hamsi_s00 ^= alpha[0x00]; \
    hamsi_s01 ^= alpha[0x01] ^ (sph_u32)(rc); \
    hamsi_s02 ^= alpha[0x02]; \
    hamsi_s03 ^= alpha[0x03]; \
    hamsi_s04 ^= alpha[0x04]; \
    hamsi_s05 ^= alpha[0x05]; \
    hamsi_s06 ^= alpha[0x06]; \
    hamsi_s07 ^= alpha[0x07]; \
    hamsi_s08 ^= alpha[0x08]; \
    hamsi_s09 ^= alpha[0x09]; \
    hamsi_s0A ^= alpha[0x0A]; \
    hamsi_s0B ^= alpha[0x0B]; \
    hamsi_s0C ^= alpha[0x0C]; \
    hamsi_s0D ^= alpha[0x0D]; \
    hamsi_s0E ^= alpha[0x0E]; \
    hamsi_s0F ^= alpha[0x0F]; \
    hamsi_s10 ^= alpha[0x10]; \
    hamsi_s11 ^= alpha[0x11]; \
    hamsi_s12 ^= alpha[0x12]; \
    hamsi_s13 ^= alpha[0x13]; \
    hamsi_s14 ^= alpha[0x14]; \
    hamsi_s15 ^= alpha[0x15]; \
    hamsi_s16 ^= alpha[0x16]; \
    hamsi_s17 ^= alpha[0x17]; \
    hamsi_s18 ^= alpha[0x18]; \
    hamsi_s19 ^= alpha[0x19]; \
    hamsi_s1A ^= alpha[0x1A]; \
    hamsi_s1B ^= alpha[0x1B]; \
    hamsi_s1C ^= alpha[0x1C]; \
    hamsi_s1D ^= alpha[0x1D]; \
    hamsi_s1E ^= alpha[0x1E]; \
    hamsi_s1F ^= alpha[0x1F]; \
    SBOX(hamsi_s00, hamsi_s08, hamsi_s10, hamsi_s18); \
    SBOX(hamsi_s01, hamsi_s09, hamsi_s11, hamsi_s19); \
    SBOX(hamsi_s02, hamsi_s0A, hamsi_s12, hamsi_s1A); \
    SBOX(hamsi_s03, hamsi_s0B, hamsi_s13, hamsi_s1B); \
    SBOX(hamsi_s04, hamsi_s0C, hamsi_s14, hamsi_s1C); \
    SBOX(hamsi_s05, hamsi_s0D, hamsi_s15, hamsi_s1D); \
    SBOX(hamsi_s06, hamsi_s0E, hamsi_s16, hamsi_s1E); \
    SBOX(hamsi_s07, hamsi_s0F, hamsi_s17, hamsi_s1F); \
    HAMSI_L(hamsi_s00, hamsi_s09, hamsi_s12, hamsi_s1B); \
    HAMSI_L(hamsi_s01, hamsi_s0A, hamsi_s13, hamsi_s1C); \
    HAMSI_L(hamsi_s02, hamsi_s0B, hamsi_s14, hamsi_s1D); \
    HAMSI_L(hamsi_s03, hamsi_s0C, hamsi_s15, hamsi_s1E); \
    HAMSI_L(hamsi_s04, hamsi_s0D, hamsi_s16, hamsi_s1F); \
    HAMSI_L(hamsi_s05, hamsi_s0E, hamsi_s17, hamsi_s18); \
    HAMSI_L(hamsi_s06, hamsi_s0F, hamsi_s10, hamsi_s19); \
    HAMSI_L(hamsi_s07, hamsi_s08, hamsi_s11, hamsi_s1A); \
    HAMSI_L(hamsi_s00, hamsi_s02, hamsi_s05, hamsi_s07); \
    HAMSI_L(hamsi_s10, hamsi_s13, hamsi_s15, hamsi_s16); \
    HAMSI_L(hamsi_s09, hamsi_s0B, hamsi_s0C, hamsi_s0E); \
    HAMSI_L(hamsi_s19, hamsi_s1A, hamsi_s1C, hamsi_s1F); \
  } while (0)


#define P_BIG   do { \
    ROUND_BIG(0, alpha_n); \
    ROUND_BIG(1, alpha_n); \
    ROUND_BIG(2, alpha_n); \
    ROUND_BIG(3, alpha_n); \
    ROUND_BIG(4, alpha_n); \
    ROUND_BIG(5, alpha_n); \
  } while (0)

#define PF_BIG   do { \
    ROUND_BIG(0, alpha_f); \
    ROUND_BIG(1, alpha_f); \
    ROUND_BIG(2, alpha_f); \
    ROUND_BIG(3, alpha_f); \
    ROUND_BIG(4, alpha_f); \
    ROUND_BIG(5, alpha_f); \
    ROUND_BIG(6, alpha_f); \
    ROUND_BIG(7, alpha_f); \
    ROUND_BIG(8, alpha_f); \
    ROUND_BIG(9, alpha_f); \
    ROUND_BIG(10, alpha_f); \
    ROUND_BIG(11, alpha_f); \
  } while (0)

#define T_BIG   do { \
    /* order is important */ \
    cF = (h[0xF] ^= hamsi_s17); \
    cE = (h[0xE] ^= hamsi_s16); \
    cD = (h[0xD] ^= hamsi_s15); \
    cC = (h[0xC] ^= hamsi_s14); \
    cB = (h[0xB] ^= hamsi_s13); \
    cA = (h[0xA] ^= hamsi_s12); \
    c9 = (h[0x9] ^= hamsi_s11); \
    c8 = (h[0x8] ^= hamsi_s10); \
    c7 = (h[0x7] ^= hamsi_s07); \
    c6 = (h[0x6] ^= hamsi_s06); \
    c5 = (h[0x5] ^= hamsi_s05); \
    c4 = (h[0x4] ^= hamsi_s04); \
    c3 = (h[0x3] ^= hamsi_s03); \
    c2 = (h[0x2] ^= hamsi_s02); \
    c1 = (h[0x1] ^= hamsi_s01); \
    c0 = (h[0x0] ^= hamsi_s00); \
  } while (0)


#define SPH_ROTL32 ROTL32


/*
__constant__ static const sph_u32 T512[64][16] = {
  { SPH_C32(0xef0b0270), SPH_C32(0x3afd0000), SPH_C32(0x5dae0000),
    SPH_C32(0x69490000), SPH_C32(0x9b0f3c06), SPH_C32(0x4405b5f9),
    SPH_C32(0x66140a51), SPH_C32(0x924f5d0a), SPH_C32(0xc96b0030),
    SPH_C32(0xe7250000), SPH_C32(0x2f840000), SPH_C32(0x264f0000),
    SPH_C32(0x08695bf9), SPH_C32(0x6dfcf137), SPH_C32(0x509f6984),
    SPH_C32(0x9e69af68) },
  { SPH_C32(0xc96b0030), SPH_C32(0xe7250000), SPH_C32(0x2f840000),
    SPH_C32(0x264f0000), SPH_C32(0x08695bf9), SPH_C32(0x6dfcf137),
    SPH_C32(0x509f6984), SPH_C32(0x9e69af68), SPH_C32(0x26600240),
    SPH_C32(0xddd80000), SPH_C32(0x722a0000), SPH_C32(0x4f060000),
    SPH_C32(0x936667ff), SPH_C32(0x29f944ce), SPH_C32(0x368b63d5),
    SPH_C32(0x0c26f262) },
  { SPH_C32(0x145a3c00), SPH_C32(0xb9e90000), SPH_C32(0x61270000),
    SPH_C32(0xf1610000), SPH_C32(0xce613d6c), SPH_C32(0xb0493d78),
    SPH_C32(0x47a96720), SPH_C32(0xe18e24c5), SPH_C32(0x23671400),
    SPH_C32(0xc8b90000), SPH_C32(0xf4c70000), SPH_C32(0xfb750000),
    SPH_C32(0x73cd2465), SPH_C32(0xf8a6a549), SPH_C32(0x02c40a3f),
    SPH_C32(0xdc24e61f) },
  { SPH_C32(0x23671400), SPH_C32(0xc8b90000), SPH_C32(0xf4c70000),
    SPH_C32(0xfb750000), SPH_C32(0x73cd2465), SPH_C32(0xf8a6a549),
    SPH_C32(0x02c40a3f), SPH_C32(0xdc24e61f), SPH_C32(0x373d2800),
    SPH_C32(0x71500000), SPH_C32(0x95e00000), SPH_C32(0x0a140000),
    SPH_C32(0xbdac1909), SPH_C32(0x48ef9831), SPH_C32(0x456d6d1f),
    SPH_C32(0x3daac2da) },
  { SPH_C32(0x54285c00), SPH_C32(0xeaed0000), SPH_C32(0xc5d60000),
    SPH_C32(0xa1c50000), SPH_C32(0xb3a26770), SPH_C32(0x94a5c4e1),
    SPH_C32(0x6bb0419d), SPH_C32(0x551b3782), SPH_C32(0x9cbb1800),
    SPH_C32(0xb0d30000), SPH_C32(0x92510000), SPH_C32(0xed930000),
    SPH_C32(0x593a4345), SPH_C32(0xe114d5f4), SPH_C32(0x430633da),
    SPH_C32(0x78cace29) },
  { SPH_C32(0x9cbb1800), SPH_C32(0xb0d30000), SPH_C32(0x92510000),
    SPH_C32(0xed930000), SPH_C32(0x593a4345), SPH_C32(0xe114d5f4),
    SPH_C32(0x430633da), SPH_C32(0x78cace29), SPH_C32(0xc8934400),
    SPH_C32(0x5a3e0000), SPH_C32(0x57870000), SPH_C32(0x4c560000),
    SPH_C32(0xea982435), SPH_C32(0x75b11115), SPH_C32(0x28b67247),
    SPH_C32(0x2dd1f9ab) },
  { SPH_C32(0x29449c00), SPH_C32(0x64e70000), SPH_C32(0xf24b0000),
    SPH_C32(0xc2f30000), SPH_C32(0x0ede4e8f), SPH_C32(0x56c23745),
    SPH_C32(0xf3e04259), SPH_C32(0x8d0d9ec4), SPH_C32(0x466d0c00),
    SPH_C32(0x08620000), SPH_C32(0xdd5d0000), SPH_C32(0xbadd0000),
    SPH_C32(0x6a927942), SPH_C32(0x441f2b93), SPH_C32(0x218ace6f),
    SPH_C32(0xbf2c0be2) },
  { SPH_C32(0x466d0c00), SPH_C32(0x08620000), SPH_C32(0xdd5d0000),
    SPH_C32(0xbadd0000), SPH_C32(0x6a927942), SPH_C32(0x441f2b93),
    SPH_C32(0x218ace6f), SPH_C32(0xbf2c0be2), SPH_C32(0x6f299000),
    SPH_C32(0x6c850000), SPH_C32(0x2f160000), SPH_C32(0x782e0000),
    SPH_C32(0x644c37cd), SPH_C32(0x12dd1cd6), SPH_C32(0xd26a8c36),
    SPH_C32(0x32219526) },
  { SPH_C32(0xf6800005), SPH_C32(0x3443c000), SPH_C32(0x24070000),
    SPH_C32(0x8f3d0000), SPH_C32(0x21373bfb), SPH_C32(0x0ab8d5ae),
    SPH_C32(0xcdc58b19), SPH_C32(0xd795ba31), SPH_C32(0xa67f0001),
    SPH_C32(0x71378000), SPH_C32(0x19fc0000), SPH_C32(0x96db0000),
    SPH_C32(0x3a8b6dfd), SPH_C32(0xebcaaef3), SPH_C32(0x2c6d478f),
    SPH_C32(0xac8e6c88) },
  { SPH_C32(0xa67f0001), SPH_C32(0x71378000), SPH_C32(0x19fc0000),
    SPH_C32(0x96db0000), SPH_C32(0x3a8b6dfd), SPH_C32(0xebcaaef3),
    SPH_C32(0x2c6d478f), SPH_C32(0xac8e6c88), SPH_C32(0x50ff0004),
    SPH_C32(0x45744000), SPH_C32(0x3dfb0000), SPH_C32(0x19e60000),
    SPH_C32(0x1bbc5606), SPH_C32(0xe1727b5d), SPH_C32(0xe1a8cc96),
    SPH_C32(0x7b1bd6b9) },
  { SPH_C32(0xf7750009), SPH_C32(0xcf3cc000), SPH_C32(0xc3d60000),
    SPH_C32(0x04920000), SPH_C32(0x029519a9), SPH_C32(0xf8e836ba),
    SPH_C32(0x7a87f14e), SPH_C32(0x9e16981a), SPH_C32(0xd46a0000),
    SPH_C32(0x8dc8c000), SPH_C32(0xa5af0000), SPH_C32(0x4a290000),
    SPH_C32(0xfc4e427a), SPH_C32(0xc9b4866c), SPH_C32(0x98369604),
    SPH_C32(0xf746c320) },
  { SPH_C32(0xd46a0000), SPH_C32(0x8dc8c000), SPH_C32(0xa5af0000),
    SPH_C32(0x4a290000), SPH_C32(0xfc4e427a), SPH_C32(0xc9b4866c),
    SPH_C32(0x98369604), SPH_C32(0xf746c320), SPH_C32(0x231f0009),
    SPH_C32(0x42f40000), SPH_C32(0x66790000), SPH_C32(0x4ebb0000),
    SPH_C32(0xfedb5bd3), SPH_C32(0x315cb0d6), SPH_C32(0xe2b1674a),
    SPH_C32(0x69505b3a) },
  { SPH_C32(0x774400f0), SPH_C32(0xf15a0000), SPH_C32(0xf5b20000),
    SPH_C32(0x34140000), SPH_C32(0x89377e8c), SPH_C32(0x5a8bec25),
    SPH_C32(0x0bc3cd1e), SPH_C32(0xcf3775cb), SPH_C32(0xf46c0050),
    SPH_C32(0x96180000), SPH_C32(0x14a50000), SPH_C32(0x031f0000),
    SPH_C32(0x42947eb8), SPH_C32(0x66bf7e19), SPH_C32(0x9ca470d2),
    SPH_C32(0x8a341574) },
  { SPH_C32(0xf46c0050), SPH_C32(0x96180000), SPH_C32(0x14a50000),
    SPH_C32(0x031f0000), SPH_C32(0x42947eb8), SPH_C32(0x66bf7e19),
    SPH_C32(0x9ca470d2), SPH_C32(0x8a341574), SPH_C32(0x832800a0),
    SPH_C32(0x67420000), SPH_C32(0xe1170000), SPH_C32(0x370b0000),
    SPH_C32(0xcba30034), SPH_C32(0x3c34923c), SPH_C32(0x9767bdcc),
    SPH_C32(0x450360bf) },
  { SPH_C32(0xe8870170), SPH_C32(0x9d720000), SPH_C32(0x12db0000),
    SPH_C32(0xd4220000), SPH_C32(0xf2886b27), SPH_C32(0xa921e543),
    SPH_C32(0x4ef8b518), SPH_C32(0x618813b1), SPH_C32(0xb4370060),
    SPH_C32(0x0c4c0000), SPH_C32(0x56c20000), SPH_C32(0x5cae0000),
    SPH_C32(0x94541f3f), SPH_C32(0x3b3ef825), SPH_C32(0x1b365f3d),
    SPH_C32(0xf3d45758) },
  { SPH_C32(0xb4370060), SPH_C32(0x0c4c0000), SPH_C32(0x56c20000),
    SPH_C32(0x5cae0000), SPH_C32(0x94541f3f), SPH_C32(0x3b3ef825),
    SPH_C32(0x1b365f3d), SPH_C32(0xf3d45758), SPH_C32(0x5cb00110),
    SPH_C32(0x913e0000), SPH_C32(0x44190000), SPH_C32(0x888c0000),
    SPH_C32(0x66dc7418), SPH_C32(0x921f1d66), SPH_C32(0x55ceea25),
    SPH_C32(0x925c44e9) },
  { SPH_C32(0x0c720000), SPH_C32(0x49e50f00), SPH_C32(0x42790000),
    SPH_C32(0x5cea0000), SPH_C32(0x33aa301a), SPH_C32(0x15822514),
    SPH_C32(0x95a34b7b), SPH_C32(0xb44b0090), SPH_C32(0xfe220000),
    SPH_C32(0xa7580500), SPH_C32(0x25d10000), SPH_C32(0xf7600000),
    SPH_C32(0x893178da), SPH_C32(0x1fd4f860), SPH_C32(0x4ed0a315),
    SPH_C32(0xa123ff9f) },
  { SPH_C32(0xfe220000), SPH_C32(0xa7580500), SPH_C32(0x25d10000),
    SPH_C32(0xf7600000), SPH_C32(0x893178da), SPH_C32(0x1fd4f860),
    SPH_C32(0x4ed0a315), SPH_C32(0xa123ff9f), SPH_C32(0xf2500000),
    SPH_C32(0xeebd0a00), SPH_C32(0x67a80000), SPH_C32(0xab8a0000),
    SPH_C32(0xba9b48c0), SPH_C32(0x0a56dd74), SPH_C32(0xdb73e86e),
    SPH_C32(0x1568ff0f) },
  { SPH_C32(0x45180000), SPH_C32(0xa5b51700), SPH_C32(0xf96a0000),
    SPH_C32(0x3b480000), SPH_C32(0x1ecc142c), SPH_C32(0x231395d6),
    SPH_C32(0x16bca6b0), SPH_C32(0xdf33f4df), SPH_C32(0xb83d0000),
    SPH_C32(0x16710600), SPH_C32(0x379a0000), SPH_C32(0xf5b10000),
    SPH_C32(0x228161ac), SPH_C32(0xae48f145), SPH_C32(0x66241616),
    SPH_C32(0xc5c1eb3e) },
  { SPH_C32(0xb83d0000), SPH_C32(0x16710600), SPH_C32(0x379a0000),
    SPH_C32(0xf5b10000), SPH_C32(0x228161ac), SPH_C32(0xae48f145),
    SPH_C32(0x66241616), SPH_C32(0xc5c1eb3e), SPH_C32(0xfd250000),
    SPH_C32(0xb3c41100), SPH_C32(0xcef00000), SPH_C32(0xcef90000),
    SPH_C32(0x3c4d7580), SPH_C32(0x8d5b6493), SPH_C32(0x7098b0a6),
    SPH_C32(0x1af21fe1) },
  { SPH_C32(0x75a40000), SPH_C32(0xc28b2700), SPH_C32(0x94a40000),
    SPH_C32(0x90f50000), SPH_C32(0xfb7857e0), SPH_C32(0x49ce0bae),
    SPH_C32(0x1767c483), SPH_C32(0xaedf667e), SPH_C32(0xd1660000),
    SPH_C32(0x1bbc0300), SPH_C32(0x9eec0000), SPH_C32(0xf6940000),
    SPH_C32(0x03024527), SPH_C32(0xcf70fcf2), SPH_C32(0xb4431b17),
    SPH_C32(0x857f3c2b) },
  { SPH_C32(0xd1660000), SPH_C32(0x1bbc0300), SPH_C32(0x9eec0000),
    SPH_C32(0xf6940000), SPH_C32(0x03024527), SPH_C32(0xcf70fcf2),
    SPH_C32(0xb4431b17), SPH_C32(0x857f3c2b), SPH_C32(0xa4c20000),
    SPH_C32(0xd9372400), SPH_C32(0x0a480000), SPH_C32(0x66610000),
    SPH_C32(0xf87a12c7), SPH_C32(0x86bef75c), SPH_C32(0xa324df94),
    SPH_C32(0x2ba05a55) },
  { SPH_C32(0x75c90003), SPH_C32(0x0e10c000), SPH_C32(0xd1200000),
    SPH_C32(0xbaea0000), SPH_C32(0x8bc42f3e), SPH_C32(0x8758b757),
    SPH_C32(0xbb28761d), SPH_C32(0x00b72e2b), SPH_C32(0xeecf0001),
    SPH_C32(0x6f564000), SPH_C32(0xf33e0000), SPH_C32(0xa79e0000),
    SPH_C32(0xbdb57219), SPH_C32(0xb711ebc5), SPH_C32(0x4a3b40ba),
    SPH_C32(0xfeabf254) },
  { SPH_C32(0xeecf0001), SPH_C32(0x6f564000), SPH_C32(0xf33e0000),
    SPH_C32(0xa79e0000), SPH_C32(0xbdb57219), SPH_C32(0xb711ebc5),
    SPH_C32(0x4a3b40ba), SPH_C32(0xfeabf254), SPH_C32(0x9b060002),
    SPH_C32(0x61468000), SPH_C32(0x221e0000), SPH_C32(0x1d740000),
    SPH_C32(0x36715d27), SPH_C32(0x30495c92), SPH_C32(0xf11336a7),
    SPH_C32(0xfe1cdc7f) },
  { SPH_C32(0x86790000), SPH_C32(0x3f390002), SPH_C32(0xe19ae000),
    SPH_C32(0x98560000), SPH_C32(0x9565670e), SPH_C32(0x4e88c8ea),
    SPH_C32(0xd3dd4944), SPH_C32(0x161ddab9), SPH_C32(0x30b70000),
    SPH_C32(0xe5d00000), SPH_C32(0xf4f46000), SPH_C32(0x42c40000),
    SPH_C32(0x63b83d6a), SPH_C32(0x78ba9460), SPH_C32(0x21afa1ea),
    SPH_C32(0xb0a51834) },
  { SPH_C32(0x30b70000), SPH_C32(0xe5d00000), SPH_C32(0xf4f46000),
    SPH_C32(0x42c40000), SPH_C32(0x63b83d6a), SPH_C32(0x78ba9460),
    SPH_C32(0x21afa1ea), SPH_C32(0xb0a51834), SPH_C32(0xb6ce0000),
    SPH_C32(0xdae90002), SPH_C32(0x156e8000), SPH_C32(0xda920000),
    SPH_C32(0xf6dd5a64), SPH_C32(0x36325c8a), SPH_C32(0xf272e8ae),
    SPH_C32(0xa6b8c28d) },
  { SPH_C32(0x14190000), SPH_C32(0x23ca003c), SPH_C32(0x50df0000),
    SPH_C32(0x44b60000), SPH_C32(0x1b6c67b0), SPH_C32(0x3cf3ac75),
    SPH_C32(0x61e610b0), SPH_C32(0xdbcadb80), SPH_C32(0xe3430000),
    SPH_C32(0x3a4e0014), SPH_C32(0xf2c60000), SPH_C32(0xaa4e0000),
    SPH_C32(0xdb1e42a6), SPH_C32(0x256bbe15), SPH_C32(0x123db156),
    SPH_C32(0x3a4e99d7) },
  { SPH_C32(0xe3430000), SPH_C32(0x3a4e0014), SPH_C32(0xf2c60000),
    SPH_C32(0xaa4e0000), SPH_C32(0xdb1e42a6), SPH_C32(0x256bbe15),
    SPH_C32(0x123db156), SPH_C32(0x3a4e99d7), SPH_C32(0xf75a0000),
    SPH_C32(0x19840028), SPH_C32(0xa2190000), SPH_C32(0xeef80000),
    SPH_C32(0xc0722516), SPH_C32(0x19981260), SPH_C32(0x73dba1e6),
    SPH_C32(0xe1844257) },
  { SPH_C32(0x54500000), SPH_C32(0x0671005c), SPH_C32(0x25ae0000),
    SPH_C32(0x6a1e0000), SPH_C32(0x2ea54edf), SPH_C32(0x664e8512),
    SPH_C32(0xbfba18c3), SPH_C32(0x7e715d17), SPH_C32(0xbc8d0000),
    SPH_C32(0xfc3b0018), SPH_C32(0x19830000), SPH_C32(0xd10b0000),
    SPH_C32(0xae1878c4), SPH_C32(0x42a69856), SPH_C32(0x0012da37),
    SPH_C32(0x2c3b504e) },
  { SPH_C32(0xbc8d0000), SPH_C32(0xfc3b0018), SPH_C32(0x19830000),
    SPH_C32(0xd10b0000), SPH_C32(0xae1878c4), SPH_C32(0x42a69856),
    SPH_C32(0x0012da37), SPH_C32(0x2c3b504e), SPH_C32(0xe8dd0000),
    SPH_C32(0xfa4a0044), SPH_C32(0x3c2d0000), SPH_C32(0xbb150000),
    SPH_C32(0x80bd361b), SPH_C32(0x24e81d44), SPH_C32(0xbfa8c2f4),
    SPH_C32(0x524a0d59) },
  { SPH_C32(0x69510000), SPH_C32(0xd4e1009c), SPH_C32(0xc3230000),
    SPH_C32(0xac2f0000), SPH_C32(0xe4950bae), SPH_C32(0xcea415dc),
    SPH_C32(0x87ec287c), SPH_C32(0xbce1a3ce), SPH_C32(0xc6730000),
    SPH_C32(0xaf8d000c), SPH_C32(0xa4c10000), SPH_C32(0x218d0000),
    SPH_C32(0x23111587), SPH_C32(0x7913512f), SPH_C32(0x1d28ac88),
    SPH_C32(0x378dd173) },
  { SPH_C32(0xc6730000), SPH_C32(0xaf8d000c), SPH_C32(0xa4c10000),
    SPH_C32(0x218d0000), SPH_C32(0x23111587), SPH_C32(0x7913512f),
    SPH_C32(0x1d28ac88), SPH_C32(0x378dd173), SPH_C32(0xaf220000),
    SPH_C32(0x7b6c0090), SPH_C32(0x67e20000), SPH_C32(0x8da20000),
    SPH_C32(0xc7841e29), SPH_C32(0xb7b744f3), SPH_C32(0x9ac484f4),
    SPH_C32(0x8b6c72bd) },
  { SPH_C32(0xcc140000), SPH_C32(0xa5630000), SPH_C32(0x5ab90780),
    SPH_C32(0x3b500000), SPH_C32(0x4bd013ff), SPH_C32(0x879b3418),
    SPH_C32(0x694348c1), SPH_C32(0xca5a87fe), SPH_C32(0x819e0000),
    SPH_C32(0xec570000), SPH_C32(0x66320280), SPH_C32(0x95f30000),
    SPH_C32(0x5da92802), SPH_C32(0x48f43cbc), SPH_C32(0xe65aa22d),
    SPH_C32(0x8e67b7fa) },
  { SPH_C32(0x819e0000), SPH_C32(0xec570000), SPH_C32(0x66320280),
    SPH_C32(0x95f30000), SPH_C32(0x5da92802), SPH_C32(0x48f43cbc),
    SPH_C32(0xe65aa22d), SPH_C32(0x8e67b7fa), SPH_C32(0x4d8a0000),
    SPH_C32(0x49340000), SPH_C32(0x3c8b0500), SPH_C32(0xaea30000),
    SPH_C32(0x16793bfd), SPH_C32(0xcf6f08a4), SPH_C32(0x8f19eaec),
    SPH_C32(0x443d3004) },
  { SPH_C32(0x78230000), SPH_C32(0x12fc0000), SPH_C32(0xa93a0b80),
    SPH_C32(0x90a50000), SPH_C32(0x713e2879), SPH_C32(0x7ee98924),
    SPH_C32(0xf08ca062), SPH_C32(0x636f8bab), SPH_C32(0x02af0000),
    SPH_C32(0xb7280000), SPH_C32(0xba1c0300), SPH_C32(0x56980000),
    SPH_C32(0xba8d45d3), SPH_C32(0x8048c667), SPH_C32(0xa95c149a),
    SPH_C32(0xf4f6ea7b) },
  { SPH_C32(0x02af0000), SPH_C32(0xb7280000), SPH_C32(0xba1c0300),
    SPH_C32(0x56980000), SPH_C32(0xba8d45d3), SPH_C32(0x8048c667),
    SPH_C32(0xa95c149a), SPH_C32(0xf4f6ea7b), SPH_C32(0x7a8c0000),
    SPH_C32(0xa5d40000), SPH_C32(0x13260880), SPH_C32(0xc63d0000),
    SPH_C32(0xcbb36daa), SPH_C32(0xfea14f43), SPH_C32(0x59d0b4f8),
    SPH_C32(0x979961d0) },
  { SPH_C32(0xac480000), SPH_C32(0x1ba60000), SPH_C32(0x45fb1380),
    SPH_C32(0x03430000), SPH_C32(0x5a85316a), SPH_C32(0x1fb250b6),
    SPH_C32(0xfe72c7fe), SPH_C32(0x91e478f6), SPH_C32(0x1e4e0000),
    SPH_C32(0xdecf0000), SPH_C32(0x6df80180), SPH_C32(0x77240000),
    SPH_C32(0xec47079e), SPH_C32(0xf4a0694e), SPH_C32(0xcda31812),
    SPH_C32(0x98aa496e) },
  { SPH_C32(0x1e4e0000), SPH_C32(0xdecf0000), SPH_C32(0x6df80180),
    SPH_C32(0x77240000), SPH_C32(0xec47079e), SPH_C32(0xf4a0694e),
    SPH_C32(0xcda31812), SPH_C32(0x98aa496e), SPH_C32(0xb2060000),
    SPH_C32(0xc5690000), SPH_C32(0x28031200), SPH_C32(0x74670000),
    SPH_C32(0xb6c236f4), SPH_C32(0xeb1239f8), SPH_C32(0x33d1dfec),
    SPH_C32(0x094e3198) },
  { SPH_C32(0xaec30000), SPH_C32(0x9c4f0001), SPH_C32(0x79d1e000),
    SPH_C32(0x2c150000), SPH_C32(0x45cc75b3), SPH_C32(0x6650b736),
    SPH_C32(0xab92f78f), SPH_C32(0xa312567b), SPH_C32(0xdb250000),
    SPH_C32(0x09290000), SPH_C32(0x49aac000), SPH_C32(0x81e10000),
    SPH_C32(0xcafe6b59), SPH_C32(0x42793431), SPH_C32(0x43566b76),
    SPH_C32(0xe86cba2e) },
  { SPH_C32(0xdb250000), SPH_C32(0x09290000), SPH_C32(0x49aac000),
    SPH_C32(0x81e10000), SPH_C32(0xcafe6b59), SPH_C32(0x42793431),
    SPH_C32(0x43566b76), SPH_C32(0xe86cba2e), SPH_C32(0x75e60000),
    SPH_C32(0x95660001), SPH_C32(0x307b2000), SPH_C32(0xadf40000),
    SPH_C32(0x8f321eea), SPH_C32(0x24298307), SPH_C32(0xe8c49cf9),
    SPH_C32(0x4b7eec55) },
  { SPH_C32(0x58430000), SPH_C32(0x807e0000), SPH_C32(0x78330001),
    SPH_C32(0xc66b3800), SPH_C32(0xe7375cdc), SPH_C32(0x79ad3fdd),
    SPH_C32(0xac73fe6f), SPH_C32(0x3a4479b1), SPH_C32(0x1d5a0000),
    SPH_C32(0x2b720000), SPH_C32(0x488d0000), SPH_C32(0xaf611800),
    SPH_C32(0x25cb2ec5), SPH_C32(0xc879bfd0), SPH_C32(0x81a20429),
    SPH_C32(0x1e7536a6) },
  { SPH_C32(0x1d5a0000), SPH_C32(0x2b720000), SPH_C32(0x488d0000),
    SPH_C32(0xaf611800), SPH_C32(0x25cb2ec5), SPH_C32(0xc879bfd0),
    SPH_C32(0x81a20429), SPH_C32(0x1e7536a6), SPH_C32(0x45190000),
    SPH_C32(0xab0c0000), SPH_C32(0x30be0001), SPH_C32(0x690a2000),
    SPH_C32(0xc2fc7219), SPH_C32(0xb1d4800d), SPH_C32(0x2dd1fa46),
    SPH_C32(0x24314f17) },
  { SPH_C32(0xa53b0000), SPH_C32(0x14260000), SPH_C32(0x4e30001e),
    SPH_C32(0x7cae0000), SPH_C32(0x8f9e0dd5), SPH_C32(0x78dfaa3d),
    SPH_C32(0xf73168d8), SPH_C32(0x0b1b4946), SPH_C32(0x07ed0000),
    SPH_C32(0xb2500000), SPH_C32(0x8774000a), SPH_C32(0x970d0000),
    SPH_C32(0x437223ae), SPH_C32(0x48c76ea4), SPH_C32(0xf4786222),
    SPH_C32(0x9075b1ce) },
  { SPH_C32(0x07ed0000), SPH_C32(0xb2500000), SPH_C32(0x8774000a),
    SPH_C32(0x970d0000), SPH_C32(0x437223ae), SPH_C32(0x48c76ea4),
    SPH_C32(0xf4786222), SPH_C32(0x9075b1ce), SPH_C32(0xa2d60000),
    SPH_C32(0xa6760000), SPH_C32(0xc9440014), SPH_C32(0xeba30000),
    SPH_C32(0xccec2e7b), SPH_C32(0x3018c499), SPH_C32(0x03490afa),
    SPH_C32(0x9b6ef888) },
  { SPH_C32(0x88980000), SPH_C32(0x1f940000), SPH_C32(0x7fcf002e),
    SPH_C32(0xfb4e0000), SPH_C32(0xf158079a), SPH_C32(0x61ae9167),
    SPH_C32(0xa895706c), SPH_C32(0xe6107494), SPH_C32(0x0bc20000),
    SPH_C32(0xdb630000), SPH_C32(0x7e88000c), SPH_C32(0x15860000),
    SPH_C32(0x91fd48f3), SPH_C32(0x7581bb43), SPH_C32(0xf460449e),
    SPH_C32(0xd8b61463) },
  { SPH_C32(0x0bc20000), SPH_C32(0xdb630000), SPH_C32(0x7e88000c),
    SPH_C32(0x15860000), SPH_C32(0x91fd48f3), SPH_C32(0x7581bb43),
    SPH_C32(0xf460449e), SPH_C32(0xd8b61463), SPH_C32(0x835a0000),
    SPH_C32(0xc4f70000), SPH_C32(0x01470022), SPH_C32(0xeec80000),
    SPH_C32(0x60a54f69), SPH_C32(0x142f2a24), SPH_C32(0x5cf534f2),
    SPH_C32(0x3ea660f7) },
  { SPH_C32(0x52500000), SPH_C32(0x29540000), SPH_C32(0x6a61004e),
    SPH_C32(0xf0ff0000), SPH_C32(0x9a317eec), SPH_C32(0x452341ce),
    SPH_C32(0xcf568fe5), SPH_C32(0x5303130f), SPH_C32(0x538d0000),
    SPH_C32(0xa9fc0000), SPH_C32(0x9ef70006), SPH_C32(0x56ff0000),
    SPH_C32(0x0ae4004e), SPH_C32(0x92c5cdf9), SPH_C32(0xa9444018),
    SPH_C32(0x7f975691) },
  { SPH_C32(0x538d0000), SPH_C32(0xa9fc0000), SPH_C32(0x9ef70006),
    SPH_C32(0x56ff0000), SPH_C32(0x0ae4004e), SPH_C32(0x92c5cdf9),
    SPH_C32(0xa9444018), SPH_C32(0x7f975691), SPH_C32(0x01dd0000),
    SPH_C32(0x80a80000), SPH_C32(0xf4960048), SPH_C32(0xa6000000),
    SPH_C32(0x90d57ea2), SPH_C32(0xd7e68c37), SPH_C32(0x6612cffd),
    SPH_C32(0x2c94459e) },
  { SPH_C32(0xe6280000), SPH_C32(0x4c4b0000), SPH_C32(0xa8550000),
    SPH_C32(0xd3d002e0), SPH_C32(0xd86130b8), SPH_C32(0x98a7b0da),
    SPH_C32(0x289506b4), SPH_C32(0xd75a4897), SPH_C32(0xf0c50000),
    SPH_C32(0x59230000), SPH_C32(0x45820000), SPH_C32(0xe18d00c0),
    SPH_C32(0x3b6d0631), SPH_C32(0xc2ed5699), SPH_C32(0xcbe0fe1c),
    SPH_C32(0x56a7b19f) },
  { SPH_C32(0xf0c50000), SPH_C32(0x59230000), SPH_C32(0x45820000),
    SPH_C32(0xe18d00c0), SPH_C32(0x3b6d0631), SPH_C32(0xc2ed5699),
    SPH_C32(0xcbe0fe1c), SPH_C32(0x56a7b19f), SPH_C32(0x16ed0000),
    SPH_C32(0x15680000), SPH_C32(0xedd70000), SPH_C32(0x325d0220),
    SPH_C32(0xe30c3689), SPH_C32(0x5a4ae643), SPH_C32(0xe375f8a8),
    SPH_C32(0x81fdf908) },
  { SPH_C32(0xb4310000), SPH_C32(0x77330000), SPH_C32(0xb15d0000),
    SPH_C32(0x7fd004e0), SPH_C32(0x78a26138), SPH_C32(0xd116c35d),
    SPH_C32(0xd256d489), SPH_C32(0x4e6f74de), SPH_C32(0xe3060000),
    SPH_C32(0xbdc10000), SPH_C32(0x87130000), SPH_C32(0xbff20060),
    SPH_C32(0x2eba0a1a), SPH_C32(0x8db53751), SPH_C32(0x73c5ab06),
    SPH_C32(0x5bd61539) },
  { SPH_C32(0xe3060000), SPH_C32(0xbdc10000), SPH_C32(0x87130000),
    SPH_C32(0xbff20060), SPH_C32(0x2eba0a1a), SPH_C32(0x8db53751),
    SPH_C32(0x73c5ab06), SPH_C32(0x5bd61539), SPH_C32(0x57370000),
    SPH_C32(0xcaf20000), SPH_C32(0x364e0000), SPH_C32(0xc0220480),
    SPH_C32(0x56186b22), SPH_C32(0x5ca3f40c), SPH_C32(0xa1937f8f),
    SPH_C32(0x15b961e7) },
  { SPH_C32(0x02f20000), SPH_C32(0xa2810000), SPH_C32(0x873f0000),
    SPH_C32(0xe36c7800), SPH_C32(0x1e1d74ef), SPH_C32(0x073d2bd6),
    SPH_C32(0xc4c23237), SPH_C32(0x7f32259e), SPH_C32(0xbadd0000),
    SPH_C32(0x13ad0000), SPH_C32(0xb7e70000), SPH_C32(0xf7282800),
    SPH_C32(0xdf45144d), SPH_C32(0x361ac33a), SPH_C32(0xea5a8d14),
    SPH_C32(0x2a2c18f0) },
  { SPH_C32(0xbadd0000), SPH_C32(0x13ad0000), SPH_C32(0xb7e70000),
    SPH_C32(0xf7282800), SPH_C32(0xdf45144d), SPH_C32(0x361ac33a),
    SPH_C32(0xea5a8d14), SPH_C32(0x2a2c18f0), SPH_C32(0xb82f0000),
    SPH_C32(0xb12c0000), SPH_C32(0x30d80000), SPH_C32(0x14445000),
    SPH_C32(0xc15860a2), SPH_C32(0x3127e8ec), SPH_C32(0x2e98bf23),
    SPH_C32(0x551e3d6e) },
  { SPH_C32(0x1e6c0000), SPH_C32(0xc4420000), SPH_C32(0x8a2e0000),
    SPH_C32(0xbcb6b800), SPH_C32(0x2c4413b6), SPH_C32(0x8bfdd3da),
    SPH_C32(0x6a0c1bc8), SPH_C32(0xb99dc2eb), SPH_C32(0x92560000),
    SPH_C32(0x1eda0000), SPH_C32(0xea510000), SPH_C32(0xe8b13000),
    SPH_C32(0xa93556a5), SPH_C32(0xebfb6199), SPH_C32(0xb15c2254),
    SPH_C32(0x33c5244f) },
  { SPH_C32(0x92560000), SPH_C32(0x1eda0000), SPH_C32(0xea510000),
    SPH_C32(0xe8b13000), SPH_C32(0xa93556a5), SPH_C32(0xebfb6199),
    SPH_C32(0xb15c2254), SPH_C32(0x33c5244f), SPH_C32(0x8c3a0000),
    SPH_C32(0xda980000), SPH_C32(0x607f0000), SPH_C32(0x54078800),
    SPH_C32(0x85714513), SPH_C32(0x6006b243), SPH_C32(0xdb50399c),
    SPH_C32(0x8a58e6a4) },
  { SPH_C32(0x033d0000), SPH_C32(0x08b30000), SPH_C32(0xf33a0000),
    SPH_C32(0x3ac20007), SPH_C32(0x51298a50), SPH_C32(0x6b6e661f),
    SPH_C32(0x0ea5cfe3), SPH_C32(0xe6da7ffe), SPH_C32(0xa8da0000),
    SPH_C32(0x96be0000), SPH_C32(0x5c1d0000), SPH_C32(0x07da0002),
    SPH_C32(0x7d669583), SPH_C32(0x1f98708a), SPH_C32(0xbb668808),
    SPH_C32(0xda878000) },
  { SPH_C32(0xa8da0000), SPH_C32(0x96be0000), SPH_C32(0x5c1d0000),
    SPH_C32(0x07da0002), SPH_C32(0x7d669583), SPH_C32(0x1f98708a),
    SPH_C32(0xbb668808), SPH_C32(0xda878000), SPH_C32(0xabe70000),
    SPH_C32(0x9e0d0000), SPH_C32(0xaf270000), SPH_C32(0x3d180005),
    SPH_C32(0x2c4f1fd3), SPH_C32(0x74f61695), SPH_C32(0xb5c347eb),
    SPH_C32(0x3c5dfffe) },
  { SPH_C32(0x01930000), SPH_C32(0xe7820000), SPH_C32(0xedfb0000),
    SPH_C32(0xcf0c000b), SPH_C32(0x8dd08d58), SPH_C32(0xbca3b42e),
    SPH_C32(0x063661e1), SPH_C32(0x536f9e7b), SPH_C32(0x92280000),
    SPH_C32(0xdc850000), SPH_C32(0x57fa0000), SPH_C32(0x56dc0003),
    SPH_C32(0xbae92316), SPH_C32(0x5aefa30c), SPH_C32(0x90cef752),
    SPH_C32(0x7b1675d7) },
  { SPH_C32(0x92280000), SPH_C32(0xdc850000), SPH_C32(0x57fa0000),
    SPH_C32(0x56dc0003), SPH_C32(0xbae92316), SPH_C32(0x5aefa30c),
    SPH_C32(0x90cef752), SPH_C32(0x7b1675d7), SPH_C32(0x93bb0000),
    SPH_C32(0x3b070000), SPH_C32(0xba010000), SPH_C32(0x99d00008),
    SPH_C32(0x3739ae4e), SPH_C32(0xe64c1722), SPH_C32(0x96f896b3),
    SPH_C32(0x2879ebac) },
  { SPH_C32(0x5fa80000), SPH_C32(0x56030000), SPH_C32(0x43ae0000),
    SPH_C32(0x64f30013), SPH_C32(0x257e86bf), SPH_C32(0x1311944e),
    SPH_C32(0x541e95bf), SPH_C32(0x8ea4db69), SPH_C32(0x00440000),
    SPH_C32(0x7f480000), SPH_C32(0xda7c0000), SPH_C32(0x2a230001),
    SPH_C32(0x3badc9cc), SPH_C32(0xa9b69c87), SPH_C32(0x030a9e60),
    SPH_C32(0xbe0a679e) },
  { SPH_C32(0x00440000), SPH_C32(0x7f480000), SPH_C32(0xda7c0000),
    SPH_C32(0x2a230001), SPH_C32(0x3badc9cc), SPH_C32(0xa9b69c87),
    SPH_C32(0x030a9e60), SPH_C32(0xbe0a679e), SPH_C32(0x5fec0000),
    SPH_C32(0x294b0000), SPH_C32(0x99d20000), SPH_C32(0x4ed00012),
    SPH_C32(0x1ed34f73), SPH_C32(0xbaa708c9), SPH_C32(0x57140bdf),
    SPH_C32(0x30aebcf7) },
  { SPH_C32(0xee930000), SPH_C32(0xd6070000), SPH_C32(0x92c10000),
    SPH_C32(0x2b9801e0), SPH_C32(0x9451287c), SPH_C32(0x3b6cfb57),
    SPH_C32(0x45312374), SPH_C32(0x201f6a64), SPH_C32(0x7b280000),
    SPH_C32(0x57420000), SPH_C32(0xa9e50000), SPH_C32(0x634300a0),
    SPH_C32(0x9edb442f), SPH_C32(0x6d9995bb), SPH_C32(0x27f83b03),
    SPH_C32(0xc7ff60f0) },
  { SPH_C32(0x7b280000), SPH_C32(0x57420000), SPH_C32(0xa9e50000),
    SPH_C32(0x634300a0), SPH_C32(0x9edb442f), SPH_C32(0x6d9995bb),
    SPH_C32(0x27f83b03), SPH_C32(0xc7ff60f0), SPH_C32(0x95bb0000),
    SPH_C32(0x81450000), SPH_C32(0x3b240000), SPH_C32(0x48db0140),
    SPH_C32(0x0a8a6c53), SPH_C32(0x56f56eec), SPH_C32(0x62c91877),
    SPH_C32(0xe7e00a94) }
};

#define INPUT_BIG   do { \
     const sph_u32 *tp = &T512[0][0]; \
    unsigned u, v; \
    m0 = 0; \
    m1 = 0; \
    m2 = 0; \
    m3 = 0; \
    m4 = 0; \
    m5 = 0; \
    m6 = 0; \
    m7 = 0; \
    m8 = 0; \
    m9 = 0; \
    mA = 0; \
    mB = 0; \
    mC = 0; \
    mD = 0; \
    mE = 0; \
    mF = 0; \
    for (u = 0; u < 8; u ++) { \
      unsigned db = buf(u); \
      for (v = 0; v < 8; v ++, db >>= 1) { \
        sph_u32 dm = SPH_T32(-(sph_u32)(db & 1)); \
        m0 ^= dm & *tp ++; \
        m1 ^= dm & *tp ++; \
        m2 ^= dm & *tp ++; \
        m3 ^= dm & *tp ++; \
        m4 ^= dm & *tp ++; \
        m5 ^= dm & *tp ++; \
        m6 ^= dm & *tp ++; \
        m7 ^= dm & *tp ++; \
        m8 ^= dm & *tp ++; \
        m9 ^= dm & *tp ++; \
        mA ^= dm & *tp ++; \
        mB ^= dm & *tp ++; \
        mC ^= dm & *tp ++; \
        mD ^= dm & *tp ++; \
        mE ^= dm & *tp ++; \
        mF ^= dm & *tp ++; \
      } \
    } \
  } while (0)
*/


/* Note: this table lists bits within each byte from least
   siginificant to most significant. */
__constant__ const sph_u32 T512[64][16] = {
	{ SPH_C32(0xef0b0270), SPH_C32(0x3afd0000), SPH_C32(0x5dae0000),
	  SPH_C32(0x69490000), SPH_C32(0x9b0f3c06), SPH_C32(0x4405b5f9),
	  SPH_C32(0x66140a51), SPH_C32(0x924f5d0a), SPH_C32(0xc96b0030),
	  SPH_C32(0xe7250000), SPH_C32(0x2f840000), SPH_C32(0x264f0000),
	  SPH_C32(0x08695bf9), SPH_C32(0x6dfcf137), SPH_C32(0x509f6984),
	  SPH_C32(0x9e69af68) },
	{ SPH_C32(0xc96b0030), SPH_C32(0xe7250000), SPH_C32(0x2f840000),
	  SPH_C32(0x264f0000), SPH_C32(0x08695bf9), SPH_C32(0x6dfcf137),
	  SPH_C32(0x509f6984), SPH_C32(0x9e69af68), SPH_C32(0x26600240),
	  SPH_C32(0xddd80000), SPH_C32(0x722a0000), SPH_C32(0x4f060000),
	  SPH_C32(0x936667ff), SPH_C32(0x29f944ce), SPH_C32(0x368b63d5),
	  SPH_C32(0x0c26f262) },
	{ SPH_C32(0x145a3c00), SPH_C32(0xb9e90000), SPH_C32(0x61270000),
	  SPH_C32(0xf1610000), SPH_C32(0xce613d6c), SPH_C32(0xb0493d78),
	  SPH_C32(0x47a96720), SPH_C32(0xe18e24c5), SPH_C32(0x23671400),
	  SPH_C32(0xc8b90000), SPH_C32(0xf4c70000), SPH_C32(0xfb750000),
	  SPH_C32(0x73cd2465), SPH_C32(0xf8a6a549), SPH_C32(0x02c40a3f),
	  SPH_C32(0xdc24e61f) },
	{ SPH_C32(0x23671400), SPH_C32(0xc8b90000), SPH_C32(0xf4c70000),
	  SPH_C32(0xfb750000), SPH_C32(0x73cd2465), SPH_C32(0xf8a6a549),
	  SPH_C32(0x02c40a3f), SPH_C32(0xdc24e61f), SPH_C32(0x373d2800),
	  SPH_C32(0x71500000), SPH_C32(0x95e00000), SPH_C32(0x0a140000),
	  SPH_C32(0xbdac1909), SPH_C32(0x48ef9831), SPH_C32(0x456d6d1f),
	  SPH_C32(0x3daac2da) },
	{ SPH_C32(0x54285c00), SPH_C32(0xeaed0000), SPH_C32(0xc5d60000),
	  SPH_C32(0xa1c50000), SPH_C32(0xb3a26770), SPH_C32(0x94a5c4e1),
	  SPH_C32(0x6bb0419d), SPH_C32(0x551b3782), SPH_C32(0x9cbb1800),
	  SPH_C32(0xb0d30000), SPH_C32(0x92510000), SPH_C32(0xed930000),
	  SPH_C32(0x593a4345), SPH_C32(0xe114d5f4), SPH_C32(0x430633da),
	  SPH_C32(0x78cace29) },
	{ SPH_C32(0x9cbb1800), SPH_C32(0xb0d30000), SPH_C32(0x92510000),
	  SPH_C32(0xed930000), SPH_C32(0x593a4345), SPH_C32(0xe114d5f4),
	  SPH_C32(0x430633da), SPH_C32(0x78cace29), SPH_C32(0xc8934400),
	  SPH_C32(0x5a3e0000), SPH_C32(0x57870000), SPH_C32(0x4c560000),
	  SPH_C32(0xea982435), SPH_C32(0x75b11115), SPH_C32(0x28b67247),
	  SPH_C32(0x2dd1f9ab) },
	{ SPH_C32(0x29449c00), SPH_C32(0x64e70000), SPH_C32(0xf24b0000),
	  SPH_C32(0xc2f30000), SPH_C32(0x0ede4e8f), SPH_C32(0x56c23745),
	  SPH_C32(0xf3e04259), SPH_C32(0x8d0d9ec4), SPH_C32(0x466d0c00),
	  SPH_C32(0x08620000), SPH_C32(0xdd5d0000), SPH_C32(0xbadd0000),
	  SPH_C32(0x6a927942), SPH_C32(0x441f2b93), SPH_C32(0x218ace6f),
	  SPH_C32(0xbf2c0be2) },
	{ SPH_C32(0x466d0c00), SPH_C32(0x08620000), SPH_C32(0xdd5d0000),
	  SPH_C32(0xbadd0000), SPH_C32(0x6a927942), SPH_C32(0x441f2b93),
	  SPH_C32(0x218ace6f), SPH_C32(0xbf2c0be2), SPH_C32(0x6f299000),
	  SPH_C32(0x6c850000), SPH_C32(0x2f160000), SPH_C32(0x782e0000),
	  SPH_C32(0x644c37cd), SPH_C32(0x12dd1cd6), SPH_C32(0xd26a8c36),
	  SPH_C32(0x32219526) },
	{ SPH_C32(0xf6800005), SPH_C32(0x3443c000), SPH_C32(0x24070000),
	  SPH_C32(0x8f3d0000), SPH_C32(0x21373bfb), SPH_C32(0x0ab8d5ae),
	  SPH_C32(0xcdc58b19), SPH_C32(0xd795ba31), SPH_C32(0xa67f0001),
	  SPH_C32(0x71378000), SPH_C32(0x19fc0000), SPH_C32(0x96db0000),
	  SPH_C32(0x3a8b6dfd), SPH_C32(0xebcaaef3), SPH_C32(0x2c6d478f),
	  SPH_C32(0xac8e6c88) },
	{ SPH_C32(0xa67f0001), SPH_C32(0x71378000), SPH_C32(0x19fc0000),
	  SPH_C32(0x96db0000), SPH_C32(0x3a8b6dfd), SPH_C32(0xebcaaef3),
	  SPH_C32(0x2c6d478f), SPH_C32(0xac8e6c88), SPH_C32(0x50ff0004),
	  SPH_C32(0x45744000), SPH_C32(0x3dfb0000), SPH_C32(0x19e60000),
	  SPH_C32(0x1bbc5606), SPH_C32(0xe1727b5d), SPH_C32(0xe1a8cc96),
	  SPH_C32(0x7b1bd6b9) },
	{ SPH_C32(0xf7750009), SPH_C32(0xcf3cc000), SPH_C32(0xc3d60000),
	  SPH_C32(0x04920000), SPH_C32(0x029519a9), SPH_C32(0xf8e836ba),
	  SPH_C32(0x7a87f14e), SPH_C32(0x9e16981a), SPH_C32(0xd46a0000),
	  SPH_C32(0x8dc8c000), SPH_C32(0xa5af0000), SPH_C32(0x4a290000),
	  SPH_C32(0xfc4e427a), SPH_C32(0xc9b4866c), SPH_C32(0x98369604),
	  SPH_C32(0xf746c320) },
	{ SPH_C32(0xd46a0000), SPH_C32(0x8dc8c000), SPH_C32(0xa5af0000),
	  SPH_C32(0x4a290000), SPH_C32(0xfc4e427a), SPH_C32(0xc9b4866c),
	  SPH_C32(0x98369604), SPH_C32(0xf746c320), SPH_C32(0x231f0009),
	  SPH_C32(0x42f40000), SPH_C32(0x66790000), SPH_C32(0x4ebb0000),
	  SPH_C32(0xfedb5bd3), SPH_C32(0x315cb0d6), SPH_C32(0xe2b1674a),
	  SPH_C32(0x69505b3a) },
	{ SPH_C32(0x774400f0), SPH_C32(0xf15a0000), SPH_C32(0xf5b20000),
	  SPH_C32(0x34140000), SPH_C32(0x89377e8c), SPH_C32(0x5a8bec25),
	  SPH_C32(0x0bc3cd1e), SPH_C32(0xcf3775cb), SPH_C32(0xf46c0050),
	  SPH_C32(0x96180000), SPH_C32(0x14a50000), SPH_C32(0x031f0000),
	  SPH_C32(0x42947eb8), SPH_C32(0x66bf7e19), SPH_C32(0x9ca470d2),
	  SPH_C32(0x8a341574) },
	{ SPH_C32(0xf46c0050), SPH_C32(0x96180000), SPH_C32(0x14a50000),
	  SPH_C32(0x031f0000), SPH_C32(0x42947eb8), SPH_C32(0x66bf7e19),
	  SPH_C32(0x9ca470d2), SPH_C32(0x8a341574), SPH_C32(0x832800a0),
	  SPH_C32(0x67420000), SPH_C32(0xe1170000), SPH_C32(0x370b0000),
	  SPH_C32(0xcba30034), SPH_C32(0x3c34923c), SPH_C32(0x9767bdcc),
	  SPH_C32(0x450360bf) },
	{ SPH_C32(0xe8870170), SPH_C32(0x9d720000), SPH_C32(0x12db0000),
	  SPH_C32(0xd4220000), SPH_C32(0xf2886b27), SPH_C32(0xa921e543),
	  SPH_C32(0x4ef8b518), SPH_C32(0x618813b1), SPH_C32(0xb4370060),
	  SPH_C32(0x0c4c0000), SPH_C32(0x56c20000), SPH_C32(0x5cae0000),
	  SPH_C32(0x94541f3f), SPH_C32(0x3b3ef825), SPH_C32(0x1b365f3d),
	  SPH_C32(0xf3d45758) },
	{ SPH_C32(0xb4370060), SPH_C32(0x0c4c0000), SPH_C32(0x56c20000),
	  SPH_C32(0x5cae0000), SPH_C32(0x94541f3f), SPH_C32(0x3b3ef825),
	  SPH_C32(0x1b365f3d), SPH_C32(0xf3d45758), SPH_C32(0x5cb00110),
	  SPH_C32(0x913e0000), SPH_C32(0x44190000), SPH_C32(0x888c0000),
	  SPH_C32(0x66dc7418), SPH_C32(0x921f1d66), SPH_C32(0x55ceea25),
	  SPH_C32(0x925c44e9) },
	{ SPH_C32(0x0c720000), SPH_C32(0x49e50f00), SPH_C32(0x42790000),
	  SPH_C32(0x5cea0000), SPH_C32(0x33aa301a), SPH_C32(0x15822514),
	  SPH_C32(0x95a34b7b), SPH_C32(0xb44b0090), SPH_C32(0xfe220000),
	  SPH_C32(0xa7580500), SPH_C32(0x25d10000), SPH_C32(0xf7600000),
	  SPH_C32(0x893178da), SPH_C32(0x1fd4f860), SPH_C32(0x4ed0a315),
	  SPH_C32(0xa123ff9f) },
	{ SPH_C32(0xfe220000), SPH_C32(0xa7580500), SPH_C32(0x25d10000),
	  SPH_C32(0xf7600000), SPH_C32(0x893178da), SPH_C32(0x1fd4f860),
	  SPH_C32(0x4ed0a315), SPH_C32(0xa123ff9f), SPH_C32(0xf2500000),
	  SPH_C32(0xeebd0a00), SPH_C32(0x67a80000), SPH_C32(0xab8a0000),
	  SPH_C32(0xba9b48c0), SPH_C32(0x0a56dd74), SPH_C32(0xdb73e86e),
	  SPH_C32(0x1568ff0f) },
	{ SPH_C32(0x45180000), SPH_C32(0xa5b51700), SPH_C32(0xf96a0000),
	  SPH_C32(0x3b480000), SPH_C32(0x1ecc142c), SPH_C32(0x231395d6),
	  SPH_C32(0x16bca6b0), SPH_C32(0xdf33f4df), SPH_C32(0xb83d0000),
	  SPH_C32(0x16710600), SPH_C32(0x379a0000), SPH_C32(0xf5b10000),
	  SPH_C32(0x228161ac), SPH_C32(0xae48f145), SPH_C32(0x66241616),
	  SPH_C32(0xc5c1eb3e) },
	{ SPH_C32(0xb83d0000), SPH_C32(0x16710600), SPH_C32(0x379a0000),
	  SPH_C32(0xf5b10000), SPH_C32(0x228161ac), SPH_C32(0xae48f145),
	  SPH_C32(0x66241616), SPH_C32(0xc5c1eb3e), SPH_C32(0xfd250000),
	  SPH_C32(0xb3c41100), SPH_C32(0xcef00000), SPH_C32(0xcef90000),
	  SPH_C32(0x3c4d7580), SPH_C32(0x8d5b6493), SPH_C32(0x7098b0a6),
	  SPH_C32(0x1af21fe1) },
	{ SPH_C32(0x75a40000), SPH_C32(0xc28b2700), SPH_C32(0x94a40000),
	  SPH_C32(0x90f50000), SPH_C32(0xfb7857e0), SPH_C32(0x49ce0bae),
	  SPH_C32(0x1767c483), SPH_C32(0xaedf667e), SPH_C32(0xd1660000),
	  SPH_C32(0x1bbc0300), SPH_C32(0x9eec0000), SPH_C32(0xf6940000),
	  SPH_C32(0x03024527), SPH_C32(0xcf70fcf2), SPH_C32(0xb4431b17),
	  SPH_C32(0x857f3c2b) },
	{ SPH_C32(0xd1660000), SPH_C32(0x1bbc0300), SPH_C32(0x9eec0000),
	  SPH_C32(0xf6940000), SPH_C32(0x03024527), SPH_C32(0xcf70fcf2),
	  SPH_C32(0xb4431b17), SPH_C32(0x857f3c2b), SPH_C32(0xa4c20000),
	  SPH_C32(0xd9372400), SPH_C32(0x0a480000), SPH_C32(0x66610000),
	  SPH_C32(0xf87a12c7), SPH_C32(0x86bef75c), SPH_C32(0xa324df94),
	  SPH_C32(0x2ba05a55) },
	{ SPH_C32(0x75c90003), SPH_C32(0x0e10c000), SPH_C32(0xd1200000),
	  SPH_C32(0xbaea0000), SPH_C32(0x8bc42f3e), SPH_C32(0x8758b757),
	  SPH_C32(0xbb28761d), SPH_C32(0x00b72e2b), SPH_C32(0xeecf0001),
	  SPH_C32(0x6f564000), SPH_C32(0xf33e0000), SPH_C32(0xa79e0000),
	  SPH_C32(0xbdb57219), SPH_C32(0xb711ebc5), SPH_C32(0x4a3b40ba),
	  SPH_C32(0xfeabf254) },
	{ SPH_C32(0xeecf0001), SPH_C32(0x6f564000), SPH_C32(0xf33e0000),
	  SPH_C32(0xa79e0000), SPH_C32(0xbdb57219), SPH_C32(0xb711ebc5),
	  SPH_C32(0x4a3b40ba), SPH_C32(0xfeabf254), SPH_C32(0x9b060002),
	  SPH_C32(0x61468000), SPH_C32(0x221e0000), SPH_C32(0x1d740000),
	  SPH_C32(0x36715d27), SPH_C32(0x30495c92), SPH_C32(0xf11336a7),
	  SPH_C32(0xfe1cdc7f) },
	{ SPH_C32(0x86790000), SPH_C32(0x3f390002), SPH_C32(0xe19ae000),
	  SPH_C32(0x98560000), SPH_C32(0x9565670e), SPH_C32(0x4e88c8ea),
	  SPH_C32(0xd3dd4944), SPH_C32(0x161ddab9), SPH_C32(0x30b70000),
	  SPH_C32(0xe5d00000), SPH_C32(0xf4f46000), SPH_C32(0x42c40000),
	  SPH_C32(0x63b83d6a), SPH_C32(0x78ba9460), SPH_C32(0x21afa1ea),
	  SPH_C32(0xb0a51834) },
	{ SPH_C32(0x30b70000), SPH_C32(0xe5d00000), SPH_C32(0xf4f46000),
	  SPH_C32(0x42c40000), SPH_C32(0x63b83d6a), SPH_C32(0x78ba9460),
	  SPH_C32(0x21afa1ea), SPH_C32(0xb0a51834), SPH_C32(0xb6ce0000),
	  SPH_C32(0xdae90002), SPH_C32(0x156e8000), SPH_C32(0xda920000),
	  SPH_C32(0xf6dd5a64), SPH_C32(0x36325c8a), SPH_C32(0xf272e8ae),
	  SPH_C32(0xa6b8c28d) },
	{ SPH_C32(0x14190000), SPH_C32(0x23ca003c), SPH_C32(0x50df0000),
	  SPH_C32(0x44b60000), SPH_C32(0x1b6c67b0), SPH_C32(0x3cf3ac75),
	  SPH_C32(0x61e610b0), SPH_C32(0xdbcadb80), SPH_C32(0xe3430000),
	  SPH_C32(0x3a4e0014), SPH_C32(0xf2c60000), SPH_C32(0xaa4e0000),
	  SPH_C32(0xdb1e42a6), SPH_C32(0x256bbe15), SPH_C32(0x123db156),
	  SPH_C32(0x3a4e99d7) },
	{ SPH_C32(0xe3430000), SPH_C32(0x3a4e0014), SPH_C32(0xf2c60000),
	  SPH_C32(0xaa4e0000), SPH_C32(0xdb1e42a6), SPH_C32(0x256bbe15),
	  SPH_C32(0x123db156), SPH_C32(0x3a4e99d7), SPH_C32(0xf75a0000),
	  SPH_C32(0x19840028), SPH_C32(0xa2190000), SPH_C32(0xeef80000),
	  SPH_C32(0xc0722516), SPH_C32(0x19981260), SPH_C32(0x73dba1e6),
	  SPH_C32(0xe1844257) },
	{ SPH_C32(0x54500000), SPH_C32(0x0671005c), SPH_C32(0x25ae0000),
	  SPH_C32(0x6a1e0000), SPH_C32(0x2ea54edf), SPH_C32(0x664e8512),
	  SPH_C32(0xbfba18c3), SPH_C32(0x7e715d17), SPH_C32(0xbc8d0000),
	  SPH_C32(0xfc3b0018), SPH_C32(0x19830000), SPH_C32(0xd10b0000),
	  SPH_C32(0xae1878c4), SPH_C32(0x42a69856), SPH_C32(0x0012da37),
	  SPH_C32(0x2c3b504e) },
	{ SPH_C32(0xbc8d0000), SPH_C32(0xfc3b0018), SPH_C32(0x19830000),
	  SPH_C32(0xd10b0000), SPH_C32(0xae1878c4), SPH_C32(0x42a69856),
	  SPH_C32(0x0012da37), SPH_C32(0x2c3b504e), SPH_C32(0xe8dd0000),
	  SPH_C32(0xfa4a0044), SPH_C32(0x3c2d0000), SPH_C32(0xbb150000),
	  SPH_C32(0x80bd361b), SPH_C32(0x24e81d44), SPH_C32(0xbfa8c2f4),
	  SPH_C32(0x524a0d59) },
	{ SPH_C32(0x69510000), SPH_C32(0xd4e1009c), SPH_C32(0xc3230000),
	  SPH_C32(0xac2f0000), SPH_C32(0xe4950bae), SPH_C32(0xcea415dc),
	  SPH_C32(0x87ec287c), SPH_C32(0xbce1a3ce), SPH_C32(0xc6730000),
	  SPH_C32(0xaf8d000c), SPH_C32(0xa4c10000), SPH_C32(0x218d0000),
	  SPH_C32(0x23111587), SPH_C32(0x7913512f), SPH_C32(0x1d28ac88),
	  SPH_C32(0x378dd173) },
	{ SPH_C32(0xc6730000), SPH_C32(0xaf8d000c), SPH_C32(0xa4c10000),
	  SPH_C32(0x218d0000), SPH_C32(0x23111587), SPH_C32(0x7913512f),
	  SPH_C32(0x1d28ac88), SPH_C32(0x378dd173), SPH_C32(0xaf220000),
	  SPH_C32(0x7b6c0090), SPH_C32(0x67e20000), SPH_C32(0x8da20000),
	  SPH_C32(0xc7841e29), SPH_C32(0xb7b744f3), SPH_C32(0x9ac484f4),
	  SPH_C32(0x8b6c72bd) },
	{ SPH_C32(0xcc140000), SPH_C32(0xa5630000), SPH_C32(0x5ab90780),
	  SPH_C32(0x3b500000), SPH_C32(0x4bd013ff), SPH_C32(0x879b3418),
	  SPH_C32(0x694348c1), SPH_C32(0xca5a87fe), SPH_C32(0x819e0000),
	  SPH_C32(0xec570000), SPH_C32(0x66320280), SPH_C32(0x95f30000),
	  SPH_C32(0x5da92802), SPH_C32(0x48f43cbc), SPH_C32(0xe65aa22d),
	  SPH_C32(0x8e67b7fa) },
	{ SPH_C32(0x819e0000), SPH_C32(0xec570000), SPH_C32(0x66320280),
	  SPH_C32(0x95f30000), SPH_C32(0x5da92802), SPH_C32(0x48f43cbc),
	  SPH_C32(0xe65aa22d), SPH_C32(0x8e67b7fa), SPH_C32(0x4d8a0000),
	  SPH_C32(0x49340000), SPH_C32(0x3c8b0500), SPH_C32(0xaea30000),
	  SPH_C32(0x16793bfd), SPH_C32(0xcf6f08a4), SPH_C32(0x8f19eaec),
	  SPH_C32(0x443d3004) },
	{ SPH_C32(0x78230000), SPH_C32(0x12fc0000), SPH_C32(0xa93a0b80),
	  SPH_C32(0x90a50000), SPH_C32(0x713e2879), SPH_C32(0x7ee98924),
	  SPH_C32(0xf08ca062), SPH_C32(0x636f8bab), SPH_C32(0x02af0000),
	  SPH_C32(0xb7280000), SPH_C32(0xba1c0300), SPH_C32(0x56980000),
	  SPH_C32(0xba8d45d3), SPH_C32(0x8048c667), SPH_C32(0xa95c149a),
	  SPH_C32(0xf4f6ea7b) },
	{ SPH_C32(0x02af0000), SPH_C32(0xb7280000), SPH_C32(0xba1c0300),
	  SPH_C32(0x56980000), SPH_C32(0xba8d45d3), SPH_C32(0x8048c667),
	  SPH_C32(0xa95c149a), SPH_C32(0xf4f6ea7b), SPH_C32(0x7a8c0000),
	  SPH_C32(0xa5d40000), SPH_C32(0x13260880), SPH_C32(0xc63d0000),
	  SPH_C32(0xcbb36daa), SPH_C32(0xfea14f43), SPH_C32(0x59d0b4f8),
	  SPH_C32(0x979961d0) },
	{ SPH_C32(0xac480000), SPH_C32(0x1ba60000), SPH_C32(0x45fb1380),
	  SPH_C32(0x03430000), SPH_C32(0x5a85316a), SPH_C32(0x1fb250b6),
	  SPH_C32(0xfe72c7fe), SPH_C32(0x91e478f6), SPH_C32(0x1e4e0000),
	  SPH_C32(0xdecf0000), SPH_C32(0x6df80180), SPH_C32(0x77240000),
	  SPH_C32(0xec47079e), SPH_C32(0xf4a0694e), SPH_C32(0xcda31812),
	  SPH_C32(0x98aa496e) },
	{ SPH_C32(0x1e4e0000), SPH_C32(0xdecf0000), SPH_C32(0x6df80180),
	  SPH_C32(0x77240000), SPH_C32(0xec47079e), SPH_C32(0xf4a0694e),
	  SPH_C32(0xcda31812), SPH_C32(0x98aa496e), SPH_C32(0xb2060000),
	  SPH_C32(0xc5690000), SPH_C32(0x28031200), SPH_C32(0x74670000),
	  SPH_C32(0xb6c236f4), SPH_C32(0xeb1239f8), SPH_C32(0x33d1dfec),
	  SPH_C32(0x094e3198) },
	{ SPH_C32(0xaec30000), SPH_C32(0x9c4f0001), SPH_C32(0x79d1e000),
	  SPH_C32(0x2c150000), SPH_C32(0x45cc75b3), SPH_C32(0x6650b736),
	  SPH_C32(0xab92f78f), SPH_C32(0xa312567b), SPH_C32(0xdb250000),
	  SPH_C32(0x09290000), SPH_C32(0x49aac000), SPH_C32(0x81e10000),
	  SPH_C32(0xcafe6b59), SPH_C32(0x42793431), SPH_C32(0x43566b76),
	  SPH_C32(0xe86cba2e) },
	{ SPH_C32(0xdb250000), SPH_C32(0x09290000), SPH_C32(0x49aac000),
	  SPH_C32(0x81e10000), SPH_C32(0xcafe6b59), SPH_C32(0x42793431),
	  SPH_C32(0x43566b76), SPH_C32(0xe86cba2e), SPH_C32(0x75e60000),
	  SPH_C32(0x95660001), SPH_C32(0x307b2000), SPH_C32(0xadf40000),
	  SPH_C32(0x8f321eea), SPH_C32(0x24298307), SPH_C32(0xe8c49cf9),
	  SPH_C32(0x4b7eec55) },
	{ SPH_C32(0x58430000), SPH_C32(0x807e0000), SPH_C32(0x78330001),
	  SPH_C32(0xc66b3800), SPH_C32(0xe7375cdc), SPH_C32(0x79ad3fdd),
	  SPH_C32(0xac73fe6f), SPH_C32(0x3a4479b1), SPH_C32(0x1d5a0000),
	  SPH_C32(0x2b720000), SPH_C32(0x488d0000), SPH_C32(0xaf611800),
	  SPH_C32(0x25cb2ec5), SPH_C32(0xc879bfd0), SPH_C32(0x81a20429),
	  SPH_C32(0x1e7536a6) },
	{ SPH_C32(0x1d5a0000), SPH_C32(0x2b720000), SPH_C32(0x488d0000),
	  SPH_C32(0xaf611800), SPH_C32(0x25cb2ec5), SPH_C32(0xc879bfd0),
	  SPH_C32(0x81a20429), SPH_C32(0x1e7536a6), SPH_C32(0x45190000),
	  SPH_C32(0xab0c0000), SPH_C32(0x30be0001), SPH_C32(0x690a2000),
	  SPH_C32(0xc2fc7219), SPH_C32(0xb1d4800d), SPH_C32(0x2dd1fa46),
	  SPH_C32(0x24314f17) },
	{ SPH_C32(0xa53b0000), SPH_C32(0x14260000), SPH_C32(0x4e30001e),
	  SPH_C32(0x7cae0000), SPH_C32(0x8f9e0dd5), SPH_C32(0x78dfaa3d),
	  SPH_C32(0xf73168d8), SPH_C32(0x0b1b4946), SPH_C32(0x07ed0000),
	  SPH_C32(0xb2500000), SPH_C32(0x8774000a), SPH_C32(0x970d0000),
	  SPH_C32(0x437223ae), SPH_C32(0x48c76ea4), SPH_C32(0xf4786222),
	  SPH_C32(0x9075b1ce) },
	{ SPH_C32(0x07ed0000), SPH_C32(0xb2500000), SPH_C32(0x8774000a),
	  SPH_C32(0x970d0000), SPH_C32(0x437223ae), SPH_C32(0x48c76ea4),
	  SPH_C32(0xf4786222), SPH_C32(0x9075b1ce), SPH_C32(0xa2d60000),
	  SPH_C32(0xa6760000), SPH_C32(0xc9440014), SPH_C32(0xeba30000),
	  SPH_C32(0xccec2e7b), SPH_C32(0x3018c499), SPH_C32(0x03490afa),
	  SPH_C32(0x9b6ef888) },
	{ SPH_C32(0x88980000), SPH_C32(0x1f940000), SPH_C32(0x7fcf002e),
	  SPH_C32(0xfb4e0000), SPH_C32(0xf158079a), SPH_C32(0x61ae9167),
	  SPH_C32(0xa895706c), SPH_C32(0xe6107494), SPH_C32(0x0bc20000),
	  SPH_C32(0xdb630000), SPH_C32(0x7e88000c), SPH_C32(0x15860000),
	  SPH_C32(0x91fd48f3), SPH_C32(0x7581bb43), SPH_C32(0xf460449e),
	  SPH_C32(0xd8b61463) },
	{ SPH_C32(0x0bc20000), SPH_C32(0xdb630000), SPH_C32(0x7e88000c),
	  SPH_C32(0x15860000), SPH_C32(0x91fd48f3), SPH_C32(0x7581bb43),
	  SPH_C32(0xf460449e), SPH_C32(0xd8b61463), SPH_C32(0x835a0000),
	  SPH_C32(0xc4f70000), SPH_C32(0x01470022), SPH_C32(0xeec80000),
	  SPH_C32(0x60a54f69), SPH_C32(0x142f2a24), SPH_C32(0x5cf534f2),
	  SPH_C32(0x3ea660f7) },
	{ SPH_C32(0x52500000), SPH_C32(0x29540000), SPH_C32(0x6a61004e),
	  SPH_C32(0xf0ff0000), SPH_C32(0x9a317eec), SPH_C32(0x452341ce),
	  SPH_C32(0xcf568fe5), SPH_C32(0x5303130f), SPH_C32(0x538d0000),
	  SPH_C32(0xa9fc0000), SPH_C32(0x9ef70006), SPH_C32(0x56ff0000),
	  SPH_C32(0x0ae4004e), SPH_C32(0x92c5cdf9), SPH_C32(0xa9444018),
	  SPH_C32(0x7f975691) },
	{ SPH_C32(0x538d0000), SPH_C32(0xa9fc0000), SPH_C32(0x9ef70006),
	  SPH_C32(0x56ff0000), SPH_C32(0x0ae4004e), SPH_C32(0x92c5cdf9),
	  SPH_C32(0xa9444018), SPH_C32(0x7f975691), SPH_C32(0x01dd0000),
	  SPH_C32(0x80a80000), SPH_C32(0xf4960048), SPH_C32(0xa6000000),
	  SPH_C32(0x90d57ea2), SPH_C32(0xd7e68c37), SPH_C32(0x6612cffd),
	  SPH_C32(0x2c94459e) },
	{ SPH_C32(0xe6280000), SPH_C32(0x4c4b0000), SPH_C32(0xa8550000),
	  SPH_C32(0xd3d002e0), SPH_C32(0xd86130b8), SPH_C32(0x98a7b0da),
	  SPH_C32(0x289506b4), SPH_C32(0xd75a4897), SPH_C32(0xf0c50000),
	  SPH_C32(0x59230000), SPH_C32(0x45820000), SPH_C32(0xe18d00c0),
	  SPH_C32(0x3b6d0631), SPH_C32(0xc2ed5699), SPH_C32(0xcbe0fe1c),
	  SPH_C32(0x56a7b19f) },
	{ SPH_C32(0xf0c50000), SPH_C32(0x59230000), SPH_C32(0x45820000),
	  SPH_C32(0xe18d00c0), SPH_C32(0x3b6d0631), SPH_C32(0xc2ed5699),
	  SPH_C32(0xcbe0fe1c), SPH_C32(0x56a7b19f), SPH_C32(0x16ed0000),
	  SPH_C32(0x15680000), SPH_C32(0xedd70000), SPH_C32(0x325d0220),
	  SPH_C32(0xe30c3689), SPH_C32(0x5a4ae643), SPH_C32(0xe375f8a8),
	  SPH_C32(0x81fdf908) },
	{ SPH_C32(0xb4310000), SPH_C32(0x77330000), SPH_C32(0xb15d0000),
	  SPH_C32(0x7fd004e0), SPH_C32(0x78a26138), SPH_C32(0xd116c35d),
	  SPH_C32(0xd256d489), SPH_C32(0x4e6f74de), SPH_C32(0xe3060000),
	  SPH_C32(0xbdc10000), SPH_C32(0x87130000), SPH_C32(0xbff20060),
	  SPH_C32(0x2eba0a1a), SPH_C32(0x8db53751), SPH_C32(0x73c5ab06),
	  SPH_C32(0x5bd61539) },
	{ SPH_C32(0xe3060000), SPH_C32(0xbdc10000), SPH_C32(0x87130000),
	  SPH_C32(0xbff20060), SPH_C32(0x2eba0a1a), SPH_C32(0x8db53751),
	  SPH_C32(0x73c5ab06), SPH_C32(0x5bd61539), SPH_C32(0x57370000),
	  SPH_C32(0xcaf20000), SPH_C32(0x364e0000), SPH_C32(0xc0220480),
	  SPH_C32(0x56186b22), SPH_C32(0x5ca3f40c), SPH_C32(0xa1937f8f),
	  SPH_C32(0x15b961e7) },
	{ SPH_C32(0x02f20000), SPH_C32(0xa2810000), SPH_C32(0x873f0000),
	  SPH_C32(0xe36c7800), SPH_C32(0x1e1d74ef), SPH_C32(0x073d2bd6),
	  SPH_C32(0xc4c23237), SPH_C32(0x7f32259e), SPH_C32(0xbadd0000),
	  SPH_C32(0x13ad0000), SPH_C32(0xb7e70000), SPH_C32(0xf7282800),
	  SPH_C32(0xdf45144d), SPH_C32(0x361ac33a), SPH_C32(0xea5a8d14),
	  SPH_C32(0x2a2c18f0) },
	{ SPH_C32(0xbadd0000), SPH_C32(0x13ad0000), SPH_C32(0xb7e70000),
	  SPH_C32(0xf7282800), SPH_C32(0xdf45144d), SPH_C32(0x361ac33a),
	  SPH_C32(0xea5a8d14), SPH_C32(0x2a2c18f0), SPH_C32(0xb82f0000),
	  SPH_C32(0xb12c0000), SPH_C32(0x30d80000), SPH_C32(0x14445000),
	  SPH_C32(0xc15860a2), SPH_C32(0x3127e8ec), SPH_C32(0x2e98bf23),
	  SPH_C32(0x551e3d6e) },
	{ SPH_C32(0x1e6c0000), SPH_C32(0xc4420000), SPH_C32(0x8a2e0000),
	  SPH_C32(0xbcb6b800), SPH_C32(0x2c4413b6), SPH_C32(0x8bfdd3da),
	  SPH_C32(0x6a0c1bc8), SPH_C32(0xb99dc2eb), SPH_C32(0x92560000),
	  SPH_C32(0x1eda0000), SPH_C32(0xea510000), SPH_C32(0xe8b13000),
	  SPH_C32(0xa93556a5), SPH_C32(0xebfb6199), SPH_C32(0xb15c2254),
	  SPH_C32(0x33c5244f) },
	{ SPH_C32(0x92560000), SPH_C32(0x1eda0000), SPH_C32(0xea510000),
	  SPH_C32(0xe8b13000), SPH_C32(0xa93556a5), SPH_C32(0xebfb6199),
	  SPH_C32(0xb15c2254), SPH_C32(0x33c5244f), SPH_C32(0x8c3a0000),
	  SPH_C32(0xda980000), SPH_C32(0x607f0000), SPH_C32(0x54078800),
	  SPH_C32(0x85714513), SPH_C32(0x6006b243), SPH_C32(0xdb50399c),
	  SPH_C32(0x8a58e6a4) },
	{ SPH_C32(0x033d0000), SPH_C32(0x08b30000), SPH_C32(0xf33a0000),
	  SPH_C32(0x3ac20007), SPH_C32(0x51298a50), SPH_C32(0x6b6e661f),
	  SPH_C32(0x0ea5cfe3), SPH_C32(0xe6da7ffe), SPH_C32(0xa8da0000),
	  SPH_C32(0x96be0000), SPH_C32(0x5c1d0000), SPH_C32(0x07da0002),
	  SPH_C32(0x7d669583), SPH_C32(0x1f98708a), SPH_C32(0xbb668808),
	  SPH_C32(0xda878000) },
	{ SPH_C32(0xa8da0000), SPH_C32(0x96be0000), SPH_C32(0x5c1d0000),
	  SPH_C32(0x07da0002), SPH_C32(0x7d669583), SPH_C32(0x1f98708a),
	  SPH_C32(0xbb668808), SPH_C32(0xda878000), SPH_C32(0xabe70000),
	  SPH_C32(0x9e0d0000), SPH_C32(0xaf270000), SPH_C32(0x3d180005),
	  SPH_C32(0x2c4f1fd3), SPH_C32(0x74f61695), SPH_C32(0xb5c347eb),
	  SPH_C32(0x3c5dfffe) },
	{ SPH_C32(0x01930000), SPH_C32(0xe7820000), SPH_C32(0xedfb0000),
	  SPH_C32(0xcf0c000b), SPH_C32(0x8dd08d58), SPH_C32(0xbca3b42e),
	  SPH_C32(0x063661e1), SPH_C32(0x536f9e7b), SPH_C32(0x92280000),
	  SPH_C32(0xdc850000), SPH_C32(0x57fa0000), SPH_C32(0x56dc0003),
	  SPH_C32(0xbae92316), SPH_C32(0x5aefa30c), SPH_C32(0x90cef752),
	  SPH_C32(0x7b1675d7) },
	{ SPH_C32(0x92280000), SPH_C32(0xdc850000), SPH_C32(0x57fa0000),
	  SPH_C32(0x56dc0003), SPH_C32(0xbae92316), SPH_C32(0x5aefa30c),
	  SPH_C32(0x90cef752), SPH_C32(0x7b1675d7), SPH_C32(0x93bb0000),
	  SPH_C32(0x3b070000), SPH_C32(0xba010000), SPH_C32(0x99d00008),
	  SPH_C32(0x3739ae4e), SPH_C32(0xe64c1722), SPH_C32(0x96f896b3),
	  SPH_C32(0x2879ebac) },
	{ SPH_C32(0x5fa80000), SPH_C32(0x56030000), SPH_C32(0x43ae0000),
	  SPH_C32(0x64f30013), SPH_C32(0x257e86bf), SPH_C32(0x1311944e),
	  SPH_C32(0x541e95bf), SPH_C32(0x8ea4db69), SPH_C32(0x00440000),
	  SPH_C32(0x7f480000), SPH_C32(0xda7c0000), SPH_C32(0x2a230001),
	  SPH_C32(0x3badc9cc), SPH_C32(0xa9b69c87), SPH_C32(0x030a9e60),
	  SPH_C32(0xbe0a679e) },
	{ SPH_C32(0x00440000), SPH_C32(0x7f480000), SPH_C32(0xda7c0000),
	  SPH_C32(0x2a230001), SPH_C32(0x3badc9cc), SPH_C32(0xa9b69c87),
	  SPH_C32(0x030a9e60), SPH_C32(0xbe0a679e), SPH_C32(0x5fec0000),
	  SPH_C32(0x294b0000), SPH_C32(0x99d20000), SPH_C32(0x4ed00012),
	  SPH_C32(0x1ed34f73), SPH_C32(0xbaa708c9), SPH_C32(0x57140bdf),
	  SPH_C32(0x30aebcf7) },
	{ SPH_C32(0xee930000), SPH_C32(0xd6070000), SPH_C32(0x92c10000),
	  SPH_C32(0x2b9801e0), SPH_C32(0x9451287c), SPH_C32(0x3b6cfb57),
	  SPH_C32(0x45312374), SPH_C32(0x201f6a64), SPH_C32(0x7b280000),
	  SPH_C32(0x57420000), SPH_C32(0xa9e50000), SPH_C32(0x634300a0),
	  SPH_C32(0x9edb442f), SPH_C32(0x6d9995bb), SPH_C32(0x27f83b03),
	  SPH_C32(0xc7ff60f0) },
	{ SPH_C32(0x7b280000), SPH_C32(0x57420000), SPH_C32(0xa9e50000),
	  SPH_C32(0x634300a0), SPH_C32(0x9edb442f), SPH_C32(0x6d9995bb),
	  SPH_C32(0x27f83b03), SPH_C32(0xc7ff60f0), SPH_C32(0x95bb0000),
	  SPH_C32(0x81450000), SPH_C32(0x3b240000), SPH_C32(0x48db0140),
	  SPH_C32(0x0a8a6c53), SPH_C32(0x56f56eec), SPH_C32(0x62c91877),
	  SPH_C32(0xe7e00a94) }
};

/*
#define INPUT_BIG   do { \
		 const sph_u32 *tp = &T512[0][0]; \
		unsigned u, v; \
		m0 = 0; \
		m1 = 0; \
		m2 = 0; \
		m3 = 0; \
		m4 = 0; \
		m5 = 0; \
		m6 = 0; \
		m7 = 0; \
		m8 = 0; \
		m9 = 0; \
		mA = 0; \
		mB = 0; \
		mC = 0; \
		mD = 0; \
		mE = 0; \
		mF = 0; \
		for (u = 0; u < 8; u ++) { \
			unsigned db = buf(u); \
			for (v = 0; v < 8; v ++, db >>= 1) { \
				sph_u32 dm = SPH_T32(-(sph_u32)(db & 1)); \
				m0 ^= dm & __ldg(tp ++); \
				m1 ^= dm & __ldg(tp ++); \
				m2 ^= dm & __ldg(tp ++); \
				m3 ^= dm & __ldg(tp ++); \
				m4 ^= dm & __ldg(tp ++); \
				m5 ^= dm & __ldg(tp ++); \
				m6 ^= dm & __ldg(tp ++); \
				m7 ^= dm & __ldg(tp ++); \
				m8 ^= dm & __ldg(tp ++); \
				m9 ^= dm & __ldg(tp ++); \
				mA ^= dm & __ldg(tp ++); \
				mB ^= dm & __ldg(tp ++); \
				mC ^= dm & __ldg(tp ++); \
				mD ^= dm & __ldg(tp ++); \
				mE ^= dm & __ldg(tp ++); \
				mF ^= dm & __ldg(tp ++); \
			} \
		} \
	} while (0)
*/

#define INPUT_BIG   do { \
                 const sph_u32 *tp = &T512[0][0]; \
                unsigned u, v; \
                m0 = 0; \
                m1 = 0; \
                m2 = 0; \
                m3 = 0; \
                m4 = 0; \
                m5 = 0; \
                m6 = 0; \
                m7 = 0; \
                m8 = 0; \
                m9 = 0; \
                mA = 0; \
                mB = 0; \
                mC = 0; \
                mD = 0; \
                mE = 0; \
                mF = 0; \
                for (u = 0; u < 8; u ++) { \
                        unsigned db = buf(u); \
                        for (v = 0; v < 8; v ++, db >>= 1) { \
                                sph_u32 dm = SPH_T32(-(sph_u32)(db & 1)); \
                                m0 ^= dm & __ldg(tp ++); \
                                m1 ^= dm & *tp ++; \
                                m2 ^= dm & *tp ++; \
                                m3 ^= dm & *tp ++; \
                                m4 ^= dm & *tp ++; \
                                m5 ^= dm & *tp ++; \
                                m6 ^= dm & *tp ++; \
                                m7 ^= dm & *tp ++; \
                                m8 ^= dm & *tp ++; \
                                m9 ^= dm & *tp ++; \
                                mA ^= dm & *tp ++; \
                                mB ^= dm & *tp ++; \
                                mC ^= dm & *tp ++; \
                                mD ^= dm & *tp ++; \
                                mE ^= dm & *tp ++; \
                                mF ^= dm & *tp ++; \
                        } \
                } \
        } while (0)




#define TPB_H 512
__global__ __launch_bounds__(TPB_H,2)
void x13_hamsi512_gpu_hash_64(uint32_t threads, uint32_t *g_hash){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t *Hash = &g_hash[thread<<4];
	//	uint32_t hx[16];
		uint8_t __align__(16) h1[64];

//		*(uint2x4*)&hx[ 0] = *(uint2x4*)&Hash[0];
//		*(uint2x4*)&hx[8 ] = *(uint2x4*)&Hash[8];

//		for(int i=0;i<16;i++)hx[i]=cuda_swab32(hx[i]);
//		*(uint2x4*)&h1[ 0] = *(uint2x4*)&hx[0];
//                *(uint2x4*)&h1[32] = *(uint2x4*)&hx[8];
/*
		const sph_u32 *tpp = &T512[0][0]; 
		__shared__ sph_u32 TXX[1024];
		if(threadIdx.x < 512){
		TXX[threadIdx.x] = tpp[threadIdx.x];
                TXX[threadIdx.x+512] = tpp[threadIdx.x+512];

}*/

                *(uint2x4*)&h1[ 0] = *(uint2x4*)&Hash[0];
                *(uint2x4*)&h1[32] = *(uint2x4*)&Hash[8];
//for(int i=0;i<32;i++) h1[i]=0;

//__syncthreads();
		  // hamsi
  sph_u32 c0 = HAMSI_IV512[0], c1 = HAMSI_IV512[1], c2 = HAMSI_IV512[2], c3 = HAMSI_IV512[3];
  sph_u32 c4 = HAMSI_IV512[4], c5 = HAMSI_IV512[5], c6 = HAMSI_IV512[6], c7 = HAMSI_IV512[7];
  sph_u32 c8 = HAMSI_IV512[8], c9 = HAMSI_IV512[9], cA = HAMSI_IV512[10], cB = HAMSI_IV512[11];
  sph_u32 cC = HAMSI_IV512[12], cD = HAMSI_IV512[13], cE = HAMSI_IV512[14], cF = HAMSI_IV512[15];
  sph_u32 m0, m1, m2, m3, m4, m5, m6, m7;
  sph_u32 m8, m9, mA, mB, mC, mD, mE, mF;
  sph_u32 h[16] = { c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, cA, cB, cC, cD, cE, cF };

  #define buf(u) h1[i + u]
//#define buf(u) 0

//#pragma unroll 8
  for(int i = 0; i < 64; i += 8)
  {

#pragma unroll 8
    INPUT_BIG;
    P_BIG;
    T_BIG;
  }

  #undef buf
  #define buf(u) 0
//#pragma unroll 8
  for(int i = 0; i < 64; i += 8)
  {
#pragma unroll 8
    INPUT_BIG;
    P_BIG;
    T_BIG;
  }

  #undef buf
  #define buf(u) (u == 0 ? 0x80 : 0)

#pragma unroll 8
  INPUT_BIG;
  P_BIG;
  T_BIG;

  #undef buf
  #define buf(u) (u == 6 ? 4 : 0)

#pragma unroll 8
  INPUT_BIG;
  PF_BIG;
  T_BIG;

//  for (unsigned u = 0; u < 16; u ++)
  //    hash->h4[u] = h[u];

#pragma unroll 16
for(int i=0;i<16;i++)h[i]=cuda_swab32(h[i]);

		*(uint2x4*)&Hash[ 0] = *(uint2x4*)&h[ 0];
		*(uint2x4*)&Hash[ 8] = *(uint2x4*)&h[ 8];
	}
}

__host__
void x13_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	const uint32_t threadsperblock = TPB_H;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x13_hamsi512_gpu_hash_64<<<grid, block>>>(threads, d_hash);

}

#define sph_u64 uint64_t


__constant__ static const sph_u64 plain_T0[256] = {
	SPH_C64(0xD83078C018601818), SPH_C64(0x2646AF05238C2323),
	SPH_C64(0xB891F97EC63FC6C6), SPH_C64(0xFBCD6F13E887E8E8),
	SPH_C64(0xCB13A14C87268787), SPH_C64(0x116D62A9B8DAB8B8),
	SPH_C64(0x0902050801040101), SPH_C64(0x0D9E6E424F214F4F),
	SPH_C64(0x9B6CEEAD36D83636), SPH_C64(0xFF510459A6A2A6A6),
	SPH_C64(0x0CB9BDDED26FD2D2), SPH_C64(0x0EF706FBF5F3F5F5),
	SPH_C64(0x96F280EF79F97979), SPH_C64(0x30DECE5F6FA16F6F),
	SPH_C64(0x6D3FEFFC917E9191), SPH_C64(0xF8A407AA52555252),
	SPH_C64(0x47C0FD27609D6060), SPH_C64(0x35657689BCCABCBC),
	SPH_C64(0x372BCDAC9B569B9B), SPH_C64(0x8A018C048E028E8E),
	SPH_C64(0xD25B1571A3B6A3A3), SPH_C64(0x6C183C600C300C0C),
	SPH_C64(0x84F68AFF7BF17B7B), SPH_C64(0x806AE1B535D43535),
	SPH_C64(0xF53A69E81D741D1D), SPH_C64(0xB3DD4753E0A7E0E0),
	SPH_C64(0x21B3ACF6D77BD7D7), SPH_C64(0x9C99ED5EC22FC2C2),
	SPH_C64(0x435C966D2EB82E2E), SPH_C64(0x29967A624B314B4B),
	SPH_C64(0x5DE121A3FEDFFEFE), SPH_C64(0xD5AE168257415757),
	SPH_C64(0xBD2A41A815541515), SPH_C64(0xE8EEB69F77C17777),
	SPH_C64(0x926EEBA537DC3737), SPH_C64(0x9ED7567BE5B3E5E5),
	SPH_C64(0x1323D98C9F469F9F), SPH_C64(0x23FD17D3F0E7F0F0),
	SPH_C64(0x20947F6A4A354A4A), SPH_C64(0x44A9959EDA4FDADA),
	SPH_C64(0xA2B025FA587D5858), SPH_C64(0xCF8FCA06C903C9C9),
	SPH_C64(0x7C528D5529A42929), SPH_C64(0x5A1422500A280A0A),
	SPH_C64(0x507F4FE1B1FEB1B1), SPH_C64(0xC95D1A69A0BAA0A0),
	SPH_C64(0x14D6DA7F6BB16B6B), SPH_C64(0xD917AB5C852E8585),
	SPH_C64(0x3C677381BDCEBDBD), SPH_C64(0x8FBA34D25D695D5D),
	SPH_C64(0x9020508010401010), SPH_C64(0x07F503F3F4F7F4F4),
	SPH_C64(0xDD8BC016CB0BCBCB), SPH_C64(0xD37CC6ED3EF83E3E),
	SPH_C64(0x2D0A112805140505), SPH_C64(0x78CEE61F67816767),
	SPH_C64(0x97D55373E4B7E4E4), SPH_C64(0x024EBB25279C2727),
	SPH_C64(0x7382583241194141), SPH_C64(0xA70B9D2C8B168B8B),
	SPH_C64(0xF6530151A7A6A7A7), SPH_C64(0xB2FA94CF7DE97D7D),
	SPH_C64(0x4937FBDC956E9595), SPH_C64(0x56AD9F8ED847D8D8),
	SPH_C64(0x70EB308BFBCBFBFB), SPH_C64(0xCDC17123EE9FEEEE),
	SPH_C64(0xBBF891C77CED7C7C), SPH_C64(0x71CCE31766856666),
	SPH_C64(0x7BA78EA6DD53DDDD), SPH_C64(0xAF2E4BB8175C1717),
	SPH_C64(0x458E460247014747), SPH_C64(0x1A21DC849E429E9E),
	SPH_C64(0xD489C51ECA0FCACA), SPH_C64(0x585A99752DB42D2D),
	SPH_C64(0x2E637991BFC6BFBF), SPH_C64(0x3F0E1B38071C0707),
	SPH_C64(0xAC472301AD8EADAD), SPH_C64(0xB0B42FEA5A755A5A),
	SPH_C64(0xEF1BB56C83368383), SPH_C64(0xB666FF8533CC3333),
	SPH_C64(0x5CC6F23F63916363), SPH_C64(0x12040A1002080202),
	SPH_C64(0x93493839AA92AAAA), SPH_C64(0xDEE2A8AF71D97171),
	SPH_C64(0xC68DCF0EC807C8C8), SPH_C64(0xD1327DC819641919),
	SPH_C64(0x3B92707249394949), SPH_C64(0x5FAF9A86D943D9D9),
	SPH_C64(0x31F91DC3F2EFF2F2), SPH_C64(0xA8DB484BE3ABE3E3),
	SPH_C64(0xB9B62AE25B715B5B), SPH_C64(0xBC0D9234881A8888),
	SPH_C64(0x3E29C8A49A529A9A), SPH_C64(0x0B4CBE2D26982626),
	SPH_C64(0xBF64FA8D32C83232), SPH_C64(0x597D4AE9B0FAB0B0),
	SPH_C64(0xF2CF6A1BE983E9E9), SPH_C64(0x771E33780F3C0F0F),
	SPH_C64(0x33B7A6E6D573D5D5), SPH_C64(0xF41DBA74803A8080),
	SPH_C64(0x27617C99BEC2BEBE), SPH_C64(0xEB87DE26CD13CDCD),
	SPH_C64(0x8968E4BD34D03434), SPH_C64(0x3290757A483D4848),
	SPH_C64(0x54E324ABFFDBFFFF), SPH_C64(0x8DF48FF77AF57A7A),
	SPH_C64(0x643DEAF4907A9090), SPH_C64(0x9DBE3EC25F615F5F),
	SPH_C64(0x3D40A01D20802020), SPH_C64(0x0FD0D56768BD6868),
	SPH_C64(0xCA3472D01A681A1A), SPH_C64(0xB7412C19AE82AEAE),
	SPH_C64(0x7D755EC9B4EAB4B4), SPH_C64(0xCEA8199A544D5454),
	SPH_C64(0x7F3BE5EC93769393), SPH_C64(0x2F44AA0D22882222),
	SPH_C64(0x63C8E907648D6464), SPH_C64(0x2AFF12DBF1E3F1F1),
	SPH_C64(0xCCE6A2BF73D17373), SPH_C64(0x82245A9012481212),
	SPH_C64(0x7A805D3A401D4040), SPH_C64(0x4810284008200808),
	SPH_C64(0x959BE856C32BC3C3), SPH_C64(0xDFC57B33EC97ECEC),
	SPH_C64(0x4DAB9096DB4BDBDB), SPH_C64(0xC05F1F61A1BEA1A1),
	SPH_C64(0x9107831C8D0E8D8D), SPH_C64(0xC87AC9F53DF43D3D),
	SPH_C64(0x5B33F1CC97669797), SPH_C64(0x0000000000000000),
	SPH_C64(0xF983D436CF1BCFCF), SPH_C64(0x6E5687452BAC2B2B),
	SPH_C64(0xE1ECB39776C57676), SPH_C64(0xE619B06482328282),
	SPH_C64(0x28B1A9FED67FD6D6), SPH_C64(0xC33677D81B6C1B1B),
	SPH_C64(0x74775BC1B5EEB5B5), SPH_C64(0xBE432911AF86AFAF),
	SPH_C64(0x1DD4DF776AB56A6A), SPH_C64(0xEAA00DBA505D5050),
	SPH_C64(0x578A4C1245094545), SPH_C64(0x38FB18CBF3EBF3F3),
	SPH_C64(0xAD60F09D30C03030), SPH_C64(0xC4C3742BEF9BEFEF),
	SPH_C64(0xDA7EC3E53FFC3F3F), SPH_C64(0xC7AA1C9255495555),
	SPH_C64(0xDB591079A2B2A2A2), SPH_C64(0xE9C96503EA8FEAEA),
	SPH_C64(0x6ACAEC0F65896565), SPH_C64(0x036968B9BAD2BABA),
	SPH_C64(0x4A5E93652FBC2F2F), SPH_C64(0x8E9DE74EC027C0C0),
	SPH_C64(0x60A181BEDE5FDEDE), SPH_C64(0xFC386CE01C701C1C),
	SPH_C64(0x46E72EBBFDD3FDFD), SPH_C64(0x1F9A64524D294D4D),
	SPH_C64(0x7639E0E492729292), SPH_C64(0xFAEABC8F75C97575),
	SPH_C64(0x360C1E3006180606), SPH_C64(0xAE0998248A128A8A),
	SPH_C64(0x4B7940F9B2F2B2B2), SPH_C64(0x85D15963E6BFE6E6),
	SPH_C64(0x7E1C36700E380E0E), SPH_C64(0xE73E63F81F7C1F1F),
	SPH_C64(0x55C4F73762956262), SPH_C64(0x3AB5A3EED477D4D4),
	SPH_C64(0x814D3229A89AA8A8), SPH_C64(0x5231F4C496629696),
	SPH_C64(0x62EF3A9BF9C3F9F9), SPH_C64(0xA397F666C533C5C5),
	SPH_C64(0x104AB13525942525), SPH_C64(0xABB220F259795959),
	SPH_C64(0xD015AE54842A8484), SPH_C64(0xC5E4A7B772D57272),
	SPH_C64(0xEC72DDD539E43939), SPH_C64(0x1698615A4C2D4C4C),
	SPH_C64(0x94BC3BCA5E655E5E), SPH_C64(0x9FF085E778FD7878),
	SPH_C64(0xE570D8DD38E03838), SPH_C64(0x980586148C0A8C8C),
	SPH_C64(0x17BFB2C6D163D1D1), SPH_C64(0xE4570B41A5AEA5A5),
	SPH_C64(0xA1D94D43E2AFE2E2), SPH_C64(0x4EC2F82F61996161),
	SPH_C64(0x427B45F1B3F6B3B3), SPH_C64(0x3442A51521842121),
	SPH_C64(0x0825D6949C4A9C9C), SPH_C64(0xEE3C66F01E781E1E),
	SPH_C64(0x6186522243114343), SPH_C64(0xB193FC76C73BC7C7),
	SPH_C64(0x4FE52BB3FCD7FCFC), SPH_C64(0x2408142004100404),
	SPH_C64(0xE3A208B251595151), SPH_C64(0x252FC7BC995E9999),
	SPH_C64(0x22DAC44F6DA96D6D), SPH_C64(0x651A39680D340D0D),
	SPH_C64(0x79E93583FACFFAFA), SPH_C64(0x69A384B6DF5BDFDF),
	SPH_C64(0xA9FC9BD77EE57E7E), SPH_C64(0x1948B43D24902424),
	SPH_C64(0xFE76D7C53BEC3B3B), SPH_C64(0x9A4B3D31AB96ABAB),
	SPH_C64(0xF081D13ECE1FCECE), SPH_C64(0x9922558811441111),
	SPH_C64(0x8303890C8F068F8F), SPH_C64(0x049C6B4A4E254E4E),
	SPH_C64(0x667351D1B7E6B7B7), SPH_C64(0xE0CB600BEB8BEBEB),
	SPH_C64(0xC178CCFD3CF03C3C), SPH_C64(0xFD1FBF7C813E8181),
	SPH_C64(0x4035FED4946A9494), SPH_C64(0x1CF30CEBF7FBF7F7),
	SPH_C64(0x186F67A1B9DEB9B9), SPH_C64(0x8B265F98134C1313),
	SPH_C64(0x51589C7D2CB02C2C), SPH_C64(0x05BBB8D6D36BD3D3),
	SPH_C64(0x8CD35C6BE7BBE7E7), SPH_C64(0x39DCCB576EA56E6E),
	SPH_C64(0xAA95F36EC437C4C4), SPH_C64(0x1B060F18030C0303),
	SPH_C64(0xDCAC138A56455656), SPH_C64(0x5E88491A440D4444),
	SPH_C64(0xA0FE9EDF7FE17F7F), SPH_C64(0x884F3721A99EA9A9),
	SPH_C64(0x6754824D2AA82A2A), SPH_C64(0x0A6B6DB1BBD6BBBB),
	SPH_C64(0x879FE246C123C1C1), SPH_C64(0xF1A602A253515353),
	SPH_C64(0x72A58BAEDC57DCDC), SPH_C64(0x531627580B2C0B0B),
	SPH_C64(0x0127D39C9D4E9D9D), SPH_C64(0x2BD8C1476CAD6C6C),
	SPH_C64(0xA462F59531C43131), SPH_C64(0xF3E8B98774CD7474),
	SPH_C64(0x15F109E3F6FFF6F6), SPH_C64(0x4C8C430A46054646),
	SPH_C64(0xA5452609AC8AACAC), SPH_C64(0xB50F973C891E8989),
	SPH_C64(0xB42844A014501414), SPH_C64(0xBADF425BE1A3E1E1),
	SPH_C64(0xA62C4EB016581616), SPH_C64(0xF774D2CD3AE83A3A),
	SPH_C64(0x06D2D06F69B96969), SPH_C64(0x41122D4809240909),
	SPH_C64(0xD7E0ADA770DD7070), SPH_C64(0x6F7154D9B6E2B6B6),
	SPH_C64(0x1EBDB7CED067D0D0), SPH_C64(0xD6C77E3BED93EDED),
	SPH_C64(0xE285DB2ECC17CCCC), SPH_C64(0x6884572A42154242),
	SPH_C64(0x2C2DC2B4985A9898), SPH_C64(0xED550E49A4AAA4A4),
	SPH_C64(0x7550885D28A02828), SPH_C64(0x86B831DA5C6D5C5C),
	SPH_C64(0x6BED3F93F8C7F8F8), SPH_C64(0xC211A44486228686)
};


__constant__ static const sph_u64 plain_T1[256] = {
	SPH_C64(0x3078C018601818D8), SPH_C64(0x46AF05238C232326),
	SPH_C64(0x91F97EC63FC6C6B8), SPH_C64(0xCD6F13E887E8E8FB),
	SPH_C64(0x13A14C87268787CB), SPH_C64(0x6D62A9B8DAB8B811),
	SPH_C64(0x0205080104010109), SPH_C64(0x9E6E424F214F4F0D),
	SPH_C64(0x6CEEAD36D836369B), SPH_C64(0x510459A6A2A6A6FF),
	SPH_C64(0xB9BDDED26FD2D20C), SPH_C64(0xF706FBF5F3F5F50E),
	SPH_C64(0xF280EF79F9797996), SPH_C64(0xDECE5F6FA16F6F30),
	SPH_C64(0x3FEFFC917E91916D), SPH_C64(0xA407AA52555252F8),
	SPH_C64(0xC0FD27609D606047), SPH_C64(0x657689BCCABCBC35),
	SPH_C64(0x2BCDAC9B569B9B37), SPH_C64(0x018C048E028E8E8A),
	SPH_C64(0x5B1571A3B6A3A3D2), SPH_C64(0x183C600C300C0C6C),
	SPH_C64(0xF68AFF7BF17B7B84), SPH_C64(0x6AE1B535D4353580),
	SPH_C64(0x3A69E81D741D1DF5), SPH_C64(0xDD4753E0A7E0E0B3),
	SPH_C64(0xB3ACF6D77BD7D721), SPH_C64(0x99ED5EC22FC2C29C),
	SPH_C64(0x5C966D2EB82E2E43), SPH_C64(0x967A624B314B4B29),
	SPH_C64(0xE121A3FEDFFEFE5D), SPH_C64(0xAE168257415757D5),
	SPH_C64(0x2A41A815541515BD), SPH_C64(0xEEB69F77C17777E8),
	SPH_C64(0x6EEBA537DC373792), SPH_C64(0xD7567BE5B3E5E59E),
	SPH_C64(0x23D98C9F469F9F13), SPH_C64(0xFD17D3F0E7F0F023),
	SPH_C64(0x947F6A4A354A4A20), SPH_C64(0xA9959EDA4FDADA44),
	SPH_C64(0xB025FA587D5858A2), SPH_C64(0x8FCA06C903C9C9CF),
	SPH_C64(0x528D5529A429297C), SPH_C64(0x1422500A280A0A5A),
	SPH_C64(0x7F4FE1B1FEB1B150), SPH_C64(0x5D1A69A0BAA0A0C9),
	SPH_C64(0xD6DA7F6BB16B6B14), SPH_C64(0x17AB5C852E8585D9),
	SPH_C64(0x677381BDCEBDBD3C), SPH_C64(0xBA34D25D695D5D8F),
	SPH_C64(0x2050801040101090), SPH_C64(0xF503F3F4F7F4F407),
	SPH_C64(0x8BC016CB0BCBCBDD), SPH_C64(0x7CC6ED3EF83E3ED3),
	SPH_C64(0x0A1128051405052D), SPH_C64(0xCEE61F6781676778),
	SPH_C64(0xD55373E4B7E4E497), SPH_C64(0x4EBB25279C272702),
	SPH_C64(0x8258324119414173), SPH_C64(0x0B9D2C8B168B8BA7),
	SPH_C64(0x530151A7A6A7A7F6), SPH_C64(0xFA94CF7DE97D7DB2),
	SPH_C64(0x37FBDC956E959549), SPH_C64(0xAD9F8ED847D8D856),
	SPH_C64(0xEB308BFBCBFBFB70), SPH_C64(0xC17123EE9FEEEECD),
	SPH_C64(0xF891C77CED7C7CBB), SPH_C64(0xCCE3176685666671),
	SPH_C64(0xA78EA6DD53DDDD7B), SPH_C64(0x2E4BB8175C1717AF),
	SPH_C64(0x8E46024701474745), SPH_C64(0x21DC849E429E9E1A),
	SPH_C64(0x89C51ECA0FCACAD4), SPH_C64(0x5A99752DB42D2D58),
	SPH_C64(0x637991BFC6BFBF2E), SPH_C64(0x0E1B38071C07073F),
	SPH_C64(0x472301AD8EADADAC), SPH_C64(0xB42FEA5A755A5AB0),
	SPH_C64(0x1BB56C83368383EF), SPH_C64(0x66FF8533CC3333B6),
	SPH_C64(0xC6F23F639163635C), SPH_C64(0x040A100208020212),
	SPH_C64(0x493839AA92AAAA93), SPH_C64(0xE2A8AF71D97171DE),
	SPH_C64(0x8DCF0EC807C8C8C6), SPH_C64(0x327DC819641919D1),
	SPH_C64(0x927072493949493B), SPH_C64(0xAF9A86D943D9D95F),
	SPH_C64(0xF91DC3F2EFF2F231), SPH_C64(0xDB484BE3ABE3E3A8),
	SPH_C64(0xB62AE25B715B5BB9), SPH_C64(0x0D9234881A8888BC),
	SPH_C64(0x29C8A49A529A9A3E), SPH_C64(0x4CBE2D269826260B),
	SPH_C64(0x64FA8D32C83232BF), SPH_C64(0x7D4AE9B0FAB0B059),
	SPH_C64(0xCF6A1BE983E9E9F2), SPH_C64(0x1E33780F3C0F0F77),
	SPH_C64(0xB7A6E6D573D5D533), SPH_C64(0x1DBA74803A8080F4),
	SPH_C64(0x617C99BEC2BEBE27), SPH_C64(0x87DE26CD13CDCDEB),
	SPH_C64(0x68E4BD34D0343489), SPH_C64(0x90757A483D484832),
	SPH_C64(0xE324ABFFDBFFFF54), SPH_C64(0xF48FF77AF57A7A8D),
	SPH_C64(0x3DEAF4907A909064), SPH_C64(0xBE3EC25F615F5F9D),
	SPH_C64(0x40A01D208020203D), SPH_C64(0xD0D56768BD68680F),
	SPH_C64(0x3472D01A681A1ACA), SPH_C64(0x412C19AE82AEAEB7),
	SPH_C64(0x755EC9B4EAB4B47D), SPH_C64(0xA8199A544D5454CE),
	SPH_C64(0x3BE5EC937693937F), SPH_C64(0x44AA0D228822222F),
	SPH_C64(0xC8E907648D646463), SPH_C64(0xFF12DBF1E3F1F12A),
	SPH_C64(0xE6A2BF73D17373CC), SPH_C64(0x245A901248121282),
	SPH_C64(0x805D3A401D40407A), SPH_C64(0x1028400820080848),
	SPH_C64(0x9BE856C32BC3C395), SPH_C64(0xC57B33EC97ECECDF),
	SPH_C64(0xAB9096DB4BDBDB4D), SPH_C64(0x5F1F61A1BEA1A1C0),
	SPH_C64(0x07831C8D0E8D8D91), SPH_C64(0x7AC9F53DF43D3DC8),
	SPH_C64(0x33F1CC976697975B), SPH_C64(0x0000000000000000),
	SPH_C64(0x83D436CF1BCFCFF9), SPH_C64(0x5687452BAC2B2B6E),
	SPH_C64(0xECB39776C57676E1), SPH_C64(0x19B06482328282E6),
	SPH_C64(0xB1A9FED67FD6D628), SPH_C64(0x3677D81B6C1B1BC3),
	SPH_C64(0x775BC1B5EEB5B574), SPH_C64(0x432911AF86AFAFBE),
	SPH_C64(0xD4DF776AB56A6A1D), SPH_C64(0xA00DBA505D5050EA),
	SPH_C64(0x8A4C124509454557), SPH_C64(0xFB18CBF3EBF3F338),
	SPH_C64(0x60F09D30C03030AD), SPH_C64(0xC3742BEF9BEFEFC4),
	SPH_C64(0x7EC3E53FFC3F3FDA), SPH_C64(0xAA1C9255495555C7),
	SPH_C64(0x591079A2B2A2A2DB), SPH_C64(0xC96503EA8FEAEAE9),
	SPH_C64(0xCAEC0F658965656A), SPH_C64(0x6968B9BAD2BABA03),
	SPH_C64(0x5E93652FBC2F2F4A), SPH_C64(0x9DE74EC027C0C08E),
	SPH_C64(0xA181BEDE5FDEDE60), SPH_C64(0x386CE01C701C1CFC),
	SPH_C64(0xE72EBBFDD3FDFD46), SPH_C64(0x9A64524D294D4D1F),
	SPH_C64(0x39E0E49272929276), SPH_C64(0xEABC8F75C97575FA),
	SPH_C64(0x0C1E300618060636), SPH_C64(0x0998248A128A8AAE),
	SPH_C64(0x7940F9B2F2B2B24B), SPH_C64(0xD15963E6BFE6E685),
	SPH_C64(0x1C36700E380E0E7E), SPH_C64(0x3E63F81F7C1F1FE7),
	SPH_C64(0xC4F7376295626255), SPH_C64(0xB5A3EED477D4D43A),
	SPH_C64(0x4D3229A89AA8A881), SPH_C64(0x31F4C49662969652),
	SPH_C64(0xEF3A9BF9C3F9F962), SPH_C64(0x97F666C533C5C5A3),
	SPH_C64(0x4AB1352594252510), SPH_C64(0xB220F259795959AB),
	SPH_C64(0x15AE54842A8484D0), SPH_C64(0xE4A7B772D57272C5),
	SPH_C64(0x72DDD539E43939EC), SPH_C64(0x98615A4C2D4C4C16),
	SPH_C64(0xBC3BCA5E655E5E94), SPH_C64(0xF085E778FD78789F),
	SPH_C64(0x70D8DD38E03838E5), SPH_C64(0x0586148C0A8C8C98),
	SPH_C64(0xBFB2C6D163D1D117), SPH_C64(0x570B41A5AEA5A5E4),
	SPH_C64(0xD94D43E2AFE2E2A1), SPH_C64(0xC2F82F619961614E),
	SPH_C64(0x7B45F1B3F6B3B342), SPH_C64(0x42A5152184212134),
	SPH_C64(0x25D6949C4A9C9C08), SPH_C64(0x3C66F01E781E1EEE),
	SPH_C64(0x8652224311434361), SPH_C64(0x93FC76C73BC7C7B1),
	SPH_C64(0xE52BB3FCD7FCFC4F), SPH_C64(0x0814200410040424),
	SPH_C64(0xA208B251595151E3), SPH_C64(0x2FC7BC995E999925),
	SPH_C64(0xDAC44F6DA96D6D22), SPH_C64(0x1A39680D340D0D65),
	SPH_C64(0xE93583FACFFAFA79), SPH_C64(0xA384B6DF5BDFDF69),
	SPH_C64(0xFC9BD77EE57E7EA9), SPH_C64(0x48B43D2490242419),
	SPH_C64(0x76D7C53BEC3B3BFE), SPH_C64(0x4B3D31AB96ABAB9A),
	SPH_C64(0x81D13ECE1FCECEF0), SPH_C64(0x2255881144111199),
	SPH_C64(0x03890C8F068F8F83), SPH_C64(0x9C6B4A4E254E4E04),
	SPH_C64(0x7351D1B7E6B7B766), SPH_C64(0xCB600BEB8BEBEBE0),
	SPH_C64(0x78CCFD3CF03C3CC1), SPH_C64(0x1FBF7C813E8181FD),
	SPH_C64(0x35FED4946A949440), SPH_C64(0xF30CEBF7FBF7F71C),
	SPH_C64(0x6F67A1B9DEB9B918), SPH_C64(0x265F98134C13138B),
	SPH_C64(0x589C7D2CB02C2C51), SPH_C64(0xBBB8D6D36BD3D305),
	SPH_C64(0xD35C6BE7BBE7E78C), SPH_C64(0xDCCB576EA56E6E39),
	SPH_C64(0x95F36EC437C4C4AA), SPH_C64(0x060F18030C03031B),
	SPH_C64(0xAC138A56455656DC), SPH_C64(0x88491A440D44445E),
	SPH_C64(0xFE9EDF7FE17F7FA0), SPH_C64(0x4F3721A99EA9A988),
	SPH_C64(0x54824D2AA82A2A67), SPH_C64(0x6B6DB1BBD6BBBB0A),
	SPH_C64(0x9FE246C123C1C187), SPH_C64(0xA602A253515353F1),
	SPH_C64(0xA58BAEDC57DCDC72), SPH_C64(0x1627580B2C0B0B53),
	SPH_C64(0x27D39C9D4E9D9D01), SPH_C64(0xD8C1476CAD6C6C2B),
	SPH_C64(0x62F59531C43131A4), SPH_C64(0xE8B98774CD7474F3),
	SPH_C64(0xF109E3F6FFF6F615), SPH_C64(0x8C430A460546464C),
	SPH_C64(0x452609AC8AACACA5), SPH_C64(0x0F973C891E8989B5),
	SPH_C64(0x2844A014501414B4), SPH_C64(0xDF425BE1A3E1E1BA),
	SPH_C64(0x2C4EB016581616A6), SPH_C64(0x74D2CD3AE83A3AF7),
	SPH_C64(0xD2D06F69B9696906), SPH_C64(0x122D480924090941),
	SPH_C64(0xE0ADA770DD7070D7), SPH_C64(0x7154D9B6E2B6B66F),
	SPH_C64(0xBDB7CED067D0D01E), SPH_C64(0xC77E3BED93EDEDD6),
	SPH_C64(0x85DB2ECC17CCCCE2), SPH_C64(0x84572A4215424268),
	SPH_C64(0x2DC2B4985A98982C), SPH_C64(0x550E49A4AAA4A4ED),
	SPH_C64(0x50885D28A0282875), SPH_C64(0xB831DA5C6D5C5C86),
	SPH_C64(0xED3F93F8C7F8F86B), SPH_C64(0x11A44486228686C2)
};

__constant__ static const sph_u64 plain_T2[256] = {
	SPH_C64(0x78C018601818D830), SPH_C64(0xAF05238C23232646),
	SPH_C64(0xF97EC63FC6C6B891), SPH_C64(0x6F13E887E8E8FBCD),
	SPH_C64(0xA14C87268787CB13), SPH_C64(0x62A9B8DAB8B8116D),
	SPH_C64(0x0508010401010902), SPH_C64(0x6E424F214F4F0D9E),
	SPH_C64(0xEEAD36D836369B6C), SPH_C64(0x0459A6A2A6A6FF51),
	SPH_C64(0xBDDED26FD2D20CB9), SPH_C64(0x06FBF5F3F5F50EF7),
	SPH_C64(0x80EF79F9797996F2), SPH_C64(0xCE5F6FA16F6F30DE),
	SPH_C64(0xEFFC917E91916D3F), SPH_C64(0x07AA52555252F8A4),
	SPH_C64(0xFD27609D606047C0), SPH_C64(0x7689BCCABCBC3565),
	SPH_C64(0xCDAC9B569B9B372B), SPH_C64(0x8C048E028E8E8A01),
	SPH_C64(0x1571A3B6A3A3D25B), SPH_C64(0x3C600C300C0C6C18),
	SPH_C64(0x8AFF7BF17B7B84F6), SPH_C64(0xE1B535D43535806A),
	SPH_C64(0x69E81D741D1DF53A), SPH_C64(0x4753E0A7E0E0B3DD),
	SPH_C64(0xACF6D77BD7D721B3), SPH_C64(0xED5EC22FC2C29C99),
	SPH_C64(0x966D2EB82E2E435C), SPH_C64(0x7A624B314B4B2996),
	SPH_C64(0x21A3FEDFFEFE5DE1), SPH_C64(0x168257415757D5AE),
	SPH_C64(0x41A815541515BD2A), SPH_C64(0xB69F77C17777E8EE),
	SPH_C64(0xEBA537DC3737926E), SPH_C64(0x567BE5B3E5E59ED7),
	SPH_C64(0xD98C9F469F9F1323), SPH_C64(0x17D3F0E7F0F023FD),
	SPH_C64(0x7F6A4A354A4A2094), SPH_C64(0x959EDA4FDADA44A9),
	SPH_C64(0x25FA587D5858A2B0), SPH_C64(0xCA06C903C9C9CF8F),
	SPH_C64(0x8D5529A429297C52), SPH_C64(0x22500A280A0A5A14),
	SPH_C64(0x4FE1B1FEB1B1507F), SPH_C64(0x1A69A0BAA0A0C95D),
	SPH_C64(0xDA7F6BB16B6B14D6), SPH_C64(0xAB5C852E8585D917),
	SPH_C64(0x7381BDCEBDBD3C67), SPH_C64(0x34D25D695D5D8FBA),
	SPH_C64(0x5080104010109020), SPH_C64(0x03F3F4F7F4F407F5),
	SPH_C64(0xC016CB0BCBCBDD8B), SPH_C64(0xC6ED3EF83E3ED37C),
	SPH_C64(0x1128051405052D0A), SPH_C64(0xE61F6781676778CE),
	SPH_C64(0x5373E4B7E4E497D5), SPH_C64(0xBB25279C2727024E),
	SPH_C64(0x5832411941417382), SPH_C64(0x9D2C8B168B8BA70B),
	SPH_C64(0x0151A7A6A7A7F653), SPH_C64(0x94CF7DE97D7DB2FA),
	SPH_C64(0xFBDC956E95954937), SPH_C64(0x9F8ED847D8D856AD),
	SPH_C64(0x308BFBCBFBFB70EB), SPH_C64(0x7123EE9FEEEECDC1),
	SPH_C64(0x91C77CED7C7CBBF8), SPH_C64(0xE3176685666671CC),
	SPH_C64(0x8EA6DD53DDDD7BA7), SPH_C64(0x4BB8175C1717AF2E),
	SPH_C64(0x460247014747458E), SPH_C64(0xDC849E429E9E1A21),
	SPH_C64(0xC51ECA0FCACAD489), SPH_C64(0x99752DB42D2D585A),
	SPH_C64(0x7991BFC6BFBF2E63), SPH_C64(0x1B38071C07073F0E),
	SPH_C64(0x2301AD8EADADAC47), SPH_C64(0x2FEA5A755A5AB0B4),
	SPH_C64(0xB56C83368383EF1B), SPH_C64(0xFF8533CC3333B666),
	SPH_C64(0xF23F639163635CC6), SPH_C64(0x0A10020802021204),
	SPH_C64(0x3839AA92AAAA9349), SPH_C64(0xA8AF71D97171DEE2),
	SPH_C64(0xCF0EC807C8C8C68D), SPH_C64(0x7DC819641919D132),
	SPH_C64(0x7072493949493B92), SPH_C64(0x9A86D943D9D95FAF),
	SPH_C64(0x1DC3F2EFF2F231F9), SPH_C64(0x484BE3ABE3E3A8DB),
	SPH_C64(0x2AE25B715B5BB9B6), SPH_C64(0x9234881A8888BC0D),
	SPH_C64(0xC8A49A529A9A3E29), SPH_C64(0xBE2D269826260B4C),
	SPH_C64(0xFA8D32C83232BF64), SPH_C64(0x4AE9B0FAB0B0597D),
	SPH_C64(0x6A1BE983E9E9F2CF), SPH_C64(0x33780F3C0F0F771E),
	SPH_C64(0xA6E6D573D5D533B7), SPH_C64(0xBA74803A8080F41D),
	SPH_C64(0x7C99BEC2BEBE2761), SPH_C64(0xDE26CD13CDCDEB87),
	SPH_C64(0xE4BD34D034348968), SPH_C64(0x757A483D48483290),
	SPH_C64(0x24ABFFDBFFFF54E3), SPH_C64(0x8FF77AF57A7A8DF4),
	SPH_C64(0xEAF4907A9090643D), SPH_C64(0x3EC25F615F5F9DBE),
	SPH_C64(0xA01D208020203D40), SPH_C64(0xD56768BD68680FD0),
	SPH_C64(0x72D01A681A1ACA34), SPH_C64(0x2C19AE82AEAEB741),
	SPH_C64(0x5EC9B4EAB4B47D75), SPH_C64(0x199A544D5454CEA8),
	SPH_C64(0xE5EC937693937F3B), SPH_C64(0xAA0D228822222F44),
	SPH_C64(0xE907648D646463C8), SPH_C64(0x12DBF1E3F1F12AFF),
	SPH_C64(0xA2BF73D17373CCE6), SPH_C64(0x5A90124812128224),
	SPH_C64(0x5D3A401D40407A80), SPH_C64(0x2840082008084810),
	SPH_C64(0xE856C32BC3C3959B), SPH_C64(0x7B33EC97ECECDFC5),
	SPH_C64(0x9096DB4BDBDB4DAB), SPH_C64(0x1F61A1BEA1A1C05F),
	SPH_C64(0x831C8D0E8D8D9107), SPH_C64(0xC9F53DF43D3DC87A),
	SPH_C64(0xF1CC976697975B33), SPH_C64(0x0000000000000000),
	SPH_C64(0xD436CF1BCFCFF983), SPH_C64(0x87452BAC2B2B6E56),
	SPH_C64(0xB39776C57676E1EC), SPH_C64(0xB06482328282E619),
	SPH_C64(0xA9FED67FD6D628B1), SPH_C64(0x77D81B6C1B1BC336),
	SPH_C64(0x5BC1B5EEB5B57477), SPH_C64(0x2911AF86AFAFBE43),
	SPH_C64(0xDF776AB56A6A1DD4), SPH_C64(0x0DBA505D5050EAA0),
	SPH_C64(0x4C1245094545578A), SPH_C64(0x18CBF3EBF3F338FB),
	SPH_C64(0xF09D30C03030AD60), SPH_C64(0x742BEF9BEFEFC4C3),
	SPH_C64(0xC3E53FFC3F3FDA7E), SPH_C64(0x1C9255495555C7AA),
	SPH_C64(0x1079A2B2A2A2DB59), SPH_C64(0x6503EA8FEAEAE9C9),
	SPH_C64(0xEC0F658965656ACA), SPH_C64(0x68B9BAD2BABA0369),
	SPH_C64(0x93652FBC2F2F4A5E), SPH_C64(0xE74EC027C0C08E9D),
	SPH_C64(0x81BEDE5FDEDE60A1), SPH_C64(0x6CE01C701C1CFC38),
	SPH_C64(0x2EBBFDD3FDFD46E7), SPH_C64(0x64524D294D4D1F9A),
	SPH_C64(0xE0E4927292927639), SPH_C64(0xBC8F75C97575FAEA),
	SPH_C64(0x1E3006180606360C), SPH_C64(0x98248A128A8AAE09),
	SPH_C64(0x40F9B2F2B2B24B79), SPH_C64(0x5963E6BFE6E685D1),
	SPH_C64(0x36700E380E0E7E1C), SPH_C64(0x63F81F7C1F1FE73E),
	SPH_C64(0xF7376295626255C4), SPH_C64(0xA3EED477D4D43AB5),
	SPH_C64(0x3229A89AA8A8814D), SPH_C64(0xF4C4966296965231),
	SPH_C64(0x3A9BF9C3F9F962EF), SPH_C64(0xF666C533C5C5A397),
	SPH_C64(0xB13525942525104A), SPH_C64(0x20F259795959ABB2),
	SPH_C64(0xAE54842A8484D015), SPH_C64(0xA7B772D57272C5E4),
	SPH_C64(0xDDD539E43939EC72), SPH_C64(0x615A4C2D4C4C1698),
	SPH_C64(0x3BCA5E655E5E94BC), SPH_C64(0x85E778FD78789FF0),
	SPH_C64(0xD8DD38E03838E570), SPH_C64(0x86148C0A8C8C9805),
	SPH_C64(0xB2C6D163D1D117BF), SPH_C64(0x0B41A5AEA5A5E457),
	SPH_C64(0x4D43E2AFE2E2A1D9), SPH_C64(0xF82F619961614EC2),
	SPH_C64(0x45F1B3F6B3B3427B), SPH_C64(0xA515218421213442),
	SPH_C64(0xD6949C4A9C9C0825), SPH_C64(0x66F01E781E1EEE3C),
	SPH_C64(0x5222431143436186), SPH_C64(0xFC76C73BC7C7B193),
	SPH_C64(0x2BB3FCD7FCFC4FE5), SPH_C64(0x1420041004042408),
	SPH_C64(0x08B251595151E3A2), SPH_C64(0xC7BC995E9999252F),
	SPH_C64(0xC44F6DA96D6D22DA), SPH_C64(0x39680D340D0D651A),
	SPH_C64(0x3583FACFFAFA79E9), SPH_C64(0x84B6DF5BDFDF69A3),
	SPH_C64(0x9BD77EE57E7EA9FC), SPH_C64(0xB43D249024241948),
	SPH_C64(0xD7C53BEC3B3BFE76), SPH_C64(0x3D31AB96ABAB9A4B),
	SPH_C64(0xD13ECE1FCECEF081), SPH_C64(0x5588114411119922),
	SPH_C64(0x890C8F068F8F8303), SPH_C64(0x6B4A4E254E4E049C),
	SPH_C64(0x51D1B7E6B7B76673), SPH_C64(0x600BEB8BEBEBE0CB),
	SPH_C64(0xCCFD3CF03C3CC178), SPH_C64(0xBF7C813E8181FD1F),
	SPH_C64(0xFED4946A94944035), SPH_C64(0x0CEBF7FBF7F71CF3),
	SPH_C64(0x67A1B9DEB9B9186F), SPH_C64(0x5F98134C13138B26),
	SPH_C64(0x9C7D2CB02C2C5158), SPH_C64(0xB8D6D36BD3D305BB),
	SPH_C64(0x5C6BE7BBE7E78CD3), SPH_C64(0xCB576EA56E6E39DC),
	SPH_C64(0xF36EC437C4C4AA95), SPH_C64(0x0F18030C03031B06),
	SPH_C64(0x138A56455656DCAC), SPH_C64(0x491A440D44445E88),
	SPH_C64(0x9EDF7FE17F7FA0FE), SPH_C64(0x3721A99EA9A9884F),
	SPH_C64(0x824D2AA82A2A6754), SPH_C64(0x6DB1BBD6BBBB0A6B),
	SPH_C64(0xE246C123C1C1879F), SPH_C64(0x02A253515353F1A6),
	SPH_C64(0x8BAEDC57DCDC72A5), SPH_C64(0x27580B2C0B0B5316),
	SPH_C64(0xD39C9D4E9D9D0127), SPH_C64(0xC1476CAD6C6C2BD8),
	SPH_C64(0xF59531C43131A462), SPH_C64(0xB98774CD7474F3E8),
	SPH_C64(0x09E3F6FFF6F615F1), SPH_C64(0x430A460546464C8C),
	SPH_C64(0x2609AC8AACACA545), SPH_C64(0x973C891E8989B50F),
	SPH_C64(0x44A014501414B428), SPH_C64(0x425BE1A3E1E1BADF),
	SPH_C64(0x4EB016581616A62C), SPH_C64(0xD2CD3AE83A3AF774),
	SPH_C64(0xD06F69B9696906D2), SPH_C64(0x2D48092409094112),
	SPH_C64(0xADA770DD7070D7E0), SPH_C64(0x54D9B6E2B6B66F71),
	SPH_C64(0xB7CED067D0D01EBD), SPH_C64(0x7E3BED93EDEDD6C7),
	SPH_C64(0xDB2ECC17CCCCE285), SPH_C64(0x572A421542426884),
	SPH_C64(0xC2B4985A98982C2D), SPH_C64(0x0E49A4AAA4A4ED55),
	SPH_C64(0x885D28A028287550), SPH_C64(0x31DA5C6D5C5C86B8),
	SPH_C64(0x3F93F8C7F8F86BED), SPH_C64(0xA44486228686C211)
};

__constant__ static const sph_u64 plain_T3[256] = {
	SPH_C64(0xC018601818D83078), SPH_C64(0x05238C23232646AF),
	SPH_C64(0x7EC63FC6C6B891F9), SPH_C64(0x13E887E8E8FBCD6F),
	SPH_C64(0x4C87268787CB13A1), SPH_C64(0xA9B8DAB8B8116D62),
	SPH_C64(0x0801040101090205), SPH_C64(0x424F214F4F0D9E6E),
	SPH_C64(0xAD36D836369B6CEE), SPH_C64(0x59A6A2A6A6FF5104),
	SPH_C64(0xDED26FD2D20CB9BD), SPH_C64(0xFBF5F3F5F50EF706),
	SPH_C64(0xEF79F9797996F280), SPH_C64(0x5F6FA16F6F30DECE),
	SPH_C64(0xFC917E91916D3FEF), SPH_C64(0xAA52555252F8A407),
	SPH_C64(0x27609D606047C0FD), SPH_C64(0x89BCCABCBC356576),
	SPH_C64(0xAC9B569B9B372BCD), SPH_C64(0x048E028E8E8A018C),
	SPH_C64(0x71A3B6A3A3D25B15), SPH_C64(0x600C300C0C6C183C),
	SPH_C64(0xFF7BF17B7B84F68A), SPH_C64(0xB535D43535806AE1),
	SPH_C64(0xE81D741D1DF53A69), SPH_C64(0x53E0A7E0E0B3DD47),
	SPH_C64(0xF6D77BD7D721B3AC), SPH_C64(0x5EC22FC2C29C99ED),
	SPH_C64(0x6D2EB82E2E435C96), SPH_C64(0x624B314B4B29967A),
	SPH_C64(0xA3FEDFFEFE5DE121), SPH_C64(0x8257415757D5AE16),
	SPH_C64(0xA815541515BD2A41), SPH_C64(0x9F77C17777E8EEB6),
	SPH_C64(0xA537DC3737926EEB), SPH_C64(0x7BE5B3E5E59ED756),
	SPH_C64(0x8C9F469F9F1323D9), SPH_C64(0xD3F0E7F0F023FD17),
	SPH_C64(0x6A4A354A4A20947F), SPH_C64(0x9EDA4FDADA44A995),
	SPH_C64(0xFA587D5858A2B025), SPH_C64(0x06C903C9C9CF8FCA),
	SPH_C64(0x5529A429297C528D), SPH_C64(0x500A280A0A5A1422),
	SPH_C64(0xE1B1FEB1B1507F4F), SPH_C64(0x69A0BAA0A0C95D1A),
	SPH_C64(0x7F6BB16B6B14D6DA), SPH_C64(0x5C852E8585D917AB),
	SPH_C64(0x81BDCEBDBD3C6773), SPH_C64(0xD25D695D5D8FBA34),
	SPH_C64(0x8010401010902050), SPH_C64(0xF3F4F7F4F407F503),
	SPH_C64(0x16CB0BCBCBDD8BC0), SPH_C64(0xED3EF83E3ED37CC6),
	SPH_C64(0x28051405052D0A11), SPH_C64(0x1F6781676778CEE6),
	SPH_C64(0x73E4B7E4E497D553), SPH_C64(0x25279C2727024EBB),
	SPH_C64(0x3241194141738258), SPH_C64(0x2C8B168B8BA70B9D),
	SPH_C64(0x51A7A6A7A7F65301), SPH_C64(0xCF7DE97D7DB2FA94),
	SPH_C64(0xDC956E95954937FB), SPH_C64(0x8ED847D8D856AD9F),
	SPH_C64(0x8BFBCBFBFB70EB30), SPH_C64(0x23EE9FEEEECDC171),
	SPH_C64(0xC77CED7C7CBBF891), SPH_C64(0x176685666671CCE3),
	SPH_C64(0xA6DD53DDDD7BA78E), SPH_C64(0xB8175C1717AF2E4B),
	SPH_C64(0x0247014747458E46), SPH_C64(0x849E429E9E1A21DC),
	SPH_C64(0x1ECA0FCACAD489C5), SPH_C64(0x752DB42D2D585A99),
	SPH_C64(0x91BFC6BFBF2E6379), SPH_C64(0x38071C07073F0E1B),
	SPH_C64(0x01AD8EADADAC4723), SPH_C64(0xEA5A755A5AB0B42F),
	SPH_C64(0x6C83368383EF1BB5), SPH_C64(0x8533CC3333B666FF),
	SPH_C64(0x3F639163635CC6F2), SPH_C64(0x100208020212040A),
	SPH_C64(0x39AA92AAAA934938), SPH_C64(0xAF71D97171DEE2A8),
	SPH_C64(0x0EC807C8C8C68DCF), SPH_C64(0xC819641919D1327D),
	SPH_C64(0x72493949493B9270), SPH_C64(0x86D943D9D95FAF9A),
	SPH_C64(0xC3F2EFF2F231F91D), SPH_C64(0x4BE3ABE3E3A8DB48),
	SPH_C64(0xE25B715B5BB9B62A), SPH_C64(0x34881A8888BC0D92),
	SPH_C64(0xA49A529A9A3E29C8), SPH_C64(0x2D269826260B4CBE),
	SPH_C64(0x8D32C83232BF64FA), SPH_C64(0xE9B0FAB0B0597D4A),
	SPH_C64(0x1BE983E9E9F2CF6A), SPH_C64(0x780F3C0F0F771E33),
	SPH_C64(0xE6D573D5D533B7A6), SPH_C64(0x74803A8080F41DBA),
	SPH_C64(0x99BEC2BEBE27617C), SPH_C64(0x26CD13CDCDEB87DE),
	SPH_C64(0xBD34D034348968E4), SPH_C64(0x7A483D4848329075),
	SPH_C64(0xABFFDBFFFF54E324), SPH_C64(0xF77AF57A7A8DF48F),
	SPH_C64(0xF4907A9090643DEA), SPH_C64(0xC25F615F5F9DBE3E),
	SPH_C64(0x1D208020203D40A0), SPH_C64(0x6768BD68680FD0D5),
	SPH_C64(0xD01A681A1ACA3472), SPH_C64(0x19AE82AEAEB7412C),
	SPH_C64(0xC9B4EAB4B47D755E), SPH_C64(0x9A544D5454CEA819),
	SPH_C64(0xEC937693937F3BE5), SPH_C64(0x0D228822222F44AA),
	SPH_C64(0x07648D646463C8E9), SPH_C64(0xDBF1E3F1F12AFF12),
	SPH_C64(0xBF73D17373CCE6A2), SPH_C64(0x901248121282245A),
	SPH_C64(0x3A401D40407A805D), SPH_C64(0x4008200808481028),
	SPH_C64(0x56C32BC3C3959BE8), SPH_C64(0x33EC97ECECDFC57B),
	SPH_C64(0x96DB4BDBDB4DAB90), SPH_C64(0x61A1BEA1A1C05F1F),
	SPH_C64(0x1C8D0E8D8D910783), SPH_C64(0xF53DF43D3DC87AC9),
	SPH_C64(0xCC976697975B33F1), SPH_C64(0x0000000000000000),
	SPH_C64(0x36CF1BCFCFF983D4), SPH_C64(0x452BAC2B2B6E5687),
	SPH_C64(0x9776C57676E1ECB3), SPH_C64(0x6482328282E619B0),
	SPH_C64(0xFED67FD6D628B1A9), SPH_C64(0xD81B6C1B1BC33677),
	SPH_C64(0xC1B5EEB5B574775B), SPH_C64(0x11AF86AFAFBE4329),
	SPH_C64(0x776AB56A6A1DD4DF), SPH_C64(0xBA505D5050EAA00D),
	SPH_C64(0x1245094545578A4C), SPH_C64(0xCBF3EBF3F338FB18),
	SPH_C64(0x9D30C03030AD60F0), SPH_C64(0x2BEF9BEFEFC4C374),
	SPH_C64(0xE53FFC3F3FDA7EC3), SPH_C64(0x9255495555C7AA1C),
	SPH_C64(0x79A2B2A2A2DB5910), SPH_C64(0x03EA8FEAEAE9C965),
	SPH_C64(0x0F658965656ACAEC), SPH_C64(0xB9BAD2BABA036968),
	SPH_C64(0x652FBC2F2F4A5E93), SPH_C64(0x4EC027C0C08E9DE7),
	SPH_C64(0xBEDE5FDEDE60A181), SPH_C64(0xE01C701C1CFC386C),
	SPH_C64(0xBBFDD3FDFD46E72E), SPH_C64(0x524D294D4D1F9A64),
	SPH_C64(0xE4927292927639E0), SPH_C64(0x8F75C97575FAEABC),
	SPH_C64(0x3006180606360C1E), SPH_C64(0x248A128A8AAE0998),
	SPH_C64(0xF9B2F2B2B24B7940), SPH_C64(0x63E6BFE6E685D159),
	SPH_C64(0x700E380E0E7E1C36), SPH_C64(0xF81F7C1F1FE73E63),
	SPH_C64(0x376295626255C4F7), SPH_C64(0xEED477D4D43AB5A3),
	SPH_C64(0x29A89AA8A8814D32), SPH_C64(0xC4966296965231F4),
	SPH_C64(0x9BF9C3F9F962EF3A), SPH_C64(0x66C533C5C5A397F6),
	SPH_C64(0x3525942525104AB1), SPH_C64(0xF259795959ABB220),
	SPH_C64(0x54842A8484D015AE), SPH_C64(0xB772D57272C5E4A7),
	SPH_C64(0xD539E43939EC72DD), SPH_C64(0x5A4C2D4C4C169861),
	SPH_C64(0xCA5E655E5E94BC3B), SPH_C64(0xE778FD78789FF085),
	SPH_C64(0xDD38E03838E570D8), SPH_C64(0x148C0A8C8C980586),
	SPH_C64(0xC6D163D1D117BFB2), SPH_C64(0x41A5AEA5A5E4570B),
	SPH_C64(0x43E2AFE2E2A1D94D), SPH_C64(0x2F619961614EC2F8),
	SPH_C64(0xF1B3F6B3B3427B45), SPH_C64(0x15218421213442A5),
	SPH_C64(0x949C4A9C9C0825D6), SPH_C64(0xF01E781E1EEE3C66),
	SPH_C64(0x2243114343618652), SPH_C64(0x76C73BC7C7B193FC),
	SPH_C64(0xB3FCD7FCFC4FE52B), SPH_C64(0x2004100404240814),
	SPH_C64(0xB251595151E3A208), SPH_C64(0xBC995E9999252FC7),
	SPH_C64(0x4F6DA96D6D22DAC4), SPH_C64(0x680D340D0D651A39),
	SPH_C64(0x83FACFFAFA79E935), SPH_C64(0xB6DF5BDFDF69A384),
	SPH_C64(0xD77EE57E7EA9FC9B), SPH_C64(0x3D249024241948B4),
	SPH_C64(0xC53BEC3B3BFE76D7), SPH_C64(0x31AB96ABAB9A4B3D),
	SPH_C64(0x3ECE1FCECEF081D1), SPH_C64(0x8811441111992255),
	SPH_C64(0x0C8F068F8F830389), SPH_C64(0x4A4E254E4E049C6B),
	SPH_C64(0xD1B7E6B7B7667351), SPH_C64(0x0BEB8BEBEBE0CB60),
	SPH_C64(0xFD3CF03C3CC178CC), SPH_C64(0x7C813E8181FD1FBF),
	SPH_C64(0xD4946A94944035FE), SPH_C64(0xEBF7FBF7F71CF30C),
	SPH_C64(0xA1B9DEB9B9186F67), SPH_C64(0x98134C13138B265F),
	SPH_C64(0x7D2CB02C2C51589C), SPH_C64(0xD6D36BD3D305BBB8),
	SPH_C64(0x6BE7BBE7E78CD35C), SPH_C64(0x576EA56E6E39DCCB),
	SPH_C64(0x6EC437C4C4AA95F3), SPH_C64(0x18030C03031B060F),
	SPH_C64(0x8A56455656DCAC13), SPH_C64(0x1A440D44445E8849),
	SPH_C64(0xDF7FE17F7FA0FE9E), SPH_C64(0x21A99EA9A9884F37),
	SPH_C64(0x4D2AA82A2A675482), SPH_C64(0xB1BBD6BBBB0A6B6D),
	SPH_C64(0x46C123C1C1879FE2), SPH_C64(0xA253515353F1A602),
	SPH_C64(0xAEDC57DCDC72A58B), SPH_C64(0x580B2C0B0B531627),
	SPH_C64(0x9C9D4E9D9D0127D3), SPH_C64(0x476CAD6C6C2BD8C1),
	SPH_C64(0x9531C43131A462F5), SPH_C64(0x8774CD7474F3E8B9),
	SPH_C64(0xE3F6FFF6F615F109), SPH_C64(0x0A460546464C8C43),
	SPH_C64(0x09AC8AACACA54526), SPH_C64(0x3C891E8989B50F97),
	SPH_C64(0xA014501414B42844), SPH_C64(0x5BE1A3E1E1BADF42),
	SPH_C64(0xB016581616A62C4E), SPH_C64(0xCD3AE83A3AF774D2),
	SPH_C64(0x6F69B9696906D2D0), SPH_C64(0x480924090941122D),
	SPH_C64(0xA770DD7070D7E0AD), SPH_C64(0xD9B6E2B6B66F7154),
	SPH_C64(0xCED067D0D01EBDB7), SPH_C64(0x3BED93EDEDD6C77E),
	SPH_C64(0x2ECC17CCCCE285DB), SPH_C64(0x2A42154242688457),
	SPH_C64(0xB4985A98982C2DC2), SPH_C64(0x49A4AAA4A4ED550E),
	SPH_C64(0x5D28A02828755088), SPH_C64(0xDA5C6D5C5C86B831),
	SPH_C64(0x93F8C7F8F86BED3F), SPH_C64(0x4486228686C211A4)
};

__constant__ static const sph_u64 plain_T4[256] = {
	SPH_C64(0x18601818D83078C0), SPH_C64(0x238C23232646AF05),
	SPH_C64(0xC63FC6C6B891F97E), SPH_C64(0xE887E8E8FBCD6F13),
	SPH_C64(0x87268787CB13A14C), SPH_C64(0xB8DAB8B8116D62A9),
	SPH_C64(0x0104010109020508), SPH_C64(0x4F214F4F0D9E6E42),
	SPH_C64(0x36D836369B6CEEAD), SPH_C64(0xA6A2A6A6FF510459),
	SPH_C64(0xD26FD2D20CB9BDDE), SPH_C64(0xF5F3F5F50EF706FB),
	SPH_C64(0x79F9797996F280EF), SPH_C64(0x6FA16F6F30DECE5F),
	SPH_C64(0x917E91916D3FEFFC), SPH_C64(0x52555252F8A407AA),
	SPH_C64(0x609D606047C0FD27), SPH_C64(0xBCCABCBC35657689),
	SPH_C64(0x9B569B9B372BCDAC), SPH_C64(0x8E028E8E8A018C04),
	SPH_C64(0xA3B6A3A3D25B1571), SPH_C64(0x0C300C0C6C183C60),
	SPH_C64(0x7BF17B7B84F68AFF), SPH_C64(0x35D43535806AE1B5),
	SPH_C64(0x1D741D1DF53A69E8), SPH_C64(0xE0A7E0E0B3DD4753),
	SPH_C64(0xD77BD7D721B3ACF6), SPH_C64(0xC22FC2C29C99ED5E),
	SPH_C64(0x2EB82E2E435C966D), SPH_C64(0x4B314B4B29967A62),
	SPH_C64(0xFEDFFEFE5DE121A3), SPH_C64(0x57415757D5AE1682),
	SPH_C64(0x15541515BD2A41A8), SPH_C64(0x77C17777E8EEB69F),
	SPH_C64(0x37DC3737926EEBA5), SPH_C64(0xE5B3E5E59ED7567B),
	SPH_C64(0x9F469F9F1323D98C), SPH_C64(0xF0E7F0F023FD17D3),
	SPH_C64(0x4A354A4A20947F6A), SPH_C64(0xDA4FDADA44A9959E),
	SPH_C64(0x587D5858A2B025FA), SPH_C64(0xC903C9C9CF8FCA06),
	SPH_C64(0x29A429297C528D55), SPH_C64(0x0A280A0A5A142250),
	SPH_C64(0xB1FEB1B1507F4FE1), SPH_C64(0xA0BAA0A0C95D1A69),
	SPH_C64(0x6BB16B6B14D6DA7F), SPH_C64(0x852E8585D917AB5C),
	SPH_C64(0xBDCEBDBD3C677381), SPH_C64(0x5D695D5D8FBA34D2),
	SPH_C64(0x1040101090205080), SPH_C64(0xF4F7F4F407F503F3),
	SPH_C64(0xCB0BCBCBDD8BC016), SPH_C64(0x3EF83E3ED37CC6ED),
	SPH_C64(0x051405052D0A1128), SPH_C64(0x6781676778CEE61F),
	SPH_C64(0xE4B7E4E497D55373), SPH_C64(0x279C2727024EBB25),
	SPH_C64(0x4119414173825832), SPH_C64(0x8B168B8BA70B9D2C),
	SPH_C64(0xA7A6A7A7F6530151), SPH_C64(0x7DE97D7DB2FA94CF),
	SPH_C64(0x956E95954937FBDC), SPH_C64(0xD847D8D856AD9F8E),
	SPH_C64(0xFBCBFBFB70EB308B), SPH_C64(0xEE9FEEEECDC17123),
	SPH_C64(0x7CED7C7CBBF891C7), SPH_C64(0x6685666671CCE317),
	SPH_C64(0xDD53DDDD7BA78EA6), SPH_C64(0x175C1717AF2E4BB8),
	SPH_C64(0x47014747458E4602), SPH_C64(0x9E429E9E1A21DC84),
	SPH_C64(0xCA0FCACAD489C51E), SPH_C64(0x2DB42D2D585A9975),
	SPH_C64(0xBFC6BFBF2E637991), SPH_C64(0x071C07073F0E1B38),
	SPH_C64(0xAD8EADADAC472301), SPH_C64(0x5A755A5AB0B42FEA),
	SPH_C64(0x83368383EF1BB56C), SPH_C64(0x33CC3333B666FF85),
	SPH_C64(0x639163635CC6F23F), SPH_C64(0x0208020212040A10),
	SPH_C64(0xAA92AAAA93493839), SPH_C64(0x71D97171DEE2A8AF),
	SPH_C64(0xC807C8C8C68DCF0E), SPH_C64(0x19641919D1327DC8),
	SPH_C64(0x493949493B927072), SPH_C64(0xD943D9D95FAF9A86),
	SPH_C64(0xF2EFF2F231F91DC3), SPH_C64(0xE3ABE3E3A8DB484B),
	SPH_C64(0x5B715B5BB9B62AE2), SPH_C64(0x881A8888BC0D9234),
	SPH_C64(0x9A529A9A3E29C8A4), SPH_C64(0x269826260B4CBE2D),
	SPH_C64(0x32C83232BF64FA8D), SPH_C64(0xB0FAB0B0597D4AE9),
	SPH_C64(0xE983E9E9F2CF6A1B), SPH_C64(0x0F3C0F0F771E3378),
	SPH_C64(0xD573D5D533B7A6E6), SPH_C64(0x803A8080F41DBA74),
	SPH_C64(0xBEC2BEBE27617C99), SPH_C64(0xCD13CDCDEB87DE26),
	SPH_C64(0x34D034348968E4BD), SPH_C64(0x483D48483290757A),
	SPH_C64(0xFFDBFFFF54E324AB), SPH_C64(0x7AF57A7A8DF48FF7),
	SPH_C64(0x907A9090643DEAF4), SPH_C64(0x5F615F5F9DBE3EC2),
	SPH_C64(0x208020203D40A01D), SPH_C64(0x68BD68680FD0D567),
	SPH_C64(0x1A681A1ACA3472D0), SPH_C64(0xAE82AEAEB7412C19),
	SPH_C64(0xB4EAB4B47D755EC9), SPH_C64(0x544D5454CEA8199A),
	SPH_C64(0x937693937F3BE5EC), SPH_C64(0x228822222F44AA0D),
	SPH_C64(0x648D646463C8E907), SPH_C64(0xF1E3F1F12AFF12DB),
	SPH_C64(0x73D17373CCE6A2BF), SPH_C64(0x1248121282245A90),
	SPH_C64(0x401D40407A805D3A), SPH_C64(0x0820080848102840),
	SPH_C64(0xC32BC3C3959BE856), SPH_C64(0xEC97ECECDFC57B33),
	SPH_C64(0xDB4BDBDB4DAB9096), SPH_C64(0xA1BEA1A1C05F1F61),
	SPH_C64(0x8D0E8D8D9107831C), SPH_C64(0x3DF43D3DC87AC9F5),
	SPH_C64(0x976697975B33F1CC), SPH_C64(0x0000000000000000),
	SPH_C64(0xCF1BCFCFF983D436), SPH_C64(0x2BAC2B2B6E568745),
	SPH_C64(0x76C57676E1ECB397), SPH_C64(0x82328282E619B064),
	SPH_C64(0xD67FD6D628B1A9FE), SPH_C64(0x1B6C1B1BC33677D8),
	SPH_C64(0xB5EEB5B574775BC1), SPH_C64(0xAF86AFAFBE432911),
	SPH_C64(0x6AB56A6A1DD4DF77), SPH_C64(0x505D5050EAA00DBA),
	SPH_C64(0x45094545578A4C12), SPH_C64(0xF3EBF3F338FB18CB),
	SPH_C64(0x30C03030AD60F09D), SPH_C64(0xEF9BEFEFC4C3742B),
	SPH_C64(0x3FFC3F3FDA7EC3E5), SPH_C64(0x55495555C7AA1C92),
	SPH_C64(0xA2B2A2A2DB591079), SPH_C64(0xEA8FEAEAE9C96503),
	SPH_C64(0x658965656ACAEC0F), SPH_C64(0xBAD2BABA036968B9),
	SPH_C64(0x2FBC2F2F4A5E9365), SPH_C64(0xC027C0C08E9DE74E),
	SPH_C64(0xDE5FDEDE60A181BE), SPH_C64(0x1C701C1CFC386CE0),
	SPH_C64(0xFDD3FDFD46E72EBB), SPH_C64(0x4D294D4D1F9A6452),
	SPH_C64(0x927292927639E0E4), SPH_C64(0x75C97575FAEABC8F),
	SPH_C64(0x06180606360C1E30), SPH_C64(0x8A128A8AAE099824),
	SPH_C64(0xB2F2B2B24B7940F9), SPH_C64(0xE6BFE6E685D15963),
	SPH_C64(0x0E380E0E7E1C3670), SPH_C64(0x1F7C1F1FE73E63F8),
	SPH_C64(0x6295626255C4F737), SPH_C64(0xD477D4D43AB5A3EE),
	SPH_C64(0xA89AA8A8814D3229), SPH_C64(0x966296965231F4C4),
	SPH_C64(0xF9C3F9F962EF3A9B), SPH_C64(0xC533C5C5A397F666),
	SPH_C64(0x25942525104AB135), SPH_C64(0x59795959ABB220F2),
	SPH_C64(0x842A8484D015AE54), SPH_C64(0x72D57272C5E4A7B7),
	SPH_C64(0x39E43939EC72DDD5), SPH_C64(0x4C2D4C4C1698615A),
	SPH_C64(0x5E655E5E94BC3BCA), SPH_C64(0x78FD78789FF085E7),
	SPH_C64(0x38E03838E570D8DD), SPH_C64(0x8C0A8C8C98058614),
	SPH_C64(0xD163D1D117BFB2C6), SPH_C64(0xA5AEA5A5E4570B41),
	SPH_C64(0xE2AFE2E2A1D94D43), SPH_C64(0x619961614EC2F82F),
	SPH_C64(0xB3F6B3B3427B45F1), SPH_C64(0x218421213442A515),
	SPH_C64(0x9C4A9C9C0825D694), SPH_C64(0x1E781E1EEE3C66F0),
	SPH_C64(0x4311434361865222), SPH_C64(0xC73BC7C7B193FC76),
	SPH_C64(0xFCD7FCFC4FE52BB3), SPH_C64(0x0410040424081420),
	SPH_C64(0x51595151E3A208B2), SPH_C64(0x995E9999252FC7BC),
	SPH_C64(0x6DA96D6D22DAC44F), SPH_C64(0x0D340D0D651A3968),
	SPH_C64(0xFACFFAFA79E93583), SPH_C64(0xDF5BDFDF69A384B6),
	SPH_C64(0x7EE57E7EA9FC9BD7), SPH_C64(0x249024241948B43D),
	SPH_C64(0x3BEC3B3BFE76D7C5), SPH_C64(0xAB96ABAB9A4B3D31),
	SPH_C64(0xCE1FCECEF081D13E), SPH_C64(0x1144111199225588),
	SPH_C64(0x8F068F8F8303890C), SPH_C64(0x4E254E4E049C6B4A),
	SPH_C64(0xB7E6B7B7667351D1), SPH_C64(0xEB8BEBEBE0CB600B),
	SPH_C64(0x3CF03C3CC178CCFD), SPH_C64(0x813E8181FD1FBF7C),
	SPH_C64(0x946A94944035FED4), SPH_C64(0xF7FBF7F71CF30CEB),
	SPH_C64(0xB9DEB9B9186F67A1), SPH_C64(0x134C13138B265F98),
	SPH_C64(0x2CB02C2C51589C7D), SPH_C64(0xD36BD3D305BBB8D6),
	SPH_C64(0xE7BBE7E78CD35C6B), SPH_C64(0x6EA56E6E39DCCB57),
	SPH_C64(0xC437C4C4AA95F36E), SPH_C64(0x030C03031B060F18),
	SPH_C64(0x56455656DCAC138A), SPH_C64(0x440D44445E88491A),
	SPH_C64(0x7FE17F7FA0FE9EDF), SPH_C64(0xA99EA9A9884F3721),
	SPH_C64(0x2AA82A2A6754824D), SPH_C64(0xBBD6BBBB0A6B6DB1),
	SPH_C64(0xC123C1C1879FE246), SPH_C64(0x53515353F1A602A2),
	SPH_C64(0xDC57DCDC72A58BAE), SPH_C64(0x0B2C0B0B53162758),
	SPH_C64(0x9D4E9D9D0127D39C), SPH_C64(0x6CAD6C6C2BD8C147),
	SPH_C64(0x31C43131A462F595), SPH_C64(0x74CD7474F3E8B987),
	SPH_C64(0xF6FFF6F615F109E3), SPH_C64(0x460546464C8C430A),
	SPH_C64(0xAC8AACACA5452609), SPH_C64(0x891E8989B50F973C),
	SPH_C64(0x14501414B42844A0), SPH_C64(0xE1A3E1E1BADF425B),
	SPH_C64(0x16581616A62C4EB0), SPH_C64(0x3AE83A3AF774D2CD),
	SPH_C64(0x69B9696906D2D06F), SPH_C64(0x0924090941122D48),
	SPH_C64(0x70DD7070D7E0ADA7), SPH_C64(0xB6E2B6B66F7154D9),
	SPH_C64(0xD067D0D01EBDB7CE), SPH_C64(0xED93EDEDD6C77E3B),
	SPH_C64(0xCC17CCCCE285DB2E), SPH_C64(0x421542426884572A),
	SPH_C64(0x985A98982C2DC2B4), SPH_C64(0xA4AAA4A4ED550E49),
	SPH_C64(0x28A028287550885D), SPH_C64(0x5C6D5C5C86B831DA),
	SPH_C64(0xF8C7F8F86BED3F93), SPH_C64(0x86228686C211A444)
};

__constant__ static const sph_u64 plain_T5[256] = {
	SPH_C64(0x601818D83078C018), SPH_C64(0x8C23232646AF0523),
	SPH_C64(0x3FC6C6B891F97EC6), SPH_C64(0x87E8E8FBCD6F13E8),
	SPH_C64(0x268787CB13A14C87), SPH_C64(0xDAB8B8116D62A9B8),
	SPH_C64(0x0401010902050801), SPH_C64(0x214F4F0D9E6E424F),
	SPH_C64(0xD836369B6CEEAD36), SPH_C64(0xA2A6A6FF510459A6),
	SPH_C64(0x6FD2D20CB9BDDED2), SPH_C64(0xF3F5F50EF706FBF5),
	SPH_C64(0xF9797996F280EF79), SPH_C64(0xA16F6F30DECE5F6F),
	SPH_C64(0x7E91916D3FEFFC91), SPH_C64(0x555252F8A407AA52),
	SPH_C64(0x9D606047C0FD2760), SPH_C64(0xCABCBC35657689BC),
	SPH_C64(0x569B9B372BCDAC9B), SPH_C64(0x028E8E8A018C048E),
	SPH_C64(0xB6A3A3D25B1571A3), SPH_C64(0x300C0C6C183C600C),
	SPH_C64(0xF17B7B84F68AFF7B), SPH_C64(0xD43535806AE1B535),
	SPH_C64(0x741D1DF53A69E81D), SPH_C64(0xA7E0E0B3DD4753E0),
	SPH_C64(0x7BD7D721B3ACF6D7), SPH_C64(0x2FC2C29C99ED5EC2),
	SPH_C64(0xB82E2E435C966D2E), SPH_C64(0x314B4B29967A624B),
	SPH_C64(0xDFFEFE5DE121A3FE), SPH_C64(0x415757D5AE168257),
	SPH_C64(0x541515BD2A41A815), SPH_C64(0xC17777E8EEB69F77),
	SPH_C64(0xDC3737926EEBA537), SPH_C64(0xB3E5E59ED7567BE5),
	SPH_C64(0x469F9F1323D98C9F), SPH_C64(0xE7F0F023FD17D3F0),
	SPH_C64(0x354A4A20947F6A4A), SPH_C64(0x4FDADA44A9959EDA),
	SPH_C64(0x7D5858A2B025FA58), SPH_C64(0x03C9C9CF8FCA06C9),
	SPH_C64(0xA429297C528D5529), SPH_C64(0x280A0A5A1422500A),
	SPH_C64(0xFEB1B1507F4FE1B1), SPH_C64(0xBAA0A0C95D1A69A0),
	SPH_C64(0xB16B6B14D6DA7F6B), SPH_C64(0x2E8585D917AB5C85),
	SPH_C64(0xCEBDBD3C677381BD), SPH_C64(0x695D5D8FBA34D25D),
	SPH_C64(0x4010109020508010), SPH_C64(0xF7F4F407F503F3F4),
	SPH_C64(0x0BCBCBDD8BC016CB), SPH_C64(0xF83E3ED37CC6ED3E),
	SPH_C64(0x1405052D0A112805), SPH_C64(0x81676778CEE61F67),
	SPH_C64(0xB7E4E497D55373E4), SPH_C64(0x9C2727024EBB2527),
	SPH_C64(0x1941417382583241), SPH_C64(0x168B8BA70B9D2C8B),
	SPH_C64(0xA6A7A7F6530151A7), SPH_C64(0xE97D7DB2FA94CF7D),
	SPH_C64(0x6E95954937FBDC95), SPH_C64(0x47D8D856AD9F8ED8),
	SPH_C64(0xCBFBFB70EB308BFB), SPH_C64(0x9FEEEECDC17123EE),
	SPH_C64(0xED7C7CBBF891C77C), SPH_C64(0x85666671CCE31766),
	SPH_C64(0x53DDDD7BA78EA6DD), SPH_C64(0x5C1717AF2E4BB817),
	SPH_C64(0x014747458E460247), SPH_C64(0x429E9E1A21DC849E),
	SPH_C64(0x0FCACAD489C51ECA), SPH_C64(0xB42D2D585A99752D),
	SPH_C64(0xC6BFBF2E637991BF), SPH_C64(0x1C07073F0E1B3807),
	SPH_C64(0x8EADADAC472301AD), SPH_C64(0x755A5AB0B42FEA5A),
	SPH_C64(0x368383EF1BB56C83), SPH_C64(0xCC3333B666FF8533),
	SPH_C64(0x9163635CC6F23F63), SPH_C64(0x08020212040A1002),
	SPH_C64(0x92AAAA93493839AA), SPH_C64(0xD97171DEE2A8AF71),
	SPH_C64(0x07C8C8C68DCF0EC8), SPH_C64(0x641919D1327DC819),
	SPH_C64(0x3949493B92707249), SPH_C64(0x43D9D95FAF9A86D9),
	SPH_C64(0xEFF2F231F91DC3F2), SPH_C64(0xABE3E3A8DB484BE3),
	SPH_C64(0x715B5BB9B62AE25B), SPH_C64(0x1A8888BC0D923488),
	SPH_C64(0x529A9A3E29C8A49A), SPH_C64(0x9826260B4CBE2D26),
	SPH_C64(0xC83232BF64FA8D32), SPH_C64(0xFAB0B0597D4AE9B0),
	SPH_C64(0x83E9E9F2CF6A1BE9), SPH_C64(0x3C0F0F771E33780F),
	SPH_C64(0x73D5D533B7A6E6D5), SPH_C64(0x3A8080F41DBA7480),
	SPH_C64(0xC2BEBE27617C99BE), SPH_C64(0x13CDCDEB87DE26CD),
	SPH_C64(0xD034348968E4BD34), SPH_C64(0x3D48483290757A48),
	SPH_C64(0xDBFFFF54E324ABFF), SPH_C64(0xF57A7A8DF48FF77A),
	SPH_C64(0x7A9090643DEAF490), SPH_C64(0x615F5F9DBE3EC25F),
	SPH_C64(0x8020203D40A01D20), SPH_C64(0xBD68680FD0D56768),
	SPH_C64(0x681A1ACA3472D01A), SPH_C64(0x82AEAEB7412C19AE),
	SPH_C64(0xEAB4B47D755EC9B4), SPH_C64(0x4D5454CEA8199A54),
	SPH_C64(0x7693937F3BE5EC93), SPH_C64(0x8822222F44AA0D22),
	SPH_C64(0x8D646463C8E90764), SPH_C64(0xE3F1F12AFF12DBF1),
	SPH_C64(0xD17373CCE6A2BF73), SPH_C64(0x48121282245A9012),
	SPH_C64(0x1D40407A805D3A40), SPH_C64(0x2008084810284008),
	SPH_C64(0x2BC3C3959BE856C3), SPH_C64(0x97ECECDFC57B33EC),
	SPH_C64(0x4BDBDB4DAB9096DB), SPH_C64(0xBEA1A1C05F1F61A1),
	SPH_C64(0x0E8D8D9107831C8D), SPH_C64(0xF43D3DC87AC9F53D),
	SPH_C64(0x6697975B33F1CC97), SPH_C64(0x0000000000000000),
	SPH_C64(0x1BCFCFF983D436CF), SPH_C64(0xAC2B2B6E5687452B),
	SPH_C64(0xC57676E1ECB39776), SPH_C64(0x328282E619B06482),
	SPH_C64(0x7FD6D628B1A9FED6), SPH_C64(0x6C1B1BC33677D81B),
	SPH_C64(0xEEB5B574775BC1B5), SPH_C64(0x86AFAFBE432911AF),
	SPH_C64(0xB56A6A1DD4DF776A), SPH_C64(0x5D5050EAA00DBA50),
	SPH_C64(0x094545578A4C1245), SPH_C64(0xEBF3F338FB18CBF3),
	SPH_C64(0xC03030AD60F09D30), SPH_C64(0x9BEFEFC4C3742BEF),
	SPH_C64(0xFC3F3FDA7EC3E53F), SPH_C64(0x495555C7AA1C9255),
	SPH_C64(0xB2A2A2DB591079A2), SPH_C64(0x8FEAEAE9C96503EA),
	SPH_C64(0x8965656ACAEC0F65), SPH_C64(0xD2BABA036968B9BA),
	SPH_C64(0xBC2F2F4A5E93652F), SPH_C64(0x27C0C08E9DE74EC0),
	SPH_C64(0x5FDEDE60A181BEDE), SPH_C64(0x701C1CFC386CE01C),
	SPH_C64(0xD3FDFD46E72EBBFD), SPH_C64(0x294D4D1F9A64524D),
	SPH_C64(0x7292927639E0E492), SPH_C64(0xC97575FAEABC8F75),
	SPH_C64(0x180606360C1E3006), SPH_C64(0x128A8AAE0998248A),
	SPH_C64(0xF2B2B24B7940F9B2), SPH_C64(0xBFE6E685D15963E6),
	SPH_C64(0x380E0E7E1C36700E), SPH_C64(0x7C1F1FE73E63F81F),
	SPH_C64(0x95626255C4F73762), SPH_C64(0x77D4D43AB5A3EED4),
	SPH_C64(0x9AA8A8814D3229A8), SPH_C64(0x6296965231F4C496),
	SPH_C64(0xC3F9F962EF3A9BF9), SPH_C64(0x33C5C5A397F666C5),
	SPH_C64(0x942525104AB13525), SPH_C64(0x795959ABB220F259),
	SPH_C64(0x2A8484D015AE5484), SPH_C64(0xD57272C5E4A7B772),
	SPH_C64(0xE43939EC72DDD539), SPH_C64(0x2D4C4C1698615A4C),
	SPH_C64(0x655E5E94BC3BCA5E), SPH_C64(0xFD78789FF085E778),
	SPH_C64(0xE03838E570D8DD38), SPH_C64(0x0A8C8C980586148C),
	SPH_C64(0x63D1D117BFB2C6D1), SPH_C64(0xAEA5A5E4570B41A5),
	SPH_C64(0xAFE2E2A1D94D43E2), SPH_C64(0x9961614EC2F82F61),
	SPH_C64(0xF6B3B3427B45F1B3), SPH_C64(0x8421213442A51521),
	SPH_C64(0x4A9C9C0825D6949C), SPH_C64(0x781E1EEE3C66F01E),
	SPH_C64(0x1143436186522243), SPH_C64(0x3BC7C7B193FC76C7),
	SPH_C64(0xD7FCFC4FE52BB3FC), SPH_C64(0x1004042408142004),
	SPH_C64(0x595151E3A208B251), SPH_C64(0x5E9999252FC7BC99),
	SPH_C64(0xA96D6D22DAC44F6D), SPH_C64(0x340D0D651A39680D),
	SPH_C64(0xCFFAFA79E93583FA), SPH_C64(0x5BDFDF69A384B6DF),
	SPH_C64(0xE57E7EA9FC9BD77E), SPH_C64(0x9024241948B43D24),
	SPH_C64(0xEC3B3BFE76D7C53B), SPH_C64(0x96ABAB9A4B3D31AB),
	SPH_C64(0x1FCECEF081D13ECE), SPH_C64(0x4411119922558811),
	SPH_C64(0x068F8F8303890C8F), SPH_C64(0x254E4E049C6B4A4E),
	SPH_C64(0xE6B7B7667351D1B7), SPH_C64(0x8BEBEBE0CB600BEB),
	SPH_C64(0xF03C3CC178CCFD3C), SPH_C64(0x3E8181FD1FBF7C81),
	SPH_C64(0x6A94944035FED494), SPH_C64(0xFBF7F71CF30CEBF7),
	SPH_C64(0xDEB9B9186F67A1B9), SPH_C64(0x4C13138B265F9813),
	SPH_C64(0xB02C2C51589C7D2C), SPH_C64(0x6BD3D305BBB8D6D3),
	SPH_C64(0xBBE7E78CD35C6BE7), SPH_C64(0xA56E6E39DCCB576E),
	SPH_C64(0x37C4C4AA95F36EC4), SPH_C64(0x0C03031B060F1803),
	SPH_C64(0x455656DCAC138A56), SPH_C64(0x0D44445E88491A44),
	SPH_C64(0xE17F7FA0FE9EDF7F), SPH_C64(0x9EA9A9884F3721A9),
	SPH_C64(0xA82A2A6754824D2A), SPH_C64(0xD6BBBB0A6B6DB1BB),
	SPH_C64(0x23C1C1879FE246C1), SPH_C64(0x515353F1A602A253),
	SPH_C64(0x57DCDC72A58BAEDC), SPH_C64(0x2C0B0B531627580B),
	SPH_C64(0x4E9D9D0127D39C9D), SPH_C64(0xAD6C6C2BD8C1476C),
	SPH_C64(0xC43131A462F59531), SPH_C64(0xCD7474F3E8B98774),
	SPH_C64(0xFFF6F615F109E3F6), SPH_C64(0x0546464C8C430A46),
	SPH_C64(0x8AACACA5452609AC), SPH_C64(0x1E8989B50F973C89),
	SPH_C64(0x501414B42844A014), SPH_C64(0xA3E1E1BADF425BE1),
	SPH_C64(0x581616A62C4EB016), SPH_C64(0xE83A3AF774D2CD3A),
	SPH_C64(0xB9696906D2D06F69), SPH_C64(0x24090941122D4809),
	SPH_C64(0xDD7070D7E0ADA770), SPH_C64(0xE2B6B66F7154D9B6),
	SPH_C64(0x67D0D01EBDB7CED0), SPH_C64(0x93EDEDD6C77E3BED),
	SPH_C64(0x17CCCCE285DB2ECC), SPH_C64(0x1542426884572A42),
	SPH_C64(0x5A98982C2DC2B498), SPH_C64(0xAAA4A4ED550E49A4),
	SPH_C64(0xA028287550885D28), SPH_C64(0x6D5C5C86B831DA5C),
	SPH_C64(0xC7F8F86BED3F93F8), SPH_C64(0x228686C211A44486)
};

__constant__ static const sph_u64 plain_T6[256] = {
	SPH_C64(0x1818D83078C01860), SPH_C64(0x23232646AF05238C),
	SPH_C64(0xC6C6B891F97EC63F), SPH_C64(0xE8E8FBCD6F13E887),
	SPH_C64(0x8787CB13A14C8726), SPH_C64(0xB8B8116D62A9B8DA),
	SPH_C64(0x0101090205080104), SPH_C64(0x4F4F0D9E6E424F21),
	SPH_C64(0x36369B6CEEAD36D8), SPH_C64(0xA6A6FF510459A6A2),
	SPH_C64(0xD2D20CB9BDDED26F), SPH_C64(0xF5F50EF706FBF5F3),
	SPH_C64(0x797996F280EF79F9), SPH_C64(0x6F6F30DECE5F6FA1),
	SPH_C64(0x91916D3FEFFC917E), SPH_C64(0x5252F8A407AA5255),
	SPH_C64(0x606047C0FD27609D), SPH_C64(0xBCBC35657689BCCA),
	SPH_C64(0x9B9B372BCDAC9B56), SPH_C64(0x8E8E8A018C048E02),
	SPH_C64(0xA3A3D25B1571A3B6), SPH_C64(0x0C0C6C183C600C30),
	SPH_C64(0x7B7B84F68AFF7BF1), SPH_C64(0x3535806AE1B535D4),
	SPH_C64(0x1D1DF53A69E81D74), SPH_C64(0xE0E0B3DD4753E0A7),
	SPH_C64(0xD7D721B3ACF6D77B), SPH_C64(0xC2C29C99ED5EC22F),
	SPH_C64(0x2E2E435C966D2EB8), SPH_C64(0x4B4B29967A624B31),
	SPH_C64(0xFEFE5DE121A3FEDF), SPH_C64(0x5757D5AE16825741),
	SPH_C64(0x1515BD2A41A81554), SPH_C64(0x7777E8EEB69F77C1),
	SPH_C64(0x3737926EEBA537DC), SPH_C64(0xE5E59ED7567BE5B3),
	SPH_C64(0x9F9F1323D98C9F46), SPH_C64(0xF0F023FD17D3F0E7),
	SPH_C64(0x4A4A20947F6A4A35), SPH_C64(0xDADA44A9959EDA4F),
	SPH_C64(0x5858A2B025FA587D), SPH_C64(0xC9C9CF8FCA06C903),
	SPH_C64(0x29297C528D5529A4), SPH_C64(0x0A0A5A1422500A28),
	SPH_C64(0xB1B1507F4FE1B1FE), SPH_C64(0xA0A0C95D1A69A0BA),
	SPH_C64(0x6B6B14D6DA7F6BB1), SPH_C64(0x8585D917AB5C852E),
	SPH_C64(0xBDBD3C677381BDCE), SPH_C64(0x5D5D8FBA34D25D69),
	SPH_C64(0x1010902050801040), SPH_C64(0xF4F407F503F3F4F7),
	SPH_C64(0xCBCBDD8BC016CB0B), SPH_C64(0x3E3ED37CC6ED3EF8),
	SPH_C64(0x05052D0A11280514), SPH_C64(0x676778CEE61F6781),
	SPH_C64(0xE4E497D55373E4B7), SPH_C64(0x2727024EBB25279C),
	SPH_C64(0x4141738258324119), SPH_C64(0x8B8BA70B9D2C8B16),
	SPH_C64(0xA7A7F6530151A7A6), SPH_C64(0x7D7DB2FA94CF7DE9),
	SPH_C64(0x95954937FBDC956E), SPH_C64(0xD8D856AD9F8ED847),
	SPH_C64(0xFBFB70EB308BFBCB), SPH_C64(0xEEEECDC17123EE9F),
	SPH_C64(0x7C7CBBF891C77CED), SPH_C64(0x666671CCE3176685),
	SPH_C64(0xDDDD7BA78EA6DD53), SPH_C64(0x1717AF2E4BB8175C),
	SPH_C64(0x4747458E46024701), SPH_C64(0x9E9E1A21DC849E42),
	SPH_C64(0xCACAD489C51ECA0F), SPH_C64(0x2D2D585A99752DB4),
	SPH_C64(0xBFBF2E637991BFC6), SPH_C64(0x07073F0E1B38071C),
	SPH_C64(0xADADAC472301AD8E), SPH_C64(0x5A5AB0B42FEA5A75),
	SPH_C64(0x8383EF1BB56C8336), SPH_C64(0x3333B666FF8533CC),
	SPH_C64(0x63635CC6F23F6391), SPH_C64(0x020212040A100208),
	SPH_C64(0xAAAA93493839AA92), SPH_C64(0x7171DEE2A8AF71D9),
	SPH_C64(0xC8C8C68DCF0EC807), SPH_C64(0x1919D1327DC81964),
	SPH_C64(0x49493B9270724939), SPH_C64(0xD9D95FAF9A86D943),
	SPH_C64(0xF2F231F91DC3F2EF), SPH_C64(0xE3E3A8DB484BE3AB),
	SPH_C64(0x5B5BB9B62AE25B71), SPH_C64(0x8888BC0D9234881A),
	SPH_C64(0x9A9A3E29C8A49A52), SPH_C64(0x26260B4CBE2D2698),
	SPH_C64(0x3232BF64FA8D32C8), SPH_C64(0xB0B0597D4AE9B0FA),
	SPH_C64(0xE9E9F2CF6A1BE983), SPH_C64(0x0F0F771E33780F3C),
	SPH_C64(0xD5D533B7A6E6D573), SPH_C64(0x8080F41DBA74803A),
	SPH_C64(0xBEBE27617C99BEC2), SPH_C64(0xCDCDEB87DE26CD13),
	SPH_C64(0x34348968E4BD34D0), SPH_C64(0x48483290757A483D),
	SPH_C64(0xFFFF54E324ABFFDB), SPH_C64(0x7A7A8DF48FF77AF5),
	SPH_C64(0x9090643DEAF4907A), SPH_C64(0x5F5F9DBE3EC25F61),
	SPH_C64(0x20203D40A01D2080), SPH_C64(0x68680FD0D56768BD),
	SPH_C64(0x1A1ACA3472D01A68), SPH_C64(0xAEAEB7412C19AE82),
	SPH_C64(0xB4B47D755EC9B4EA), SPH_C64(0x5454CEA8199A544D),
	SPH_C64(0x93937F3BE5EC9376), SPH_C64(0x22222F44AA0D2288),
	SPH_C64(0x646463C8E907648D), SPH_C64(0xF1F12AFF12DBF1E3),
	SPH_C64(0x7373CCE6A2BF73D1), SPH_C64(0x121282245A901248),
	SPH_C64(0x40407A805D3A401D), SPH_C64(0x0808481028400820),
	SPH_C64(0xC3C3959BE856C32B), SPH_C64(0xECECDFC57B33EC97),
	SPH_C64(0xDBDB4DAB9096DB4B), SPH_C64(0xA1A1C05F1F61A1BE),
	SPH_C64(0x8D8D9107831C8D0E), SPH_C64(0x3D3DC87AC9F53DF4),
	SPH_C64(0x97975B33F1CC9766), SPH_C64(0x0000000000000000),
	SPH_C64(0xCFCFF983D436CF1B), SPH_C64(0x2B2B6E5687452BAC),
	SPH_C64(0x7676E1ECB39776C5), SPH_C64(0x8282E619B0648232),
	SPH_C64(0xD6D628B1A9FED67F), SPH_C64(0x1B1BC33677D81B6C),
	SPH_C64(0xB5B574775BC1B5EE), SPH_C64(0xAFAFBE432911AF86),
	SPH_C64(0x6A6A1DD4DF776AB5), SPH_C64(0x5050EAA00DBA505D),
	SPH_C64(0x4545578A4C124509), SPH_C64(0xF3F338FB18CBF3EB),
	SPH_C64(0x3030AD60F09D30C0), SPH_C64(0xEFEFC4C3742BEF9B),
	SPH_C64(0x3F3FDA7EC3E53FFC), SPH_C64(0x5555C7AA1C925549),
	SPH_C64(0xA2A2DB591079A2B2), SPH_C64(0xEAEAE9C96503EA8F),
	SPH_C64(0x65656ACAEC0F6589), SPH_C64(0xBABA036968B9BAD2),
	SPH_C64(0x2F2F4A5E93652FBC), SPH_C64(0xC0C08E9DE74EC027),
	SPH_C64(0xDEDE60A181BEDE5F), SPH_C64(0x1C1CFC386CE01C70),
	SPH_C64(0xFDFD46E72EBBFDD3), SPH_C64(0x4D4D1F9A64524D29),
	SPH_C64(0x92927639E0E49272), SPH_C64(0x7575FAEABC8F75C9),
	SPH_C64(0x0606360C1E300618), SPH_C64(0x8A8AAE0998248A12),
	SPH_C64(0xB2B24B7940F9B2F2), SPH_C64(0xE6E685D15963E6BF),
	SPH_C64(0x0E0E7E1C36700E38), SPH_C64(0x1F1FE73E63F81F7C),
	SPH_C64(0x626255C4F7376295), SPH_C64(0xD4D43AB5A3EED477),
	SPH_C64(0xA8A8814D3229A89A), SPH_C64(0x96965231F4C49662),
	SPH_C64(0xF9F962EF3A9BF9C3), SPH_C64(0xC5C5A397F666C533),
	SPH_C64(0x2525104AB1352594), SPH_C64(0x5959ABB220F25979),
	SPH_C64(0x8484D015AE54842A), SPH_C64(0x7272C5E4A7B772D5),
	SPH_C64(0x3939EC72DDD539E4), SPH_C64(0x4C4C1698615A4C2D),
	SPH_C64(0x5E5E94BC3BCA5E65), SPH_C64(0x78789FF085E778FD),
	SPH_C64(0x3838E570D8DD38E0), SPH_C64(0x8C8C980586148C0A),
	SPH_C64(0xD1D117BFB2C6D163), SPH_C64(0xA5A5E4570B41A5AE),
	SPH_C64(0xE2E2A1D94D43E2AF), SPH_C64(0x61614EC2F82F6199),
	SPH_C64(0xB3B3427B45F1B3F6), SPH_C64(0x21213442A5152184),
	SPH_C64(0x9C9C0825D6949C4A), SPH_C64(0x1E1EEE3C66F01E78),
	SPH_C64(0x4343618652224311), SPH_C64(0xC7C7B193FC76C73B),
	SPH_C64(0xFCFC4FE52BB3FCD7), SPH_C64(0x0404240814200410),
	SPH_C64(0x5151E3A208B25159), SPH_C64(0x9999252FC7BC995E),
	SPH_C64(0x6D6D22DAC44F6DA9), SPH_C64(0x0D0D651A39680D34),
	SPH_C64(0xFAFA79E93583FACF), SPH_C64(0xDFDF69A384B6DF5B),
	SPH_C64(0x7E7EA9FC9BD77EE5), SPH_C64(0x24241948B43D2490),
	SPH_C64(0x3B3BFE76D7C53BEC), SPH_C64(0xABAB9A4B3D31AB96),
	SPH_C64(0xCECEF081D13ECE1F), SPH_C64(0x1111992255881144),
	SPH_C64(0x8F8F8303890C8F06), SPH_C64(0x4E4E049C6B4A4E25),
	SPH_C64(0xB7B7667351D1B7E6), SPH_C64(0xEBEBE0CB600BEB8B),
	SPH_C64(0x3C3CC178CCFD3CF0), SPH_C64(0x8181FD1FBF7C813E),
	SPH_C64(0x94944035FED4946A), SPH_C64(0xF7F71CF30CEBF7FB),
	SPH_C64(0xB9B9186F67A1B9DE), SPH_C64(0x13138B265F98134C),
	SPH_C64(0x2C2C51589C7D2CB0), SPH_C64(0xD3D305BBB8D6D36B),
	SPH_C64(0xE7E78CD35C6BE7BB), SPH_C64(0x6E6E39DCCB576EA5),
	SPH_C64(0xC4C4AA95F36EC437), SPH_C64(0x03031B060F18030C),
	SPH_C64(0x5656DCAC138A5645), SPH_C64(0x44445E88491A440D),
	SPH_C64(0x7F7FA0FE9EDF7FE1), SPH_C64(0xA9A9884F3721A99E),
	SPH_C64(0x2A2A6754824D2AA8), SPH_C64(0xBBBB0A6B6DB1BBD6),
	SPH_C64(0xC1C1879FE246C123), SPH_C64(0x5353F1A602A25351),
	SPH_C64(0xDCDC72A58BAEDC57), SPH_C64(0x0B0B531627580B2C),
	SPH_C64(0x9D9D0127D39C9D4E), SPH_C64(0x6C6C2BD8C1476CAD),
	SPH_C64(0x3131A462F59531C4), SPH_C64(0x7474F3E8B98774CD),
	SPH_C64(0xF6F615F109E3F6FF), SPH_C64(0x46464C8C430A4605),
	SPH_C64(0xACACA5452609AC8A), SPH_C64(0x8989B50F973C891E),
	SPH_C64(0x1414B42844A01450), SPH_C64(0xE1E1BADF425BE1A3),
	SPH_C64(0x1616A62C4EB01658), SPH_C64(0x3A3AF774D2CD3AE8),
	SPH_C64(0x696906D2D06F69B9), SPH_C64(0x090941122D480924),
	SPH_C64(0x7070D7E0ADA770DD), SPH_C64(0xB6B66F7154D9B6E2),
	SPH_C64(0xD0D01EBDB7CED067), SPH_C64(0xEDEDD6C77E3BED93),
	SPH_C64(0xCCCCE285DB2ECC17), SPH_C64(0x42426884572A4215),
	SPH_C64(0x98982C2DC2B4985A), SPH_C64(0xA4A4ED550E49A4AA),
	SPH_C64(0x28287550885D28A0), SPH_C64(0x5C5C86B831DA5C6D),
	SPH_C64(0xF8F86BED3F93F8C7), SPH_C64(0x8686C211A4448622)
};

__constant__ static const sph_u64 plain_T7[256] = {
	SPH_C64(0x18D83078C0186018), SPH_C64(0x232646AF05238C23),
	SPH_C64(0xC6B891F97EC63FC6), SPH_C64(0xE8FBCD6F13E887E8),
	SPH_C64(0x87CB13A14C872687), SPH_C64(0xB8116D62A9B8DAB8),
	SPH_C64(0x0109020508010401), SPH_C64(0x4F0D9E6E424F214F),
	SPH_C64(0x369B6CEEAD36D836), SPH_C64(0xA6FF510459A6A2A6),
	SPH_C64(0xD20CB9BDDED26FD2), SPH_C64(0xF50EF706FBF5F3F5),
	SPH_C64(0x7996F280EF79F979), SPH_C64(0x6F30DECE5F6FA16F),
	SPH_C64(0x916D3FEFFC917E91), SPH_C64(0x52F8A407AA525552),
	SPH_C64(0x6047C0FD27609D60), SPH_C64(0xBC35657689BCCABC),
	SPH_C64(0x9B372BCDAC9B569B), SPH_C64(0x8E8A018C048E028E),
	SPH_C64(0xA3D25B1571A3B6A3), SPH_C64(0x0C6C183C600C300C),
	SPH_C64(0x7B84F68AFF7BF17B), SPH_C64(0x35806AE1B535D435),
	SPH_C64(0x1DF53A69E81D741D), SPH_C64(0xE0B3DD4753E0A7E0),
	SPH_C64(0xD721B3ACF6D77BD7), SPH_C64(0xC29C99ED5EC22FC2),
	SPH_C64(0x2E435C966D2EB82E), SPH_C64(0x4B29967A624B314B),
	SPH_C64(0xFE5DE121A3FEDFFE), SPH_C64(0x57D5AE1682574157),
	SPH_C64(0x15BD2A41A8155415), SPH_C64(0x77E8EEB69F77C177),
	SPH_C64(0x37926EEBA537DC37), SPH_C64(0xE59ED7567BE5B3E5),
	SPH_C64(0x9F1323D98C9F469F), SPH_C64(0xF023FD17D3F0E7F0),
	SPH_C64(0x4A20947F6A4A354A), SPH_C64(0xDA44A9959EDA4FDA),
	SPH_C64(0x58A2B025FA587D58), SPH_C64(0xC9CF8FCA06C903C9),
	SPH_C64(0x297C528D5529A429), SPH_C64(0x0A5A1422500A280A),
	SPH_C64(0xB1507F4FE1B1FEB1), SPH_C64(0xA0C95D1A69A0BAA0),
	SPH_C64(0x6B14D6DA7F6BB16B), SPH_C64(0x85D917AB5C852E85),
	SPH_C64(0xBD3C677381BDCEBD), SPH_C64(0x5D8FBA34D25D695D),
	SPH_C64(0x1090205080104010), SPH_C64(0xF407F503F3F4F7F4),
	SPH_C64(0xCBDD8BC016CB0BCB), SPH_C64(0x3ED37CC6ED3EF83E),
	SPH_C64(0x052D0A1128051405), SPH_C64(0x6778CEE61F678167),
	SPH_C64(0xE497D55373E4B7E4), SPH_C64(0x27024EBB25279C27),
	SPH_C64(0x4173825832411941), SPH_C64(0x8BA70B9D2C8B168B),
	SPH_C64(0xA7F6530151A7A6A7), SPH_C64(0x7DB2FA94CF7DE97D),
	SPH_C64(0x954937FBDC956E95), SPH_C64(0xD856AD9F8ED847D8),
	SPH_C64(0xFB70EB308BFBCBFB), SPH_C64(0xEECDC17123EE9FEE),
	SPH_C64(0x7CBBF891C77CED7C), SPH_C64(0x6671CCE317668566),
	SPH_C64(0xDD7BA78EA6DD53DD), SPH_C64(0x17AF2E4BB8175C17),
	SPH_C64(0x47458E4602470147), SPH_C64(0x9E1A21DC849E429E),
	SPH_C64(0xCAD489C51ECA0FCA), SPH_C64(0x2D585A99752DB42D),
	SPH_C64(0xBF2E637991BFC6BF), SPH_C64(0x073F0E1B38071C07),
	SPH_C64(0xADAC472301AD8EAD), SPH_C64(0x5AB0B42FEA5A755A),
	SPH_C64(0x83EF1BB56C833683), SPH_C64(0x33B666FF8533CC33),
	SPH_C64(0x635CC6F23F639163), SPH_C64(0x0212040A10020802),
	SPH_C64(0xAA93493839AA92AA), SPH_C64(0x71DEE2A8AF71D971),
	SPH_C64(0xC8C68DCF0EC807C8), SPH_C64(0x19D1327DC8196419),
	SPH_C64(0x493B927072493949), SPH_C64(0xD95FAF9A86D943D9),
	SPH_C64(0xF231F91DC3F2EFF2), SPH_C64(0xE3A8DB484BE3ABE3),
	SPH_C64(0x5BB9B62AE25B715B), SPH_C64(0x88BC0D9234881A88),
	SPH_C64(0x9A3E29C8A49A529A), SPH_C64(0x260B4CBE2D269826),
	SPH_C64(0x32BF64FA8D32C832), SPH_C64(0xB0597D4AE9B0FAB0),
	SPH_C64(0xE9F2CF6A1BE983E9), SPH_C64(0x0F771E33780F3C0F),
	SPH_C64(0xD533B7A6E6D573D5), SPH_C64(0x80F41DBA74803A80),
	SPH_C64(0xBE27617C99BEC2BE), SPH_C64(0xCDEB87DE26CD13CD),
	SPH_C64(0x348968E4BD34D034), SPH_C64(0x483290757A483D48),
	SPH_C64(0xFF54E324ABFFDBFF), SPH_C64(0x7A8DF48FF77AF57A),
	SPH_C64(0x90643DEAF4907A90), SPH_C64(0x5F9DBE3EC25F615F),
	SPH_C64(0x203D40A01D208020), SPH_C64(0x680FD0D56768BD68),
	SPH_C64(0x1ACA3472D01A681A), SPH_C64(0xAEB7412C19AE82AE),
	SPH_C64(0xB47D755EC9B4EAB4), SPH_C64(0x54CEA8199A544D54),
	SPH_C64(0x937F3BE5EC937693), SPH_C64(0x222F44AA0D228822),
	SPH_C64(0x6463C8E907648D64), SPH_C64(0xF12AFF12DBF1E3F1),
	SPH_C64(0x73CCE6A2BF73D173), SPH_C64(0x1282245A90124812),
	SPH_C64(0x407A805D3A401D40), SPH_C64(0x0848102840082008),
	SPH_C64(0xC3959BE856C32BC3), SPH_C64(0xECDFC57B33EC97EC),
	SPH_C64(0xDB4DAB9096DB4BDB), SPH_C64(0xA1C05F1F61A1BEA1),
	SPH_C64(0x8D9107831C8D0E8D), SPH_C64(0x3DC87AC9F53DF43D),
	SPH_C64(0x975B33F1CC976697), SPH_C64(0x0000000000000000),
	SPH_C64(0xCFF983D436CF1BCF), SPH_C64(0x2B6E5687452BAC2B),
	SPH_C64(0x76E1ECB39776C576), SPH_C64(0x82E619B064823282),
	SPH_C64(0xD628B1A9FED67FD6), SPH_C64(0x1BC33677D81B6C1B),
	SPH_C64(0xB574775BC1B5EEB5), SPH_C64(0xAFBE432911AF86AF),
	SPH_C64(0x6A1DD4DF776AB56A), SPH_C64(0x50EAA00DBA505D50),
	SPH_C64(0x45578A4C12450945), SPH_C64(0xF338FB18CBF3EBF3),
	SPH_C64(0x30AD60F09D30C030), SPH_C64(0xEFC4C3742BEF9BEF),
	SPH_C64(0x3FDA7EC3E53FFC3F), SPH_C64(0x55C7AA1C92554955),
	SPH_C64(0xA2DB591079A2B2A2), SPH_C64(0xEAE9C96503EA8FEA),
	SPH_C64(0x656ACAEC0F658965), SPH_C64(0xBA036968B9BAD2BA),
	SPH_C64(0x2F4A5E93652FBC2F), SPH_C64(0xC08E9DE74EC027C0),
	SPH_C64(0xDE60A181BEDE5FDE), SPH_C64(0x1CFC386CE01C701C),
	SPH_C64(0xFD46E72EBBFDD3FD), SPH_C64(0x4D1F9A64524D294D),
	SPH_C64(0x927639E0E4927292), SPH_C64(0x75FAEABC8F75C975),
	SPH_C64(0x06360C1E30061806), SPH_C64(0x8AAE0998248A128A),
	SPH_C64(0xB24B7940F9B2F2B2), SPH_C64(0xE685D15963E6BFE6),
	SPH_C64(0x0E7E1C36700E380E), SPH_C64(0x1FE73E63F81F7C1F),
	SPH_C64(0x6255C4F737629562), SPH_C64(0xD43AB5A3EED477D4),
	SPH_C64(0xA8814D3229A89AA8), SPH_C64(0x965231F4C4966296),
	SPH_C64(0xF962EF3A9BF9C3F9), SPH_C64(0xC5A397F666C533C5),
	SPH_C64(0x25104AB135259425), SPH_C64(0x59ABB220F2597959),
	SPH_C64(0x84D015AE54842A84), SPH_C64(0x72C5E4A7B772D572),
	SPH_C64(0x39EC72DDD539E439), SPH_C64(0x4C1698615A4C2D4C),
	SPH_C64(0x5E94BC3BCA5E655E), SPH_C64(0x789FF085E778FD78),
	SPH_C64(0x38E570D8DD38E038), SPH_C64(0x8C980586148C0A8C),
	SPH_C64(0xD117BFB2C6D163D1), SPH_C64(0xA5E4570B41A5AEA5),
	SPH_C64(0xE2A1D94D43E2AFE2), SPH_C64(0x614EC2F82F619961),
	SPH_C64(0xB3427B45F1B3F6B3), SPH_C64(0x213442A515218421),
	SPH_C64(0x9C0825D6949C4A9C), SPH_C64(0x1EEE3C66F01E781E),
	SPH_C64(0x4361865222431143), SPH_C64(0xC7B193FC76C73BC7),
	SPH_C64(0xFC4FE52BB3FCD7FC), SPH_C64(0x0424081420041004),
	SPH_C64(0x51E3A208B2515951), SPH_C64(0x99252FC7BC995E99),
	SPH_C64(0x6D22DAC44F6DA96D), SPH_C64(0x0D651A39680D340D),
	SPH_C64(0xFA79E93583FACFFA), SPH_C64(0xDF69A384B6DF5BDF),
	SPH_C64(0x7EA9FC9BD77EE57E), SPH_C64(0x241948B43D249024),
	SPH_C64(0x3BFE76D7C53BEC3B), SPH_C64(0xAB9A4B3D31AB96AB),
	SPH_C64(0xCEF081D13ECE1FCE), SPH_C64(0x1199225588114411),
	SPH_C64(0x8F8303890C8F068F), SPH_C64(0x4E049C6B4A4E254E),
	SPH_C64(0xB7667351D1B7E6B7), SPH_C64(0xEBE0CB600BEB8BEB),
	SPH_C64(0x3CC178CCFD3CF03C), SPH_C64(0x81FD1FBF7C813E81),
	SPH_C64(0x944035FED4946A94), SPH_C64(0xF71CF30CEBF7FBF7),
	SPH_C64(0xB9186F67A1B9DEB9), SPH_C64(0x138B265F98134C13),
	SPH_C64(0x2C51589C7D2CB02C), SPH_C64(0xD305BBB8D6D36BD3),
	SPH_C64(0xE78CD35C6BE7BBE7), SPH_C64(0x6E39DCCB576EA56E),
	SPH_C64(0xC4AA95F36EC437C4), SPH_C64(0x031B060F18030C03),
	SPH_C64(0x56DCAC138A564556), SPH_C64(0x445E88491A440D44),
	SPH_C64(0x7FA0FE9EDF7FE17F), SPH_C64(0xA9884F3721A99EA9),
	SPH_C64(0x2A6754824D2AA82A), SPH_C64(0xBB0A6B6DB1BBD6BB),
	SPH_C64(0xC1879FE246C123C1), SPH_C64(0x53F1A602A2535153),
	SPH_C64(0xDC72A58BAEDC57DC), SPH_C64(0x0B531627580B2C0B),
	SPH_C64(0x9D0127D39C9D4E9D), SPH_C64(0x6C2BD8C1476CAD6C),
	SPH_C64(0x31A462F59531C431), SPH_C64(0x74F3E8B98774CD74),
	SPH_C64(0xF615F109E3F6FFF6), SPH_C64(0x464C8C430A460546),
	SPH_C64(0xACA5452609AC8AAC), SPH_C64(0x89B50F973C891E89),
	SPH_C64(0x14B42844A0145014), SPH_C64(0xE1BADF425BE1A3E1),
	SPH_C64(0x16A62C4EB0165816), SPH_C64(0x3AF774D2CD3AE83A),
	SPH_C64(0x6906D2D06F69B969), SPH_C64(0x0941122D48092409),
	SPH_C64(0x70D7E0ADA770DD70), SPH_C64(0xB66F7154D9B6E2B6),
	SPH_C64(0xD01EBDB7CED067D0), SPH_C64(0xEDD6C77E3BED93ED),
	SPH_C64(0xCCE285DB2ECC17CC), SPH_C64(0x426884572A421542),
	SPH_C64(0x982C2DC2B4985A98), SPH_C64(0xA4ED550E49A4AAA4),
	SPH_C64(0x287550885D28A028), SPH_C64(0x5C86B831DA5C6D5C),
	SPH_C64(0xF86BED3F93F8C7F8), SPH_C64(0x86C211A444862286)
};


/*
 * Round constants.
 */
__constant__ static const sph_u64 plain_RC[10] = {
	SPH_C64(0x4F01B887E8C62318),
	SPH_C64(0x52916F79F5D2A636),
	SPH_C64(0x357B0CA38E9BBC60),
	SPH_C64(0x57FE4B2EC2D7E01D),
	SPH_C64(0xDA4AF09FE5377715),
	SPH_C64(0x856BA0B10A29C958),
	SPH_C64(0x67053ECBF4105DBD),
	SPH_C64(0xD8957DA78B4127E4),
	SPH_C64(0x9E4717DD667CEEFB),
	SPH_C64(0x33835AAD07BF2DCA)
};

/* ====================================================================== */

//#define BYTE(x, y)	(amd_bfe((uint)((x) >> ((y >= 32U) ? 32U : 0U)), (y) - (((y) >= 32) ? 32U : 0), 8U))
//#define BYTE(x,y)  ((y < 32U) ? (__byte_perm(_LODWORD((x)), 0, 0x4440 + ((y) / 8))) : (__byte_perm(_HIDWORD((x)), 0, 0x4440 + (((y)-32) / 8))))

#define BYTE(x, y) ((uint8_t)(((x) >> (y)) & 0xff))

#define ROUND_ELT(table, in, i0, i1, i2, i3, i4, i5, i6, i7) \
        xor8(table ## 0[BYTE(in ## i0, 0U)] \
        , table ## 1[BYTE(in ## i1, 8U)] \
        , table ## 2[BYTE(in ## i2, 16U)] \
        , table ## 3[BYTE(in ## i3, 24U)] \
        , table ## 4[BYTE(in ## i4, 32U)] \
        , table ## 5[BYTE(in ## i5, 40U)] \
        , __ldg(&plain_T6[BYTE(in ## i6, 48U)]) \
        , __ldg(&plain_T7[BYTE(in ## i7, 56U)]))


#define ROUND_ELT1(table, in, i0, i1, i2, i3, i4, i5, i6, i7) \
        xor8(table ## 0[BYTE(in ## i0, 0U)] \
        , table ## 1[BYTE(in ## i1, 8U)] \
        , __ldg(&plain_T2[BYTE(in ## i2, 16U)]) \
        , table ## 3[BYTE(in ## i3, 24U)] \
        , table ## 4[BYTE(in ## i4, 32U)] \
        , __ldg(&plain_T5[BYTE(in ## i5, 40U)]) \
        , __ldg(&plain_T6[BYTE(in ## i6, 48U)]) \
        , __ldg(&plain_T7[BYTE(in ## i7, 56U)]))

/*
#define ROUND_ELT(table, in, i0, i1, i2, i3, i4, i5, i6, i7, c) \
        xor9(table ## 0[BYTE(in ## i0, 0U)] \
        , table ## 1[BYTE(in ## i1, 8U)] \
        , table ## 2[BYTE(in ## i2, 16U)] \
        , table ## 3[BYTE(in ## i3, 24U)] \
        , table ## 4[BYTE(in ## i4, 32U)] \
        , table ## 5[BYTE(in ## i5, 40U)] \
        , __ldg(&plain_T6[BYTE(in ## i6, 48U)]) \
        , __ldg(&plain_T7[BYTE(in ## i7, 56U)]), c)


#define ROUND_ELT1(table, in, i0, i1, i2, i3, i4, i5, i6, i7, c) \
        xor9(table ## 0[BYTE(in ## i0, 0U)] \
        , table ## 1[BYTE(in ## i1, 8U)] \
        , __ldg(&plain_T2[BYTE(in ## i2, 16U)]) \
        , table ## 3[BYTE(in ## i3, 24U)] \
        , table ## 4[BYTE(in ## i4, 32U)] \
        , __ldg(&plain_T5[BYTE(in ## i5, 40U)]) \
        , __ldg(&plain_T6[BYTE(in ## i6, 48U)]) \
        , __ldg(&plain_T7[BYTE(in ## i7, 56U)]), c)

*/
#define ROUND(table, in, out, c0, c1, c2, c3, c4, c5, c6, c7)   do { \
		out ## 0 = xor1(ROUND_ELT1(table, in, 0, 7, 6, 5, 4, 3, 2, 1) , c0); \
		out ## 1 = xor1(ROUND_ELT(table, in, 1, 0, 7, 6, 5, 4, 3, 2) , c1); \
		out ## 2 = xor1(ROUND_ELT(table, in, 2, 1, 0, 7, 6, 5, 4, 3) , c2); \
		out ## 3 = xor1(ROUND_ELT1(table, in, 3, 2, 1, 0, 7, 6, 5, 4) , c3); \
		out ## 4 = xor1(ROUND_ELT1(table, in, 4, 3, 2, 1, 0, 7, 6, 5) , c4); \
		out ## 5 = xor1(ROUND_ELT(table, in, 5, 4, 3, 2, 1, 0, 7, 6) , c5); \
		out ## 6 = xor1(ROUND_ELT1(table, in, 6, 5, 4, 3, 2, 1, 0, 7) , c6); \
		out ## 7 = xor1(ROUND_ELT1(table, in, 7, 6, 5, 4, 3, 2, 1, 0) , c7); \
	} while (0)

#define ROUND5(table, in, out, c0, c1, c2, c3, c4, c5, c6, c7)   do { \
                out ## 0 = xor1(ROUND_ELT1(table, in, 0, 7, 6, 5, 4, 3, 2, 1) , c0); \
                out ## 1 = ROUND_ELT(table, in, 1, 0, 7, 6, 5, 4, 3, 2); \
                out ## 2 = ROUND_ELT(table, in, 2, 1, 0, 7, 6, 5, 4, 3); \
                out ## 3 = ROUND_ELT1(table, in, 3, 2, 1, 0, 7, 6, 5, 4); \
                out ## 4 = ROUND_ELT1(table, in, 4, 3, 2, 1, 0, 7, 6, 5); \
                out ## 5 = ROUND_ELT(table, in, 5, 4, 3, 2, 1, 0, 7, 6) ; \
                out ## 6 = ROUND_ELT1(table, in, 6, 5, 4, 3, 2, 1, 0, 7); \
                out ## 7 = ROUND_ELT1(table, in, 7, 6, 5, 4, 3, 2, 1, 0); \
        } while (0)

/*
#define ROUND(table, in, out, c0, c1, c2, c3, c4, c5, c6, c7)   do { \
                out ## 0 = ROUND_ELT1(table, in, 0, 7, 6, 5, 4, 3, 2, 1) ^ c0; \
                out ## 1 = ROUND_ELT(table, in, 1, 0, 7, 6, 5, 4, 3, 2) ^ c1; \
                out ## 2 = ROUND_ELT(table, in, 2, 1, 0, 7, 6, 5, 4, 3) ^ c2; \
                out ## 3 = ROUND_ELT1(table, in, 3, 2, 1, 0, 7, 6, 5, 4) ^ c3; \
                out ## 4 = ROUND_ELT1(table, in, 4, 3, 2, 1, 0, 7, 6, 5) ^ c4; \
                out ## 5 = ROUND_ELT(table, in, 5, 4, 3, 2, 1, 0, 7, 6) ^ c5; \
                out ## 6 = ROUND_ELT1(table, in, 6, 5, 4, 3, 2, 1, 0, 7) ^ c6; \
                out ## 7 = ROUND_ELT1(table, in, 7, 6, 5, 4, 3, 2, 1, 0) ^ c7; \
        } while (0)
*/

/*
#define ROUND(table, in, out, c0, c1, c2, c3, c4, c5, c6, c7)   do { \
                out ## 0 = ROUND_ELT1(table, in, 0, 7, 6, 5, 4, 3, 2, 1, c0); \
                out ## 1 = ROUND_ELT(table, in, 1, 0, 7, 6, 5, 4, 3, 2, c1); \
                out ## 2 = ROUND_ELT(table, in, 2, 1, 0, 7, 6, 5, 4, 3, c2); \
                out ## 3 = ROUND_ELT1(table, in, 3, 2, 1, 0, 7, 6, 5, 4, c3); \
                out ## 4 = ROUND_ELT1(table, in, 4, 3, 2, 1, 0, 7, 6, 5, c4); \
                out ## 5 = ROUND_ELT(table, in, 5, 4, 3, 2, 1, 0, 7, 6, c5); \
                out ## 6 = ROUND_ELT1(table, in, 6, 5, 4, 3, 2, 1, 0, 7, c6); \
                out ## 7 = ROUND_ELT1(table, in, 7, 6, 5, 4, 3, 2, 1, 0, c7); \
        } while (0)

*/

#define ROUND_KSCHED(table, in, out, c) \
	ROUND(table, in, out, c, 0, 0, 0, 0, 0, 0, 0)

#define ROUND_WENC(table, in, key, out) \
	ROUND(table, in, out, key ## 0, key ## 1, key ## 2, \
		key ## 3, key ## 4, key ## 5, key ## 6, key ## 7)

#define TRANSFER(dst, src)   do { \
		dst ## 0 = src ## 0; \
		dst ## 1 = src ## 1; \
		dst ## 2 = src ## 2; \
		dst ## 3 = src ## 3; \
		dst ## 4 = src ## 4; \
		dst ## 5 = src ## 5; \
		dst ## 6 = src ## 6; \
		dst ## 7 = src ## 7; \
	} while (0)


#define TPB_W 256
__global__ __launch_bounds__(TPB_W,2)
void xevan_whirlpool_gpu_hash_64(uint32_t threads, uint32_t *g_hash){

        const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

 __shared__ sph_u64 LT0[256], LT1[256], LT2[256], LT3[256], LT4[256], LT5[256];//, LT6[256], LT7[256];

        if (thread < threads)
        {
                uint32_t *Hash = &g_hash[thread<<4];
                uint64_t hx[8];
  // whirlpool

if(threadIdx.x < 256)
{
	uint64_t temp = plain_T0[threadIdx.x];
    LT0[threadIdx.x] = temp;
    LT1[threadIdx.x] = ROTL64(temp,8);
    LT2[threadIdx.x] = ROTL64(temp,16);
    LT3[threadIdx.x] = ROTL64(temp,24);
    LT4[threadIdx.x] = SWAPDWORDS(temp);;
    LT5[threadIdx.x] = ROTR64(temp,24);
//    LT6[threadIdx.x] = plain_T6[threadIdx.x];
//    LT7[threadIdx.x] = plain_T7[threadIdx.x];
  }
                *(uint2x4*)&hx[ 0] = __ldg4((uint2x4*)&Hash[0]);
                *(uint2x4*)&hx[ 4] = __ldg4((uint2x4*)&Hash[8]);
//__syncthreads();
	
  sph_u64 n0, n1, n2, n3, n4, n5, n6, n7;
  sph_u64 h0, h1, h2, h3, h4, h5, h6, h7;
  sph_u64 state[8];

  n0 = (hx[0]);
  n1 = (hx[1]);
  n2 = (hx[2]);
  n3 = (hx[3]);
  n4 = (hx[4]);
  n5 = (hx[5]);
  n6 = (hx[6]);
  n7 = (hx[7]);

  h0 = h1 = h2 = h3 = h4 = h5 = h6 = h7 = 0;

  n0 ^= h0;
  n1 ^= h1;
  n2 ^= h2;
  n3 ^= h3;
  n4 ^= h4;
  n5 ^= h5;
  n6 ^= h6;
  n7 ^= h7;
__syncthreads();
  for (unsigned r = 0; r < 10; r ++)
  {
    sph_u64 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

    ROUND_KSCHED(LT, h, tmp, plain_RC[r]);
    TRANSFER(h, tmp);
    ROUND_WENC(LT, n, h, tmp);
    TRANSFER(n, tmp);
  }

  state[0] = n0 ^ (hx[0]);
  state[1] = n1 ^ (hx[1]);
  state[2] = n2 ^ (hx[2]);
  state[3] = n3 ^ (hx[3]);
  state[4] = n4 ^ (hx[4]);
  state[5] = n5 ^ (hx[5]);
  state[6] = n6 ^ (hx[6]);
  state[7] = n7 ^ (hx[7]);

  n0 = n1 = n2 = n3 = n4 = n5 = n6 = n7 = 0;

  h0 = state[0];
  h1 = state[1];
  h2 = state[2];
  h3 = state[3];
  h4 = state[4];
  h5 = state[5];
  h6 = state[6];
  h7 = state[7];

  n0 ^= h0;
  n1 ^= h1;
  n2 ^= h2;
  n3 ^= h3;
  n4 ^= h4;
  n5 ^= h5;
  n6 ^= h6;
  n7 ^= h7;

//#pragma unroll 10
  for (unsigned r = 0; r < 10; r++)
  {
    sph_u64 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

    ROUND_KSCHED(LT, h, tmp, plain_RC[r]);
    TRANSFER(h, tmp);
    ROUND_WENC(LT, n, h, tmp);
    TRANSFER(n, tmp);
  }

  state[0] = n0 ^ state[0];
  state[1] = n1 ^ state[1];
  state[2] = n2 ^ state[2];
  state[3] = n3 ^ state[3];
  state[4] = n4 ^ state[4];
  state[5] = n5 ^ state[5];
  state[6] = n6 ^ state[6];
  state[7] = n7 ^ state[7];

  n0 = 0x80;
  n1 = n2 = n3 = n4 = n5 = n6 = 0;
  n7 = 0x4000000000000;

  h0 = state[0];
  h1 = state[1];
  h2 = state[2];
  h3 = state[3];
  h4 = state[4];
  h5 = state[5];
  h6 = state[6];
  h7 = state[7];

  n0 ^= h0;
  n1 ^= h1;
  n2 ^= h2;
  n3 ^= h3;
  n4 ^= h4;
  n5 ^= h5;
  n6 ^= h6;
  n7 ^= h7;

//  #pragma unroll 10
  for (unsigned r = 0; r < 10; r ++)
  {
    sph_u64 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

    ROUND_KSCHED(LT, h, tmp, plain_RC[r]);
    TRANSFER(h, tmp);
    ROUND_WENC(LT, n, h, tmp);
    TRANSFER(n, tmp);
  }

  state[0] ^= n0 ^ 0x80;
  state[1] ^= n1;
  state[2] ^= n2;
  state[3] ^= n3;
  state[4] ^= n4;
  state[5] ^= n5;
  state[6] ^= n6;
  state[7] ^= n7 ^ 0x4000000000000;
#pragma unroll 8
  for (unsigned i = 0; i < 8; i ++)
    hx[i] = state[i];


                *(uint2x4*)&Hash[0] = *(uint2x4*)&hx[ 0];
                *(uint2x4*)&Hash[8] = *(uint2x4*)&hx[ 4];
	}
}


__host__
void xevan_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
        const uint32_t threadsperblock = TPB_W;

        dim3 grid((threads + threadsperblock-1)/threadsperblock);
        dim3 block(threadsperblock);

        xevan_whirlpool_gpu_hash_64<<<grid, block>>>(threads, d_hash);

}
