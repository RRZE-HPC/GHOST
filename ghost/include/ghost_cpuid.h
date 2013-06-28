#ifndef CPUID_H
#define CPUID_H

#include <stdint.h>

/* Intel P6 */
#define PENTIUM_M_BANIAS     0x09U
#define PENTIUM_M_DOTHAN     0x0DU
#define CORE_DUO             0x0EU
#define CORE2_65             0x0FU
#define CORE2_45             0x17U
#define ATOM                 0x1CU
#define NEHALEM              0x1AU
#define NEHALEM_BLOOMFIELD   0x1AU
#define NEHALEM_LYNNFIELD    0x1EU
#define NEHALEM_LYNNFIELD_M  0x1FU
#define NEHALEM_WESTMERE     0x2CU
#define NEHALEM_WESTMERE_M   0x25U
#define SANDYBRIDGE          0x2AU
#define SANDYBRIDGE_EP       0x2DU
#define IVYBRIDGE            0x3AU
#define NEHALEM_EX           0x2EU
#define WESTMERE_EX          0x2FU
#define XEON_MP              0x1DU

/* Intel MIC */
#define XEON_PHI           0x01U

/* AMD K10 */
#define BARCELONA      0x02U
#define SHANGHAI       0x04U
#define ISTANBUL       0x08U
#define MAGNYCOURS     0x09U

/* AMD K8 */
#define OPTERON_SC_1MB  0x05U
#define OPTERON_DC_E    0x21U
#define OPTERON_DC_F    0x41U
#define ATHLON64_X2     0x43U
#define ATHLON64_X2_F   0x4BU
#define ATHLON64_F1     0x4FU
#define ATHLON64_F2     0x5FU
#define ATHLON64_X2_G   0x6BU
#define ATHLON64_G1     0x6FU
#define ATHLON64_G2     0x7FU


#define  P6_FAMILY        0x6U
#define  MIC_FAMILY       0xBU
#define  NETBURST_FAMILY  0xFFU
#define  K15_FAMILY       0x15U
#define  K10_FAMILY       0x10U
#define  K8_FAMILY        0xFU

typedef enum {
    NOCACHE=0,
    DATACACHE,
    INSTRUCTIONCACHE,
    UNIFIEDCACHE,
    ITLB,
    DTLB} CacheType;

typedef enum {
    SSE=0,
    AVX,
    FMA} featureBit;

typedef enum {
    NODE=0,
    SOCKET,
    CORE,
    THREAD} NodeLevel;

typedef struct {
    uint32_t family;
    uint32_t model;
    uint32_t stepping;
    uint64_t clock;
    int      turbo;
    char*  name;
    char*  features;
    int     featureFlags;
    uint32_t perf_version;
    uint32_t perf_num_ctr;
    uint32_t perf_width_ctr;
    uint32_t perf_num_fixed_ctr;
} CpuInfo;

typedef struct {
    uint32_t threadId;
    uint32_t coreId;
    uint32_t packageId;
    uint32_t apicId;
} HWThread;

typedef struct {
    int level;
    CacheType type;
    int associativity;
    int sets;
    int lineSize;
    int size;
    int threads;
    int inclusive;
} CacheLevel;

typedef struct {
    uint32_t numHWThreads;
    uint32_t numSockets;
    uint32_t numCoresPerSocket;
    uint32_t numThreadsPerCore;
    uint32_t numCacheLevels;
    HWThread* threadPool;
    CacheLevel*  cacheLevels;
//    TreeNode* topologyTree;
} CpuTopology;

/** Structure holding cpuid information
 *
 */
extern CpuInfo ghost_cpuid_info;
extern CpuTopology ghost_cpuid_topology;

/** Init routine to intialize global structure.
 *
 *  Determines: 
 *  - cpu family
 *  - cpu model
 *  - cpu stepping
 *  - cpu clock
 *  - Instruction Set Extension Flags
 *  - Performance counter features (Intel P6 only)
 *
 */
void ghost_cpuid_init (void);
void ghost_cpuid_initTopology (void);
void ghost_cpuid_initCacheTopology (void);
int  ghost_cpuid_isInCpuset(void);

static inline int ghost_cpuid_hasFeature(featureBit bit) 
{
  return (ghost_cpuid_info.featureFlags & (1<<bit));
}

#endif /*CPUID_H*/
