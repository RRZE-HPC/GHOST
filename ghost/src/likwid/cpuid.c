/*
 * =======================================================================================
 *
 *      Filename:  cpuid.c
 *
 *      Description:  Implementation of cpuid module.
 *                  Provides API to extract cpuid info on x86 processors.
 *
 *      Version:   3.0
 *      Released:  29.11.2012
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2012 Jan Treibig 
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */

#define _GNU_SOURCE

/* #####   HEADER FILE INCLUDES   ######################################### */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>

#include <error.h>
#include <cpuid.h>
#include <tree.h>
#include <bitUtil.h>
#include <strUtil.h>
#include <ghost_constants.h>

/* #####   EXPORTED VARIABLES   ########################################### */

CpuInfo ghost_cpuid_info;
CpuTopology ghost_cpuid_topology;


/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int largest_function = 0;
static int likwid_inCPUSet  = 0;

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

/* this was taken from the linux kernel */
#define CPUID                              \
    __asm__ volatile ("cpuid"                             \
            : "=a" (eax),     \
            "=b" (ebx),     \
            "=c" (ecx),     \
            "=d" (edx)      \
            : "0" (eax), "2" (ecx))


/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static char* pentium_m_b_str = "Intel Pentium M Banias processor";
static char* pentium_m_d_str = "Intel Pentium M Dothan processor";
static char* core_duo_str = "Intel Core Duo processor";
static char* core_2a_str = "Intel Core 2 65nm processor";
static char* core_2b_str = "Intel Core 2 45nm processor";
static char* nehalem_bloom_str = "Intel Core Bloomfield processor";
static char* nehalem_lynn_str = "Intel Core Lynnfield processor";
static char* nehalem_west_str = "Intel Core Westmere processor";
static char* sandybridge_str = "Intel Core SandyBridge processor";
static char* ivybridge_str = "Intel Core IvyBridge processor";
static char* sandybridge_ep_str = "Intel Core SandyBridge EP processor";
static char* nehalem_ex_str = "Intel Nehalem EX processor";
static char* westmere_ex_str = "Intel Westmere EX processor";
static char* xeon_mp_string = "Intel Xeon MP processor";
static char* xeon_phi_string = "Intel Xeon Phi Coprocessor";
static char* barcelona_str = "AMD Barcelona processor";
static char* shanghai_str = "AMD Shanghai processor";
static char* istanbul_str = "AMD Istanbul processor";
static char* magnycours_str = "AMD Magny Cours processor";
static char* interlagos_str = "AMD Interlagos processor";
static char* opteron_sc_str = "AMD Opteron single core 130nm processor";
static char* opteron_dc_e_str = "AMD Opteron Dual Core Rev E 90nm processor";
static char* opteron_dc_f_str = "AMD Opteron Dual Core Rev F 90nm processor";
static char* athlon64_str = "AMD Athlon64 X2 (AM2) Rev F 90nm processor";
static char* athlon64_f_str = "AMD Athlon64 (AM2) Rev F 90nm processor";
static char* athlon64_X2_g_str = "AMD Athlon64 X2 (AM2) Rev G 65nm processor";
static char* athlon64_g_str = "AMD Athlon64 (AM2) Rev G 65nm processor";
static char* amd_k8_str = "AMD K8 architecture";
static char* unknown_intel_str = "Unknown Intel Processor";
static char* unknown_amd_str = "Unknown AMD Processor";

static volatile int init = 0;
static uint32_t eax, ebx, ecx, edx;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static uint32_t amdGetAssociativity(uint32_t flag)
{
    uint32_t asso= 0;

    switch ( flag ) 
    {
        case 0x0:
            asso = 0;
            break;

        case 0x1:
            asso = 1;
            break;

        case 0x2:
            asso = 2;
            break;

        case 0x4:
            asso = 4;
            break;

        case 0x6:
            asso = 8;
            break;

        case 0x8:
            asso = 16;
            break;

        case 0xA:
            asso = 32;
            break;

        case 0xB:
            asso = 48;
            break;

        case 0xC:
            asso = 64;
            break;

        case 0xD:
            asso = 96;
            break;

        case 0xE:
            asso = 128;
            break;

        case 0xF:
            asso = 0;
            break;

        default:
            break;
    }
    return asso;

}

static int intelCpuidFunc_4(CacheLevel** cachePool)
{
    int i;
    int level=0;
    int maxNumLevels=0;
    uint32_t valid=1;
    CacheLevel* pool;

    while (valid)
    {
        eax = 0x04;
        ecx = level;
        CPUID;
        valid = extractBitField(eax,5,0);
        if (!valid)
        {
            break;
        }
        level++;
    }

    maxNumLevels = level;
    *cachePool = (CacheLevel*) malloc(maxNumLevels * sizeof(CacheLevel));
    pool = *cachePool;

    for (i=0; i < maxNumLevels; i++) 
    {
        eax = 0x04;
        ecx = i;
        CPUID;

        pool[i].level = extractBitField(eax,3,5);
        pool[i].type = (CacheType) extractBitField(eax,5,0);
        pool[i].associativity = extractBitField(ebx,8,22)+1;
        pool[i].sets = ecx+1;
        pool[i].lineSize = extractBitField(ebx,12,0)+1;
        pool[i].size = pool[i].sets *
            pool[i].associativity *
            pool[i].lineSize;
        pool[i].threads = extractBitField(eax,10,14)+1;

        /* WORKAROUND cpuid reports wrong number of threads on SMT processor with SMT
         * turned off */
        if (i < 3)
        {
            if ((ghost_cpuid_info.model == NEHALEM_BLOOMFIELD) ||
                    (ghost_cpuid_info.model == NEHALEM_LYNNFIELD) ||
                    (ghost_cpuid_info.model == NEHALEM_WESTMERE) ||
                    (ghost_cpuid_info.model == NEHALEM_WESTMERE_M) ||
                    (ghost_cpuid_info.model == SANDYBRIDGE) ||
                    (ghost_cpuid_info.model == SANDYBRIDGE_EP) ||
                    (ghost_cpuid_info.model == IVYBRIDGE) ||
                    (ghost_cpuid_info.model == WESTMERE_EX) ||
                    (ghost_cpuid_info.model == NEHALEM_EX))
            {
                if (ghost_cpuid_topology.numThreadsPerCore == 1)
                {
                    pool[i].threads = 1;
                }
            }
        }

        /* :WORKAROUND:08/13/2009 08:34:15 AM:jt: For L3 caches the value is sometimes 
         * too large in here. Ask Intel what is wrong here!
         * Limit threads per Socket then to the maximum possible value.*/
        if(pool[i].threads > (int)
                (ghost_cpuid_topology.numCoresPerSocket*
                 ghost_cpuid_topology.numThreadsPerCore))
        {
            pool[i].threads = ghost_cpuid_topology.numCoresPerSocket*
                ghost_cpuid_topology.numThreadsPerCore;
        }
        pool[i].inclusive = edx&0x2;
    }

    return maxNumLevels;
}


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
ghost_cpuid_init (void)
{
    int isIntel = 1;

    /* FIXME: Race condition??? */
    if (init) return;
    init =1;

    eax = 0x00;
    CPUID;

    largest_function = eax;
    if (ebx == 0x68747541U)
    {
        isIntel = 0;
    }

    eax = 0x01;
    CPUID;
    ghost_cpuid_info.family = ((eax>>8)&0xFU) + ((eax>>20)&0xFFU);
    ghost_cpuid_info.model = (((eax>>16)&0xFU)<<4) + ((eax>>4)&0xFU);
    ghost_cpuid_info.stepping =  (eax&0xFU);

    switch ( ghost_cpuid_info.family ) 
    {
        case P6_FAMILY:
            switch ( ghost_cpuid_info.model ) 
            {
                case PENTIUM_M_BANIAS:
                    ghost_cpuid_info.name = pentium_m_b_str;
                    break;

                case PENTIUM_M_DOTHAN:
                    ghost_cpuid_info.name = pentium_m_d_str;
                    break;

                case CORE_DUO:
                    ghost_cpuid_info.name = core_duo_str;
                    break;

                case CORE2_65:
                    ghost_cpuid_info.name = core_2a_str;
                    break;

                case CORE2_45:
                    ghost_cpuid_info.name = core_2b_str;
                    break;

                case NEHALEM_BLOOMFIELD:
                    ghost_cpuid_info.name = nehalem_bloom_str;
                    break;

                case NEHALEM_LYNNFIELD:
                    ghost_cpuid_info.name = nehalem_lynn_str;
                    break;

                case NEHALEM_WESTMERE_M:

                case NEHALEM_WESTMERE:
                    ghost_cpuid_info.name = nehalem_west_str;
                    break;

                case SANDYBRIDGE:
                    ghost_cpuid_info.name = sandybridge_str;
                    break;

                case SANDYBRIDGE_EP:
                    ghost_cpuid_info.name = sandybridge_ep_str;
                    break;

                case IVYBRIDGE:
                    ghost_cpuid_info.name = ivybridge_str;
                    break;

                case NEHALEM_EX:
                    ghost_cpuid_info.name = nehalem_ex_str;
                    break;

                case WESTMERE_EX:
                    ghost_cpuid_info.name = westmere_ex_str;
                    break;

                case XEON_MP:
                    ghost_cpuid_info.name = xeon_mp_string;
                    break;

                default:
                    ghost_cpuid_info.name = unknown_intel_str;
                    break;
            }
            break;

        case MIC_FAMILY:
            switch ( ghost_cpuid_info.model ) 
            {
                case XEON_PHI:
                    ghost_cpuid_info.name = xeon_phi_string;
                    break;

            }
            break;

        case K8_FAMILY:

            if (isIntel)
            {
                ERROR_PLAIN_PRINT(Netburst architecture is not supported);
            }

            switch ( ghost_cpuid_info.model ) 
            {
                case OPTERON_DC_E:
                    ghost_cpuid_info.name = opteron_dc_e_str;
                    break;

                case OPTERON_DC_F:
                    ghost_cpuid_info.name = opteron_dc_f_str;
                    break;

                case ATHLON64_X2:

                case ATHLON64_X2_F:
                    ghost_cpuid_info.name = athlon64_str;
                    break;

                case ATHLON64_F1:

                case ATHLON64_F2:
                    ghost_cpuid_info.name = athlon64_f_str;
                    break;

                case ATHLON64_X2_G:
                    ghost_cpuid_info.name = athlon64_X2_g_str;
                    break;

                case ATHLON64_G1:

                case ATHLON64_G2:
                    ghost_cpuid_info.name = athlon64_g_str;
                    break;

                case OPTERON_SC_1MB:
                    ghost_cpuid_info.name = opteron_sc_str;
                    break;

                default:
                    ghost_cpuid_info.name = amd_k8_str;
                    break;
            }

            break;

        case K10_FAMILY:
            switch ( ghost_cpuid_info.model ) 
            {
                case BARCELONA:
                    ghost_cpuid_info.name = barcelona_str;
                    break;

                case SHANGHAI:
                    ghost_cpuid_info.name = shanghai_str;
                    break;

                case ISTANBUL:
                    ghost_cpuid_info.name = istanbul_str;
                    break;

                case MAGNYCOURS:
                    ghost_cpuid_info.name = magnycours_str;
                    break;

                default:
                    ghost_cpuid_info.name = unknown_amd_str;
                    break;
            }
            break;

        case K15_FAMILY:
            ghost_cpuid_info.name = interlagos_str;
            break;

        default:
            ERROR_PLAIN_PRINT(Processor is not supported);
            break;
    }

    ghost_cpuid_info.features = (char*) malloc(100*sizeof(char));
    if (edx & (1<<25)) strcpy(ghost_cpuid_info.features, "SSE ");
    if (edx & (1<<26)) strcat(ghost_cpuid_info.features, "SSE2 ");
    if (ecx & (1<<0))  strcat(ghost_cpuid_info.features, "SSE3 ");
    if (ecx & (1<<19)) strcat(ghost_cpuid_info.features, "SSE4.1 ");
    if (ecx & (1<<20)) strcat(ghost_cpuid_info.features, "SSE4.2 ");
    if (ecx & (1<<12)) strcat(ghost_cpuid_info.features, "FMA ");
    if (ecx & (1<<25)) strcat(ghost_cpuid_info.features, "AES ");
    if (ecx & (1<<28)) strcat(ghost_cpuid_info.features, "AVX ");

    ghost_cpuid_info.perf_version   =  0;
    if( ghost_cpuid_info.family == P6_FAMILY && 0x0A <= largest_function) 
    {
        eax = 0x0A;
        CPUID;
        ghost_cpuid_info.perf_version   =  (eax&0xFFU);
        ghost_cpuid_info.perf_num_ctr   =   ((eax>>8)&0xFFU);
        ghost_cpuid_info.perf_width_ctr =  ((eax>>16)&0xFFU);
        ghost_cpuid_info.perf_num_fixed_ctr =  (edx&0xFU);

        eax = 0x06;
        CPUID;
        if (eax & (1<<1))
        {
            ghost_cpuid_info.turbo = 1;
        }
        else
        {
            ghost_cpuid_info.turbo = 0;
        }
    }

    ghost_cpuid_topology.numHWThreads = sysconf(_SC_NPROCESSORS_CONF);

    //    if (!cpuid_isInCpuset())
    //    {
    cpu_set_t cpuSet;
    CPU_ZERO(&cpuSet);
    sched_getaffinity(0,sizeof(cpu_set_t), &cpuSet);
    ghost_cpuid_initTopology();
    ghost_cpuid_initCacheTopology();

    /* restore affinity mask of process */
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuSet);
    //    }
    //    else
    //    {
    //        likwid_inCPUSet = 1;
    //        printf("WARNING: You are running inside a restricted cpuset.\n");
    //        printf("You can only use physical processor ids or use logical ids inside your cpuset.\n");
    //    }

    return;
}

#define freeStrings  \
bdestroy(filename);  \
bdestroy(grepString); \
bdestroy(cpulist)


int
ghost_cpuid_isInCpuset(void)
{
    int pos = 0;
    bstring grepString = bformat("Cpus_allowed_list:");
    bstring filename = bformat("/proc/%d/status",getpid());
    FILE* fp = fopen(bdata(filename),"r");

    if (fp == NULL)
    {
        bdestroy(filename);
        bdestroy(grepString);
        return 0;
    } 
    else
    {
        bstring  cpulist;
        //uint32_t tmpThreads[MAX_NUM_THREADS];
        uint32_t tmpThreads[GHOST_MAX_THREADS];
        bstring src = bread ((bNread) fread, fp);
        if ((pos = binstr(src,0,grepString)) != BSTR_ERR)
        {
            int end = bstrchrp(src, 10, pos);
            pos = pos+blength(grepString);
            cpulist = bmidstr(src,pos, end-pos);
            btrimws(cpulist);

            if (bstr_to_cpuset_physical(tmpThreads, cpulist) < ghost_cpuid_topology.numHWThreads)
            {
                freeStrings;
                return 1;
            }
            else
            {
                freeStrings;
                return 0;
            }
        }
        return 0;
    }
}

void
ghost_cpuid_initTopology(void)
{
    uint32_t apicId;
    uint32_t bitField;
    uint32_t i;
    int level;
    int prevOffset = 0;
    int currOffset = 0;
    cpu_set_t set;
    HWThread* hwThreadPool;
    int hasBLeaf = 0;
    int maxNumLogicalProcs;
    int maxNumLogicalProcsPerCore;
    int maxNumCores;
    TreeNode* currentNode;
    int width;

    /* check if 0x0B cpuid leaf is supported */
    if (largest_function >= 0x0B)
    {
        eax = 0x0B;
        ecx = 0;
        CPUID;

        if (ebx) 
        {
            hasBLeaf = 1;
        }
    }

    hwThreadPool = (HWThread*) malloc(ghost_cpuid_topology.numHWThreads * sizeof(HWThread));
    ghost_tree_init(&ghost_cpuid_topology.topologyTree, 0);

    if ( likwid_inCPUSet )
    {


    }
    else if (hasBLeaf)
    {
        for (i=0; i < ghost_cpuid_topology.numHWThreads; i++)
        {

            CPU_ZERO(&set);
            CPU_SET(i,&set);
            sched_setaffinity(0, sizeof(cpu_set_t), &set);
            eax = 0x0B;
            ecx = 0;
            CPUID;
            apicId = edx;
            hwThreadPool[i].apicId = apicId;

            for (level=0; level < 3; level++)
            {
                eax = 0x0B;
                ecx = level;
                CPUID;
                currOffset = eax&0xFU;

                switch ( level ) {
                    case 0:  /* SMT thread */
                        bitField = extractBitField(apicId,
                                currOffset,
                                0);
                        hwThreadPool[i].threadId = bitField;
                        break;

                    case 1:  /* Core */
                        bitField = extractBitField(apicId,
                                currOffset-prevOffset,
                                prevOffset);
                        hwThreadPool[i].coreId = bitField;
                        break;

                    case 2:  /* Package */
                        bitField = extractBitField(apicId,
                                32-prevOffset,
                                prevOffset);
                        hwThreadPool[i].packageId = bitField;
                        break;

                }
                prevOffset = currOffset;
            }
        }
    }
    else
    {
        switch ( ghost_cpuid_info.family ) 
        {

            case MIC_FAMILY:

            case P6_FAMILY:
                eax = 0x01;
                CPUID;
                maxNumLogicalProcs = extractBitField(ebx,8,16);

                /* Check number of cores per package */
                eax = 0x04;
                ecx = 0;
                CPUID;
                maxNumCores = extractBitField(eax,6,26)+1;

                maxNumLogicalProcsPerCore = maxNumLogicalProcs/maxNumCores;

                for (i=0; i<  ghost_cpuid_topology.numHWThreads; i++)
                {
                    CPU_ZERO(&set);
                    CPU_SET(i,&set);
                    sched_setaffinity(0, sizeof(cpu_set_t), &set);

                    eax = 0x01;
                    CPUID;
                    hwThreadPool[i].apicId = extractBitField(ebx,8,24);

                    /* ThreadId is extracted from th apicId using the bit width
                     * of the number of logical processors
                     * */
                    hwThreadPool[i].threadId =
                        extractBitField(hwThreadPool[i].apicId,
                                getBitFieldWidth(maxNumLogicalProcsPerCore),0); 

                    /* CoreId is extracted from th apicId using the bitWidth 
                     * of the number of logical processors as offset and the
                     * bit width of the number of cores as width
                     * */
                    hwThreadPool[i].coreId =
                        extractBitField(hwThreadPool[i].apicId,
                                getBitFieldWidth(maxNumCores),
                                getBitFieldWidth(maxNumLogicalProcsPerCore)); 

                    hwThreadPool[i].packageId =
                        extractBitField(hwThreadPool[i].apicId,
                                8-getBitFieldWidth(maxNumLogicalProcs),
                                getBitFieldWidth(maxNumLogicalProcs)); 
                }
                break;

            case K8_FAMILY:
                /* AMD Bios manual Rev. 2.28 section 3.1
                 * Legacy method */
                /*FIXME: This is a bit of a hack */

                maxNumLogicalProcsPerCore = 1;
                maxNumLogicalProcs = 1;

                eax = 0x80000008;
                CPUID;

                maxNumCores =  extractBitField(ecx,8,0)+1;

                for (i=0; i<  ghost_cpuid_topology.numHWThreads; i++)
                {
                    CPU_ZERO(&set);
                    CPU_SET(i,&set);
                    sched_setaffinity(0, sizeof(cpu_set_t), &set);

                    eax = 0x01;
                    CPUID;
                    hwThreadPool[i].apicId = extractBitField(ebx,8,24);

                    /* ThreadId is extracted from th apicId using the bit width
                     * of the number of logical processors
                     * */
                    hwThreadPool[i].threadId =
                        extractBitField(hwThreadPool[i].apicId,
                                getBitFieldWidth(maxNumLogicalProcsPerCore),0); 

                    /* CoreId is extracted from th apicId using the bitWidth 
                     * of the number of logical processors as offset and the
                     * bit width of the number of cores as width
                     * */
                    hwThreadPool[i].coreId =
                        extractBitField(hwThreadPool[i].apicId,
                                getBitFieldWidth(maxNumCores),
                                0); 

                    hwThreadPool[i].packageId =
                        extractBitField(hwThreadPool[i].apicId,
                                8-getBitFieldWidth(maxNumCores),
                                getBitFieldWidth(maxNumCores)); 
                }
                break;

            case K15_FAMILY:

            case K10_FAMILY:
                /* AMD Bios manual Rev. 2.28 section 3.2
                 * Extended method */
                eax = 0x80000008;
                CPUID;

                width =  extractBitField(ecx,4,12);

                if (width == 0)
                {
                    width =  extractBitField(ecx,8,0)+1;
                }

                eax = 0x01;
                CPUID;
                maxNumLogicalProcs =  extractBitField(ebx,8,16);
                maxNumCores = extractBitField(ecx,8,0)+1;


                for (i=0; i<  ghost_cpuid_topology.numHWThreads; i++)
                {
                    CPU_ZERO(&set);
                    CPU_SET(i,&set);
                    sched_setaffinity(0, sizeof(cpu_set_t), &set);

                    eax = 0x01;
                    CPUID;
                    hwThreadPool[i].apicId = extractBitField(ebx,8,24);
                    /* AMD only knows cores */
                    hwThreadPool[i].threadId = 0;

                    hwThreadPool[i].coreId =
                        extractBitField(hwThreadPool[i].apicId,
                                width, 0); 
                    hwThreadPool[i].packageId =
                        extractBitField(hwThreadPool[i].apicId,
                                (8-width), width); 
                }

                break;
        }
    }

    for (i=0; i<  ghost_cpuid_topology.numHWThreads; i++)
    {
        /* Add node to Topology tree */
        if (!ghost_tree_nodeExists(ghost_cpuid_topology.topologyTree,
                    hwThreadPool[i].packageId))
        {
            ghost_tree_insertNode(ghost_cpuid_topology.topologyTree,
                    hwThreadPool[i].packageId);
        }
        currentNode = ghost_tree_getNode(ghost_cpuid_topology.topologyTree,
                hwThreadPool[i].packageId);

        if (!ghost_tree_nodeExists(currentNode, hwThreadPool[i].coreId))
        {
            ghost_tree_insertNode(currentNode, hwThreadPool[i].coreId);
        }
        currentNode = ghost_tree_getNode(currentNode, hwThreadPool[i].coreId);

        if (!ghost_tree_nodeExists(currentNode, i))
        {
            /*
               printf("WARNING: Thread already exists!\n");
               */
            ghost_tree_insertNode(currentNode, i);
        }

    }

    ghost_cpuid_topology.threadPool = hwThreadPool;
    ghost_cpuid_topology.numSockets = ghost_tree_countChildren(ghost_cpuid_topology.topologyTree);
    currentNode = ghost_tree_getChildNode(ghost_cpuid_topology.topologyTree);
    ghost_cpuid_topology.numCoresPerSocket = ghost_tree_countChildren(currentNode);
    currentNode = ghost_tree_getChildNode(currentNode);
    ghost_cpuid_topology.numThreadsPerCore = ghost_tree_countChildren(currentNode);
}

void 
ghost_cpuid_initCacheTopology()
{
    int maxNumLevels=0;
    int id=0;
    CacheLevel* cachePool = NULL;
    CacheType type = DATACACHE;

    switch ( ghost_cpuid_info.family ) 
    {
        case MIC_FAMILY:

        case P6_FAMILY:

            if (largest_function >= 4)
            {
                maxNumLevels = intelCpuidFunc_4(&cachePool);
            }
            else
            {
                //				intelCpuidFunc_2(&cachePool);
            }

            break;

        case K8_FAMILY:
            maxNumLevels = 2;
            cachePool = (CacheLevel*) malloc(maxNumLevels * sizeof(CacheLevel));

            eax = 0x80000005;
            CPUID;
            cachePool[0].level = 1;
            cachePool[0].type = DATACACHE;
            cachePool[0].associativity = extractBitField(ecx,8,16);
            cachePool[0].lineSize = extractBitField(ecx,8,0);
            cachePool[0].size =  extractBitField(ecx,8,24) * 1024;
            if ((cachePool[0].associativity * cachePool[0].lineSize) != 0)
            {
                cachePool[0].sets = cachePool[0].size/
                    (cachePool[0].associativity * cachePool[0].lineSize);
            }
            cachePool[0].threads = 1;
            cachePool[0].inclusive = 1;

            eax = 0x80000006;
            CPUID;
            cachePool[1].level = 2;
            cachePool[1].type = UNIFIEDCACHE;
            cachePool[1].associativity = 
                amdGetAssociativity(extractBitField(ecx,4,12));
            cachePool[1].lineSize = extractBitField(ecx,8,0);
            cachePool[1].size =  extractBitField(ecx,16,16) * 1024;
            if ((cachePool[0].associativity * cachePool[0].lineSize) != 0)
            {
                cachePool[1].sets = cachePool[1].size/
                    (cachePool[1].associativity * cachePool[1].lineSize);
            }
            cachePool[1].threads = 1;
            cachePool[1].inclusive = 1;

            break;


        case K10_FAMILY:
            /* FIXME: Adds one level for the instruction cache on Intel
             * This fixes the level for the cores
             */
            maxNumLevels = 3;
            cachePool = (CacheLevel*) malloc(maxNumLevels * sizeof(CacheLevel));

            eax = 0x80000005;
            CPUID;
            cachePool[0].level = 1;
            cachePool[0].type = DATACACHE;
            cachePool[0].associativity = extractBitField(ecx,8,16);
            cachePool[0].lineSize = extractBitField(ecx,8,0);
            cachePool[0].size =  extractBitField(ecx,8,24) * 1024;
            if ((cachePool[0].associativity * cachePool[0].lineSize) != 0)
            {
                cachePool[0].sets = cachePool[0].size/
                    (cachePool[0].associativity * cachePool[0].lineSize);
            }
            cachePool[0].threads = 1;
            cachePool[0].inclusive = 1;

            eax = 0x80000006;
            CPUID;
            cachePool[1].level = 2;
            cachePool[1].type = UNIFIEDCACHE;
            cachePool[1].associativity = 
                amdGetAssociativity(extractBitField(ecx,4,12));
            cachePool[1].lineSize = extractBitField(ecx,8,0);
            cachePool[1].size =  extractBitField(ecx,16,16) * 1024;
            if ((cachePool[0].associativity * cachePool[0].lineSize) != 0)
            {
                cachePool[1].sets = cachePool[1].size/
                    (cachePool[1].associativity * cachePool[1].lineSize);
            }
            cachePool[1].threads = 1;
            cachePool[1].inclusive = 1;

            cachePool[2].level = 3;
            cachePool[2].type = UNIFIEDCACHE;
            cachePool[2].associativity =
                amdGetAssociativity(extractBitField(edx,4,12));
            cachePool[2].lineSize = extractBitField(edx,8,0);
            cachePool[2].size =  (extractBitField(edx,14,18)+1) * 524288;
            if ((cachePool[0].associativity * cachePool[0].lineSize) != 0)
            {
                cachePool[2].sets = cachePool[1].size/
                    (cachePool[1].associativity * cachePool[1].lineSize);
            }

            if (ghost_cpuid_info.model != MAGNYCOURS)
            {
                cachePool[2].threads = ghost_cpuid_topology.numCoresPerSocket;
            }
            else
            {
                cachePool[2].threads = ghost_cpuid_topology.numCoresPerSocket/2;
                cachePool[2].size /= 2 ;
            }

            cachePool[2].inclusive = 1;

            break;

        case K15_FAMILY:

            maxNumLevels = 0;
            cachePool = (CacheLevel*) malloc(3 * sizeof(CacheLevel));

            while (type)
            {
                ecx = id;
                eax = 0x8000001D;
                CPUID;
                type = (CacheType) extractBitField(eax,4,0);

                if ((type == DATACACHE) || (type == UNIFIEDCACHE))
                {
                    cachePool[maxNumLevels].level =   extractBitField(eax,3,5);
                    cachePool[maxNumLevels].type = type;
                    cachePool[maxNumLevels].associativity = extractBitField(ebx,10,22)+1;
                    cachePool[maxNumLevels].lineSize = extractBitField(ebx,12,0)+1;
                    cachePool[maxNumLevels].sets =  extractBitField(ecx,32,0)+1;
                    cachePool[maxNumLevels].size = cachePool[maxNumLevels].associativity *
                        cachePool[maxNumLevels].lineSize * cachePool[maxNumLevels].sets;
                    cachePool[maxNumLevels].threads =  extractBitField(eax,12,14)+1;
                    cachePool[maxNumLevels].inclusive =  (edx & (0x1<<1));
                    maxNumLevels++;
                }
                id++;
            }
            break;

        default:
            ERROR_PLAIN_PRINT(Processor is not supported);
            break;
    }

    ghost_cpuid_topology.numCacheLevels = maxNumLevels;
    ghost_cpuid_topology.cacheLevels = cachePool;
}



