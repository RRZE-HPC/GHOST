/*
 * =======================================================================================
 *
 *      Filename:  cpuid_types.h
 *
 *      Description:  Types file for cpuid module.
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

#ifndef CPUID_TYPES_H
#define CPUID_TYPES_H

typedef enum {
    NOCACHE=0,
    DATACACHE,
    INSTRUCTIONCACHE,
    UNIFIEDCACHE,
    ITLB,
    DTLB} CacheType;

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
    TreeNode* topologyTree;
} CpuTopology;


#endif /*CPUID_TYPES_H*/
