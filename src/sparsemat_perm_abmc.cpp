#include "ghost/sparsemat.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/bincrs.h"
#include <vector>
#include <map>
//uses METIS for partitioning
#ifdef GHOST_HAVE_METIS
#include "metis.h"
#endif
#ifdef GHOST_HAVE_COLPACK
#include "ColPack/ColPackHeaders.h"
#endif
#include "ghost/omp.h"
#include "ghost/pumap.h"
#include <limits>

extern "C" ghost_error ghost_sparsemat_perm_abmc(ghost_context *ctx, ghost_sparsemat *mat)
{
#if (defined GHOST_HAVE_METIS && defined GHOST_HAVE_COLPACK)
    ghost_error ret = GHOST_SUCCESS;
    GHOST_INFO_LOG("Create permutation from coloring");

    char *blockSize_env = (getenv("GHOST_ABMC_BLOCKSIZE"));

    int blocksize = 64;
    if(blockSize_env)
    {
        blocksize= atoi(blockSize_env);
    }
    ctx->blockSize = blocksize;

    //std::vector<int> block_sizes_default{8,32,64,128};
    std::vector<int> block_sizes_default{64,128};
    std::vector<int> block_sizes;
    int opt_block;

    if(blockSize_env)
    {
        block_sizes.push_back(blocksize);
        opt_block = blocksize;
    }
    else
    {
        block_sizes = block_sizes_default;
    }

    double min_time = std::numeric_limits<double>::max();

    for(int blockIdx=0; blockIdx<(int)block_sizes.size()+1; ++blockIdx)
    {
        if(blockIdx == block_sizes.size())
        {
            blocksize = opt_block;
            if(opt_block==block_sizes.back())
            {
                break;
            }
        }
        else
        {
            blocksize = block_sizes[blockIdx];
        }
        printf("blocksize = %d\n", blocksize);
        ghost_lidx *curcol = NULL;
        uint32_t** adolc;
        std::vector<int>* colvec = NULL;
        uint32_t *adolc_data = NULL;
        ColPack::GraphColoringInterface *GC=new ColPack::GraphColoringInterface(-1);
        int64_t pos=0;
        bool oldperm = false;

        int nrows_b, nnz_b;
        ghost_lidx* rowPtr_b;
        std::vector<ghost_lidx> col_b;
        std::vector<std::map<int,int>> unique_col;
        ghost_lidx *partPtr, *permPartPtr;
        std::vector<std::vector<ghost_lidx>> partRow;
        std::vector<std::vector<ghost_lidx>> color_part_map;
        ghost_lidx *color_part_ptr;
        int ctr=0;

        //ghost_permutation *oldperm = NULL;

        int me, i, j;
        ghost_lidx *rowPtr = NULL;
        //ghost_gidx *col = NULL;
        ghost_lidx *col = NULL;
        ghost_lidx nnz = 0, nnzlocal = 0;

        char *color_dist_env = getenv("GHOST_COLOR_DISTANCE");
        int color_dist = 2;

        if(color_dist_env)
        {
            color_dist = atoi(color_dist_env);
        }

        if(color_dist != 1 && color_dist != 2)
        {
            printf("Dist %d not supported, falling back to dist-2\n",color_dist);
            color_dist = 2;
        }

        ghost_lidx *rptlocal = NULL;
        ghost_lidx *collocal = NULL;

        nnz = SPM_NNZ(mat); 

        GHOST_CALL_GOTO(ghost_rank(&me,ctx->mpicomm),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&rptlocal,(ctx->row_map->ldim[me]+1) * sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&collocal,nnz * sizeof(ghost_lidx)),err,ret);

        rptlocal[0] = 0;

        for (i=0; i<ctx->row_map->dim; i++) {
            rptlocal[i+1] = rptlocal[i];

            ghost_lidx orig_row = i;
            if (ctx->row_map->loc_perm) {
                orig_row = ctx->row_map->loc_perm_inv[i];
                // orig_row = ctx->row_map->loc_perm[i];
            }
            ghost_lidx * col = &mat->col[mat->chunkStart[orig_row]];
            ghost_lidx orig_row_len = mat->chunkStart[orig_row+1]-mat->chunkStart[orig_row];

            for(j=0; j<orig_row_len; ++j) {
                if (col[j] < mat->context->row_map->dim) {
                    if(ctx->row_map->loc_perm) 
                    {
                        collocal[nnzlocal] = ctx->row_map->loc_perm[col[j]];
                    }
                    else
                    {
                        collocal[nnzlocal] = col[j];
                    }
                    nnzlocal++;
                    rptlocal[i+1]++;
                }
            }
        }

        //partition using METIS
        int nrows = ctx->row_map->dim;
        int ncon = 1;
        int nparts = (int)(nrows/(double)blocksize);
        int objval;
        int *part;
        GHOST_CALL_GOTO(ghost_malloc((void **)&part,nrows * sizeof(ghost_gidx)),err,ret);

        printf("partitioning graph to %d parts\n", nparts);
        //  METIS_PartGraphKway(&nrows, &ncon, mat->chunkStart, mat->col, NULL, NULL, NULL, &nparts, NULL, NULL, NULL, &objval, part);
        METIS_PartGraphKway(&nrows, &ncon, rptlocal, collocal, NULL, NULL, NULL, &nparts, NULL, NULL, NULL, &objval, part);
        printf("finished partitioning nparts=%d\n", nparts);

        partRow.resize(nparts);
        GHOST_CALL_GOTO(ghost_malloc((void **)&partPtr,(nparts+1)*sizeof(ghost_lidx)),err,ret);

        for (i=0;i<nparts+1;i++) {
            partPtr[i] = 0;
        }

        for (i=0;i<nrows;i++) {
            partRow[part[i]].push_back(i);
            partPtr[part[i]+1]++;
        }

        for (i=1;i<nparts+1;i++) {
            partPtr[i] += partPtr[i-1];
        }

        //#now make a new matrix from the partition
        nrows_b = nparts;
        GHOST_CALL_GOTO(ghost_malloc((void **)&rowPtr_b,(nparts+1)*sizeof(ghost_lidx)),err,ret);
        for(i=0; i<(nparts+1); ++i)
        {
            rowPtr_b[i] = 0;
        }

        //find nnz
        //col = (ghost_lidx*) mat->col;
        //rowPtr = (ghost_lidx*) mat->chunkStart;
        col = collocal;
        rowPtr = rptlocal;

        unique_col.resize(nrows_b);

        //for each part I need to know to which part it is connected
        for(int row=0; row<nrows; ++row)
        {
            int currPart_id = part[row];

            for(int idx=rowPtr[row]; idx<rowPtr[row+1]; ++idx)
            {
                int currCol_idx = col[idx];
                //find to which part does this col belong
                int partnerPart = part[currCol_idx];
                unique_col[currPart_id][partnerPart]=1;//I just need to create unique entries; therefore map
            }
        }

        //count nnzr
        for(int part_idx=0; part_idx<nrows_b; ++part_idx)
        {
            int nnzr_b = unique_col[part_idx].size();
            rowPtr_b[part_idx+1] = rowPtr_b[part_idx] + nnzr_b;
            for(auto& map_el : unique_col[part_idx])
            {
                col_b.push_back(map_el.first);
                //            printf("%d \t %d\n", part_idx+1, col_b.back()+1);
            }
        }

        //now color this graph
        nnz_b = col_b.size();
        adolc = new uint32_t*[nrows_b];
        adolc_data = new uint32_t[nnz_b+nrows_b];

        for (i=0;i<nrows_b;i++)
        {
            adolc[i]=&(adolc_data[pos]);
            adolc_data[pos++]=(rowPtr_b[i+1]-rowPtr_b[i]);
            for (j=rowPtr_b[i];j<rowPtr_b[i+1];j++)
            {
                adolc_data[pos++]=col_b[j];
            }
        }

        GC->BuildGraphFromRowCompressedFormat(adolc, nrows_b);

        if(color_dist == 1)
        {
            COLPACK_CALL_GOTO(GC->Coloring(),err,ret);
            //COLPACK_CALL_GOTO(GC->DistanceOneColoring(),err,ret);
        }
        else
        {
            COLPACK_CALL_GOTO(GC->Coloring("NATURAL", "DISTANCE_TWO"),err,ret);
            //COLPACK_CALL_GOTO(GC->DistanceTwoColoring(),err,ret);
            /*
               if (GC->CheckDistanceTwoColoring(2)) {
               GHOST_ERROR_LOG("Error in coloring!");
               ret = GHOST_ERR_COLPACK;
               goto err;
               }*/
        }

        ctx->ncolors = GC->GetVertexColorCount();

        printf("No. of colors = %d\n", ctx->ncolors);

        if (!ctx->row_map->loc_perm) {
            //GHOST_CALL_GOTO(ghost_malloc((void **)&(ctx->row_map),sizeof(ghost_map)),err,ret);
            //ctx->row_map->loc_perm->method = GHOST_PERMUTATION_UNSYMMETRIC; //you can also make it symmetric
            GHOST_CALL_GOTO(ghost_malloc((void **)&(ctx->row_map->loc_perm),sizeof(ghost_gidx)*ctx->row_map->dim),err,ret);
            GHOST_CALL_GOTO(ghost_malloc((void **)&(ctx->row_map->loc_perm_inv),sizeof(ghost_gidx)*ctx->row_map->dim),err,ret);   
            /*     GHOST_CALL_GOTO(ghost_malloc((void **)ctx->col_map->loc_perm,sizeof(ghost_gidx)*ncols_halo_padded),err,ret);
                   GHOST_CALL_GOTO(ghost_malloc((void **)ctx->col_map->loc_perm_inv,sizeof(ghost_gidx)*ncols_halo_padded),err,ret);

                   for(int i=0; i<ncols_halo_padded; ++i) {
                   ctx->col_map->loc_perm[i] = i;
                   ctx->col_map->loc_perm_inv[i] = i;
                   }
                   */      
#ifdef GHOST_HAVE_CUDA
            GHOST_CALL_GOTO(ghost_cu_malloc((void **)&(ctx->row_map->loc_perm->cu_perm),sizeof(ghost_gidx)*ctx->row_map->dim),err,ret);
#endif

        } else if(ctx->row_map->loc_perm == ctx->col_map->loc_perm) { // symmetrix permutation
            oldperm = true; //ctx->row_map->loc_perm;
            //        ctx->row_map->loc_perm->method = GHOST_PERMUTATION_UNSYMMETRIC;//change to unsymmetric
            /*    GHOST_CALL_GOTO(ghost_malloc((void **)ctx->col_map->loc_perm,sizeof(ghost_gidx)*ncols_halo_padded),err,ret);
                  GHOST_CALL_GOTO(ghost_malloc((void **)ctx->col_map->loc_perm_inv,sizeof(ghost_gidx)*ncols_halo_padded),err,ret);

                  for(int i=0; i<ncols_halo_padded; ++i) {
                  ctx->col_map->loc_perm[i] = ctx->row_map->loc_perm[i];
                  ctx->col_map->loc_perm_inv[i] = ctx->row_map->loc_perm_inv[i];
                  }        */
        } else {
            oldperm = true; //ctx->row_map->loc_perm;
        }

        //permute the partPtr according to color
        GHOST_CALL_GOTO(ghost_malloc((void **)&permPartPtr,(nrows_b+1)*sizeof(ghost_lidx)),err,ret);

        GHOST_CALL_GOTO(ghost_malloc((void **)&color_part_ptr,(ctx->ncolors+1)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->color_ptr,(ctx->ncolors+1)*sizeof(ghost_lidx)),err,ret);

        GHOST_CALL_GOTO(ghost_malloc((void **)&curcol,(ctx->ncolors)*sizeof(ghost_lidx)),err,ret);
        memset(curcol,0,ctx->ncolors*sizeof(ghost_lidx));

        color_part_map.resize(ctx->ncolors);

        colvec = GC->GetVertexColorsPtr();


        for (i=0;i<ctx->ncolors+1;i++) {
            ctx->color_ptr[i] = 0;
        }

        for (i=0;i<ctx->ncolors+1;i++) {
            color_part_ptr[i] = 0;
        }


        for (i=0;i<nrows_b;++i) {
            ctx->color_ptr[(*colvec)[i]+1]++;
            int partSize = partPtr[i+1] - partPtr[i];
            //How much part contain this color
            color_part_ptr[(*colvec)[i]+1]+=partSize;
        }

        for (i=1;i<ctx->ncolors+1;i++) {
            ctx->color_ptr[i] += ctx->color_ptr[i-1];
            color_part_ptr[i] += color_part_ptr[i-1];
        }

        if (oldperm) {
            for (int partIdx=0; partIdx<nrows_b; partIdx++)
            {
                color_part_map[(*colvec)[partIdx]].push_back(partIdx);

                int partSize = partPtr[partIdx+1]-partPtr[partIdx];
                for(int rowIdx=0; rowIdx<partSize; ++rowIdx)
                {
                    //find rows in parts
                    int currRow = partRow[partIdx][rowIdx];
                    int idx = ctx->row_map->loc_perm_inv[currRow];
                    ctx->row_map->loc_perm[idx] = curcol[(*colvec)[partIdx]] + color_part_ptr[(*colvec)[partIdx]];
                    curcol[(*colvec)[partIdx]]++;
                }
            }
        } else {
            for (int partIdx=0; partIdx<nrows_b; ++partIdx)
            {
                int partSize = partPtr[partIdx+1]-partPtr[partIdx];
                color_part_map[(*colvec)[partIdx]].push_back(partIdx);

                for(int rowIdx=0; rowIdx<partSize; ++rowIdx)
                {
                    //find rows in parts
                    int currRow = partRow[partIdx][rowIdx];
                    ctx->row_map->loc_perm[currRow] = curcol[(*colvec)[partIdx]] + color_part_ptr[(*colvec)[partIdx]];
                    curcol[(*colvec)[partIdx]]++;
                }
            }
        }

        for (i=0;i<ctx->row_map->dim;i++) {
            ctx->row_map->loc_perm_inv[ctx->row_map->loc_perm[i]] = i;
        }

        permPartPtr[0] =0;
        //now permute the partPtr
        for(int chrom=0; chrom<ctx->ncolors; ++chrom)
        {
            for(int k=0; k<(int)color_part_map[chrom].size(); ++k)
            {
                int currPartIdx = color_part_map[chrom][k];
                permPartPtr[ctr+1]  = partPtr[currPartIdx+1]-partPtr[currPartIdx];
                ++ctr;
            }
        }

        for(int partIdx=0; partIdx<nrows_b; ++partIdx)
        {
            permPartPtr[partIdx+1] += permPartPtr[partIdx];
        }

        ghost_lidx *new_rptlocal = NULL;
        ghost_lidx *new_collocal = NULL;
        double *new_vallocal = NULL, *b = NULL, *x = NULL;
        //create matrix  and test performance
        ghost_malloc((void **)&new_rptlocal,(ctx->row_map->ldim[me]+1) * sizeof(ghost_lidx));
        ghost_malloc((void **)&new_collocal,nnz * sizeof(ghost_lidx));
        ghost_malloc((void **)&new_vallocal,nnz * sizeof(double));
        ghost_malloc((void **)&x,(ctx->row_map->ldim[me]) * sizeof(double));
        ghost_malloc((void **)&b,(ctx->row_map->ldim[me]) * sizeof(double));

        new_rptlocal[0] = 0;
        nnzlocal = 0;

        for (i=0; i<ctx->row_map->dim; i++) {
            b[i] = 0;
            x[i] = 0.001;
            new_rptlocal[i+1] = new_rptlocal[i];

            ghost_lidx orig_row = i;
            if (ctx->row_map->loc_perm) {
                orig_row = ctx->row_map->loc_perm_inv[i];
                // orig_row = ctx->row_map->loc_perm[i];
            }
            ghost_lidx * col = &mat->col[mat->chunkStart[orig_row]];
            ghost_lidx orig_row_len = mat->chunkStart[orig_row+1]-mat->chunkStart[orig_row];

            for(j=0; j<orig_row_len; ++j) {
                if (col[j] < mat->context->row_map->dim) {
                    if(ctx->row_map->loc_perm)
                    {
                        new_collocal[nnzlocal] = ctx->row_map->loc_perm[col[j]];
                        new_vallocal[nnzlocal] = 0;
                    }
                    else
                    {
                        new_collocal[nnzlocal] = col[j];
                        new_vallocal[nnzlocal] = 0;
                    }
                    nnzlocal++;
                    new_rptlocal[i+1]++;
                }
            }
        }

        double start_time, end_time;


        ghost_barrier();
        ghost_timing_wcmilli(&start_time);

        //do SpMV and check
#pragma omp parallel for schedule(static)
        for (ghost_lidx row=0; row<ctx->row_map->dim; ++row) {
            double temp = 0;
            ghost_lidx idx = new_rptlocal[row];
#pragma unroll
#pragma simd reduction(+:temp)
            for (ghost_lidx j=new_rptlocal[row]; j<new_rptlocal[row+1]; j++) {
                temp = temp + new_vallocal[j] * x[new_collocal[j]];
            }
            b[row]=temp;
        }

        ghost_barrier();
        ghost_timing_wcmilli(&end_time);

        double tot_time = end_time-start_time;
        if(tot_time<min_time)
        {
            min_time = tot_time;
            opt_block = blocksize;
        }

        //printf("Time taken = %f\n",(tot_time));

        free(new_rptlocal);
        free(new_collocal);
        free(new_vallocal);
        free(b);
        free(x);

        ctx->part_ptr = permPartPtr;
        delete [] adolc_data;
        delete [] adolc;
        delete GC;
        free(part);
        free(rowPtr_b);
        free(curcol);
        free(color_part_ptr);
        free(partPtr);
        // free(rpt);
        free(rptlocal);
        free(collocal);

    }
    goto out;

err:

out:
    return ret;
#else
    UNUSED(mat);
    UNUSED(ctx);
    GHOST_ERROR_LOG("ColPack not available!");
    return GHOST_ERR_NOT_IMPLEMENTED;
#endif
}

