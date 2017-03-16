#include "ghost/sparsemat.h"
#include "ghost/omp.h"
#include "ghost/locality.h"

ghost_error kacz_analyze_print_single_proc(int id, ghost_sparsemat *mat)
{
    int me;
    GHOST_CALL_RETURN(ghost_rank(&me, mat->context->mpicomm));

    if(me==id)
    {
        printf("Rank = %d\n",id);
        ghost_lidx line_size = 12;
        ghost_lidx n_lines = mat->context->kacz_setting.active_threads / line_size;
        ghost_lidx rem_lines =  mat->context->kacz_setting.active_threads % line_size;
        int start=0 ;
        int end=0;
        ghost_lidx *rows;
        ghost_lidx *nnz;

        printf("%10s:","THREADS");

        for(int line=0; line<n_lines; ++line) {
            start = line*line_size;
            end   = (line+1)*line_size;

            for(int i=start ; i<end; ++i){
                printf("|%10d",i+1);
            }
            printf("\n");
            printf("%10s:","");
        }

        start = mat->context->kacz_setting.active_threads - rem_lines;
        end   = mat->context->kacz_setting.active_threads;

        for(int i=start ; i<end; ++i){
            printf("|%10d",i+1);
        }

        printf("|%10s","TOTAL");

        const char *zone_name[4];
        zone_name[0] = "PURE ZONE";
        zone_name[1] = "RED TRANS ZONE";
        zone_name[2] = "TRANS IN TRANS ZONE";
        zone_name[3] = "BLACK TRANS ZONE";

        rows = malloc(mat->context->kacz_setting.active_threads*sizeof(ghost_lidx));
        nnz  = malloc(mat->context->kacz_setting.active_threads*sizeof(ghost_lidx));

#ifdef GHOST_HAVE_OPENMP
#pragma omp parallel shared(line_size,n_lines,rem_lines) private(start,end)
        {
#endif
            ghost_lidx tid = ghost_omp_threadnum();

            for(ghost_lidx zone=0; zone<4; ++zone) {
                rows[tid] = mat->context->zone_ptr[4*tid+zone+1] - mat->context->zone_ptr[4*tid+zone];
                nnz[tid]  = 0;

                if(rows[tid]!=0) {
                    for(int j=mat->context->zone_ptr[4*tid+zone]; j<mat->context->zone_ptr[4*tid+zone+1]; ++j) {
                        nnz[tid] += mat->rowLen[j];
                    }
                }

#pragma omp barrier

#pragma omp single
                {
                    printf("\n\n%s\n",zone_name[zone]);
                    printf("%10s:","ROWS");
                    ghost_lidx ctr = 0;
                    start=0 ;
                    end=0;

                    for(int line=0; line<n_lines; ++line) {
                        start = line*line_size;
                        end   = (line+1)*line_size;

                        for(int i=start ; i<end; ++i){
                            printf("|%10d",rows[i]);
                            ctr += rows[i];
                        }
                        printf("\n");
                        printf("%10s:","");
                    }

                    start = mat->context->kacz_setting.active_threads - rem_lines;
                    end   = mat->context->kacz_setting.active_threads;

                    for(int i=start ; i<end; ++i){
                        printf("|%10d",rows[i]);
                        ctr += rows[i];
                    }

                    printf("|%10d",ctr);
                    printf("\n%10s:","%");

                    if(ctr!=0) {

                        for(int line=0; line<n_lines; ++line) {
                            start = line*line_size;
                            end   = (line+1)*line_size;

                            for(int i=start ; i<end; ++i){
                                printf("|%10d",(int)(((double)rows[i]/ctr)*100));
                            }
                            printf("\n");
                            printf("%10s:","");
                        }

                        start = mat->context->kacz_setting.active_threads - rem_lines;
                        end   = mat->context->kacz_setting.active_threads;


                        for(int i=start ; i<end; ++i){
                            printf("|%10d",(int)(((double)rows[i]/ctr)*100));
                        }
                        printf("|%10d",100);
                    }


                    printf("\n%10s:","NNZ");
                    ctr = 0;

                    for(int line=0; line<n_lines; ++line) {
                        start = line*line_size;
                        end   = (line+1)*line_size;

                        for(int i=start ; i<end; ++i){
                            printf("|%10d",nnz[i]);
                            ctr += nnz[i];
                        }
                        printf("\n");
                        printf("%10s:","");

                    }

                    start = mat->context->kacz_setting.active_threads - rem_lines;
                    end   = mat->context->kacz_setting.active_threads;

                    for(int i=start ; i<end; ++i){
                        printf("|%10d",nnz[i]);
                        ctr += nnz[i];
                    }

                    printf("|%10d",ctr);
                    printf("\n%10s:","%");

                    if(ctr!=0) {

                        for(int line=0; line<n_lines; ++line) {
                            start = line*line_size;
                            end   = (line+1)*line_size;

                            for(int i=start ; i<end; ++i){
                                printf("|%10d",(int)(((double)nnz[i]/ctr)*100));
                            }
                            printf("\n");
                            printf("%10s:","");
                        }

                        start = mat->context->kacz_setting.active_threads - rem_lines;
                        end   = mat->context->kacz_setting.active_threads;

                        for(int i=start ; i<end; ++i){
                            printf("|%10d",(int)(((double)nnz[i]/ctr)*100));
                        }
                        printf("|%10d",100);
                    }


                }
            }
#ifdef GHOST_HAVE_OPENMP
        }
#endif
        printf("\n\n");
    }
    return GHOST_SUCCESS;
}
ghost_error kacz_analyze_print(ghost_sparsemat *mat)
{
    ghost_error ret = GHOST_SUCCESS;
    int nproc = 1;
    GHOST_CALL_RETURN(ghost_nrank(&nproc, mat->context->mpicomm));
    if(nproc > 1) {
        for(int i=0; i<nproc; ++i) {
           ret = kacz_analyze_print_single_proc(i,mat);
            MPI_Barrier(mat->context->mpicomm);
        }
    } else {
        ret = kacz_analyze_print_single_proc(0,mat);
    }
    return ret;
}
