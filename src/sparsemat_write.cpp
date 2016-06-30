  #include "ghost/config.h"
  #include "ghost/types.h"
  #include "ghost/util.h"
  #include "ghost/math.h"
  #include <fstream>
  #include <string>
  #include <sstream>
  #include "ghost/locality.h"
 
  ghost_error sparsemat_write(ghost_sparsemat *A,char* name)
  {
        GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
        int me;
        ghost_rank(&me, A->context->mpicomm);

   
        std::string base_name(name);
        std::stringstream fileName;  
        fileName<<base_name <<".mtx";
        std::fstream file;
        file.open( fileName.str().c_str() ,std::fstream::out);
        file<<"%";
  	file<<"%MatrixMarket matrix coordinate real general\n";
  	file<<"%MatrixMarketm file generated by ghost sparsemat_write.cpp\n";
	
	file<<A->nrows<<" "<<A->ncols<<" "<<A->nnz<<"\n";

  	ghost_sell *sell_mat = SELL(A);
        double *mval = (double*) sell_mat->val; 

  	for (int row=0; row<A->nrows; ++row) {
        	ghost_lidx idx =  sell_mat->chunkStart[row];
        	for (int j=0; j<sell_mat->rowLen[row]; ++j) {
                	file<<row+1<<" "<<(int)sell_mat->col[idx+j]+1<<" "<<mval[idx+j]<<"\n";
         	}
  	}
      
       GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
       return GHOST_SUCCESS;

 }


