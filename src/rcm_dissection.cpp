/*!GHOST_AUTOGEN CHUNKHEIGHT;BLOCKDIM1 */
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/math.h"
//#include "ghost/sell_kacz_plain_rcm_gen.h"
#include "ghost/omp.h"
#include "ghost/rcm_dissection.h"
#include <vector>
#include <iostream>

//returns the virtual column index; ie takes into account the permutation of halo elements also
#define virtual_col(col_idx)\
   (mat->context->flags & GHOST_PERM_NO_DISTINCTION)?( (col_ptr[col_idx]<mat->context->nrowspadded)?col_ptr[col_idx]:mat->context->perm_local->colPerm[col_ptr[col_idx]] ):col_ptr[col_idx]\


ghost_error find_transition_zone(ghost_sparsemat *mat, int n_threads)
{ 
if(n_threads>1)
{
   int lower_bw = 0;
   int upper_bw = 0;
   int nrows    = mat->nrows;
   int ncols    = mat->maxColRange+1;
   int n_zones  = 2* n_threads;//odd zone = even zone
   int total_bw = mat->bandwidth;
   double diagonal_slope = static_cast<double>(nrows)/ncols;
   int min_local_height  = static_cast<int>((diagonal_slope*total_bw));// + 1 ;//min. required height by transition zone
   int max_local_height  = static_cast<int>(static_cast<double>(nrows)/n_zones);
   int height            = std::max(min_local_height,max_local_height);

   int total_odd_rows    = n_threads*height;
   int total_even_rows   = nrows - total_odd_rows;
 
   mat->zone_ptr         = new int[n_zones+1]; 
   int* zone_ptr         = mat->zone_ptr; 
 
   bool flag_level = true; //flag for recursive call in current thread does not fit
   bool flag_lb    = true; //flag for load balancing
   
   //if overlap occurs restart with less threads
   if(total_even_rows<total_odd_rows){
	flag_level = false;
        delete[] zone_ptr;
	find_transition_zone(mat, n_threads-1);
    }
  
   int local_even_rows = static_cast<int>(static_cast<double>(total_even_rows)/n_threads); 

   //initial uniform splitting
   if(flag_level){
    	for(int i=0; i<n_threads; ++i){
		zone_ptr[2*i+1]   = (i+1)* local_even_rows + i*height -1 ;
	      	zone_ptr[2*i+2]   = zone_ptr[2*i+1] + height ;
	}
   	zone_ptr[0] = 0;
	zone_ptr[n_zones] = nrows;
    }

   //now we try for load_balancing 
   if(flag_level && flag_lb){
        //initial check: whether there is sufficient room to play
	if(max_local_height <= min_local_height){
	  	flag_lb = false;
		WARNING_LOG("RED BLACK SPLITTING : Load balancing not possible"); 
                
	} else {
       
            int nnz = mat->nnz;
            ghost_lidx *row_ptr = mat->sell->chunkStart;//for SELL-1-1
            int* load;
            load  =  new int[n_zones];
 
            int min_rows  = nrows*10; //just for checking
            int uniform_nnz = static_cast<int>(static_cast<double>(nnz)/n_zones);

	
            for(int i=0; i<n_zones; ++i){
                    int ctr_nnz = 0;
                    for(int row=zone_ptr[i]; row<zone_ptr[i+1]; ++row){
                        ctr_nnz += ( row_ptr[row+1] - row_ptr[row] );
                    }
            load[i] = static_cast<int>( static_cast<double>(uniform_nnz)/ctr_nnz * (zone_ptr[i+1] - zone_ptr[i]) );

            min_rows = std::min( load[i],min_rows);
            }
       

            //check whether this splitting is possible
            if(min_rows < min_local_height){
                flag_lb = false;
                WARNING_LOG("RED BLACK SPLITTING : Load balancing not possible");
            }
            else{
	      	zone_ptr[0] = 0;
               
		//assign zone with new loads
		for(int i=1;i<n_zones+1; ++i){
                    zone_ptr[i]=0;
                    for(int j=0; j<i; ++j){
                        zone_ptr[i] += load[j];
                    }
		}
		zone_ptr[n_zones] = nrows;

            }
	
            delete[] load;
    }
 
  }//flag_level && flag_lb
 if(flag_level)
     {
        mat->nzones = n_zones;
        mat->zone_ptr = zone_ptr;  
     }
      
}
else{
  mat->nzones = 2;
  mat->zone_ptr = new int[mat->nzones+1];
  mat->zone_ptr[0] = 0;
  mat->zone_ptr[1] = static_cast<int>(static_cast<double>(mat->nrows)/mat->nzones);
  mat->zone_ptr[2] = mat->nrows;
}

   return GHOST_SUCCESS;
}

//checks whether the partitioning was correct, its just for debugging
ghost_error checker_rcm(ghost_sparsemat *mat)
{
  ghost_lidx *row_ptr = mat->sell->chunkStart;
  ghost_lidx *col_ptr = mat->sell->col;

  ghost_lidx *upper_col_ptr = new ghost_lidx[mat->nzones];
  ghost_lidx *lower_col_ptr = new ghost_lidx[mat->nzones];

  for(int i=0; i<mat->nzones; ++i){
    upper_col_ptr[i] = 0;
    lower_col_ptr[i] = 0;
  }

  for(int i=0; i<mat->nzones;++i){
   for(int j=mat->zone_ptr[i]; j<mat->zone_ptr[i+1] ; ++j){
       ghost_lidx upper_virtual_col = virtual_col(row_ptr[j+1]-1);
       ghost_lidx lower_virtual_col = virtual_col(row_ptr[j]); 
       upper_col_ptr[i] = std::max(upper_virtual_col , upper_col_ptr[i]);
       lower_col_ptr[i] = std::max(lower_virtual_col , lower_col_ptr[i]);
   }
  }

  for(int i=0; i<mat->nzones-2; ++i){
   if(upper_col_ptr[i] >= lower_col_ptr[i+1] ){
     return  GHOST_ERR_RED_BLACK;
    }
  }

 return GHOST_SUCCESS;    
}

extern "C" ghost_error ghost_rcm_dissect(ghost_sparsemat *mat){	
    	
        int n_threads = 0;

//        int *zone_ptr;

#pragma omp parallel
 {
    #pragma omp single
    {
        n_threads = ghost_omp_nthread();

	find_transition_zone(mat, n_threads);
 
        int n_zones = mat->nzones;

	//TODO remove it only for debug phase
        std::cout<<"CHECKING whether splitting is corrext"<<std::endl;
        checker_rcm(mat);
        std::cout<<"CHECKING Finished"<<std::endl;      
        
         if(n_threads > n_zones/2){
            WARNING_LOG("RED BLACK splitting: Can't use all the threads , Usable threads = %d",n_zones/2);
         }

	mat->kacz_setting.active_threads = int( (double)(n_zones)/2);
     }    
  }  

   return GHOST_SUCCESS; 
}



//finds transition part of  RCM matrix , This function even finds whether there are 
//pure zones within the transitional part which we don't require and has much more
//overhead, compared to the currently used version
//This function(with some modification) can be used to predict rows which could 
//possibly false share 
/*std::vector<int> find_transition_zone_not_used(ghost_sparsemat *mat, int n_zones)
{
    if(n_zones >= 2)
    {
    std::vector<int> transition_zone;
    std::vector<int> compressed_transition;
     
    int n_t_zones = n_zones-1;
    int nrows     =  mat->nrows;

    int* lower_col_ptr = new int[n_t_zones];
    int* upper_col_ptr = new int[n_t_zones];
    
    for(int i=0; i!= n_t_zones; ++i){
        lower_col_ptr[i] = -1;
        upper_col_ptr[i] = -1;
    }
    
    ghost_lidx *row_ptr = mat->sell->chunkStart;
    ghost_lidx *col_ptr = mat->sell->col;
    
    //approximate
    ghost_lidx local_size = (int) (mat->nrows / n_zones);
    std::cout<<"local_size = "<<local_size<<std::endl;

    int *rhs_split;
    rhs_split = new int[n_t_zones];
    
    for(int i=0; i<n_t_zones; ++i){
        rhs_split[i] = (i+1)*local_size;
    }

    bool flag = false;
    
    for(int i=0; i<nrows; ++i){
        for(int k=0; k<n_t_zones; ++k){
            if(col_ptr[row_ptr[i]] < rhs_split[k] && col_ptr[row_ptr[i+1]-1] > rhs_split[k]) {
                lower_col_ptr[k] = std::max(col_ptr[row_ptr[i]],lower_col_ptr[k]);
                upper_col_ptr[k] = std::max(col_ptr[row_ptr[i+1]-1],upper_col_ptr[k]);
                //then we can't use this many threads, reduce threads and begin again
                if(!transition_zone.empty() &&( (transition_zone.back() == i)||check_transition_overlap(lower_col_ptr,upper_col_ptr,n_t_zones) ) ){
                    std::cout<<n_zones<<" threads not possible, Trying "<<n_zones-1<<std::endl;
                    flag = true;
                    transition_zone.clear();
                    //recursive call
                    transition_zone = find_transition_zone_not_used(mat,n_zones-1);
                    break;
                }
                else{
                    transition_zone.push_back(i);
                }
            }
            if(flag == true)
                break;
                
        }
    }

    delete[] lower_col_ptr;
    delete[] upper_col_ptr;
    return transition_zone;
    }
    else
        WARNING_LOG("RCM method for Kaczmarz cannot be done using multiple threads, Either matrix is too small or Bandwidth is large");
}
   
   
bool check_transition_overlap(int* lower_col_ptr, int* upper_col_ptr, int n_t_zones) {
    bool flag = false;
    for(int k=1; k!=n_t_zones; ++k){
        if( upper_col_ptr[k-1] != -1 && lower_col_ptr[k] != -1 && upper_col_ptr[k-1] > lower_col_ptr[k]) {
            flag = true;
            break;
        }
    }
return flag;
}
*/


/*
//this function  compresses transition zones, into [begin1; end1; begin2; end2]
std::vector<int> compress_vec(std::vector<int> transition_zone){

std::vector<int> compressed_transition;

    for(int i=0; i!= transition_zone.size(); ++i) {
        if(i==0 || i==transition_zone.size()-1){
            compressed_transition.push_back(transition_zone[i]);
        }
        else if(transition_zone[i-1] != transition_zone[i]-1){
            compressed_transition.push_back(transition_zone[i-1]);
            compressed_transition.push_back(transition_zone[i]);
        }
    }
return compressed_transition;
      
}

ghost_error adaptor_vec_ptr(int* ptr, std::vector<int> vec){
    ptr = new int [vec.size()];
    
    for(int i=0; i!=vec.size(); ++i){
        ptr[i] = vec[i];
    }
}

//not used
ghost_error ghost_rcm_dissect_not_used(ghost_sparsemat *mat, int n_zones){
    std::vector<int> transition_zone;
    std::vector<int> compressed_transition;
    int n_zones_out;
    transition_zone = find_transition_zone_not_used(mat, n_zones);
    std::cout<<"% of Transition_zone = "<< 100*(static_cast<double> (transition_zone.size())/mat->nrows)<<std::endl;
    compressed_transition = compress_vec(transition_zone);
    std::cout<<"Transition_zones are :\n";
    for(int i=0; i!=compressed_transition.size(); ++i)
        std::cout<<compressed_transition[i]<<std::endl;
    
    n_zones_out = compressed_transition.size()/2 + 1 ;
    std::cout<<"No. of threads to be used = "<< n_zones_out<<std::endl;
}*/                             
