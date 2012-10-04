      subroutine FortranJDS(nnp, zmax,anznnel, resvec, locvec, jds_ptr, 
     $     cmatrx_jds , colind_jds, mode, blocklen)

      implicit none

C===========================================================
      integer mode              ! 0-plain ; 1-unroll2 ; 3-blocking


      integer nnp               ! TOM: res->nRows
      integer zmax              ! TOM: jd->nDiags
      integer anznnel           ! TOM: 
      
      double precision resvec(nnp)
                                ! TOM: res->val[]
      double precision locvec(nnp)
                                ! TOM: vec->val[]

      integer jds_ptr(zmax+1)   ! jd->diagOffset
      
      double precision cmatrx_jds(anznnel) ! 
                                !jd->val[]

      integer colind_jds(anznnel) ! 
                                ! jd->col[]
      
C===========================================================

      integer i , diag , diagLen , offset, offset1, offset2
      integer diagLen1, diagLen2
      integer zmaxSTART
      integer blocklen , ib , block_start , block_end
      integer mynumber
      integer OMP_GET_THREAD_NUM
      !integer OMP_GET_NUM_THREADS

      goto 99

C======================================== 
C=== Plain old sparse MVM with no tricks
C======================================== 

      if(mode .eq.1) then
         
!$OMP PARALLEL private(diag,diagLen,offset,i)

!$OMP DO
         do i = 1, nnp
            resvec(i) = 0.0
         enddo

         
         do diag=1, zmax
            diagLen = (jds_ptr(diag+1)-jds_ptr(diag))
            offset  = jds_ptr(diag)

!$OMP DO
!DEC$ IVDEP
!DEC$ VECTOR ALWAYS
            do i=1, diagLen

               resvec(i) = resvec(i)+cmatrx_jds(offset+i)*
     $              locvec(colind_jds(offset+i))


            enddo

         enddo
!$OMP END PARALLEL 

      goto 99

      endif

C======================================== 
C=== unrolling by 2 (horizontally)
C======================================== 

      if(mode .eq. 2) then


!$OMP PARALLEL private(mynumber,diag,diagLen,offset1,offset2
!$OMP& ,i,zmaxSTART)

!$      mynumber = OMP_GET_THREAD_NUM()


!$OMP DO
         do i = 1, nnp
            resvec(i) = 0.0
         enddo

         zmaxSTART= zmax/2
         zmaxSTART= 2*zmaxSTART



         do diag=1, zmaxSTART , 2

            diagLen = min( (jds_ptr(diag+1)-jds_ptr(diag)) , 
     $           (jds_ptr(diag+2)-jds_ptr(diag+1)) )
            offset1 = jds_ptr(diag) 
            offset2 = jds_ptr(diag+1) 

!$OMP DO
!DEC$ IVDEP
!DEC$ VECTOR ALWAYS
            do i=1, diagLen

               resvec(i) = resvec(i)+cmatrx_jds(offset1+i)*
     $              locvec(colind_jds(offset1+i))

               resvec(i) = resvec(i)+cmatrx_jds(offset2+i)*
     $              locvec(colind_jds(offset2+i))

            enddo
!$OMP SINGLE
            offset1 = jds_ptr(diag) 
            do i=(diagLen+1),(jds_ptr(diag+1)-jds_ptr(diag))

               resvec(i) = resvec(i)+cmatrx_jds(offset1+i)*
     $              locvec(colind_jds(offset1+i))
            enddo
!$OMP END SINGLE
 
         enddo ! End OMP_PARALLEL_DO

         do diag=zmaxSTART+1, zmax
            diagLen = (jds_ptr(diag+1)-jds_ptr(diag))
            offset  = jds_ptr(diag)

!$OMP DO
!DEC$ IVDEP
!DEC$ VECTOR ALWAYS
            do i=1, diagLen
               
               resvec(i) = resvec(i)+cmatrx_jds(offset+i)*
     $              locvec(colind_jds(offset+i))
 
             
            enddo
            
         enddo


!$OMP END PARALLEL 

      goto 99

      endif

C======================================== 
C=== blocked MVM
C======================================== 

      if(mode .eq. 3) then

!$OMP PARALLEL DO private(block_start,block_end,i,diag,diagLen,offset) 
!$OMP& schedule(runtime)   
         do ib=1, nnp , blocklen
            
            block_start = ib
            block_end   = min(ib+blocklen-1, nnp)

            do i=block_start, block_end
               resvec(i) = 0.d0
            enddo

            do diag=1, zmax

               diagLen = (jds_ptr(diag+1)-jds_ptr(diag))
               offset  = jds_ptr(diag)
         
               if(diagLen .ge. block_start) then

!DEC$ IVDEP
!DEC$ VECTOR ALWAYS
!DEC$ VECTOR ALIGNED
                  do i=block_start, min(block_end,diagLen)

                     resvec(i) = resvec(i)+cmatrx_jds(offset+i)*
     $                    locvec(colind_jds(offset+i))


                  enddo

               endif

            enddo

         enddo
         goto 99
      endif

C======================================== 
C=== blocked MVM with 2-way unrolling
C======================================== 

      if(mode .eq. 4) then

!$OMP PARALLEL DO private(block_start,block_end,i,diag,diagLen,offset)
!$OMP& schedule(runtime)   
         do ib=1, nnp , blocklen
            
            block_start = ib
            block_end   = min(ib+blocklen-1, nnp)

            do i=block_start, block_end
               resvec(i) = 0.d0
            enddo
            
            zmaxSTART= zmax/2
            zmaxSTART= 2*zmaxSTART

 
            do diag=1, zmaxSTART, 2
               ! are there actually two diagonals available?
               offset1 = jds_ptr(diag)-1
               offset2 = jds_ptr(diag+1)-1
               diagLen1 = min(blocklen,(jds_ptr(diag+1)-
     $              jds_ptr(diag)-(block_start-1)))
               diagLen2 = min(blocklen,(jds_ptr(diag+2)-
     $              jds_ptr(diag+1)-(block_start-1)))
               diagLen = min(diagLen1,diagLen2)
               ! if yes, do 2-way unrolling
               if(diagLen .gt. 0) then

!DEC$ IVDEP
!DEC$ VECTOR ALWAYS
!DEC$ VECTOR ALIGNED
                  do i=block_start, block_start+diagLen-1

                     resvec(i) = resvec(i)+cmatrx_jds(offset1+i)*
     $                    locvec(colind_jds(offset1+i))
                     resvec(i) = resvec(i)+cmatrx_jds(offset2+i)*
     $                    locvec(colind_jds(offset2+i))

                  enddo
                  ! peeled-off iterations
!DEC$ IVDEP
!DEC$ VECTOR ALWAYS
!DEC$ VECTOR ALIGNED
                  do i=block_start+diagLen,block_start+diagLen1-1
                     resvec(i) = resvec(i)+cmatrx_jds(offset1+i)*
     $                    locvec(colind_jds(offset1+i))
                  enddo
               else
                  ! only 1 diagonal left
                  if(diagLen1 .gt. 0) then
!DEC$ IVDEP
!DEC$ VECTOR ALWAYS
!DEC$ VECTOR ALIGNED
                     do i=block_start, block_start+diagLen1-1
                        resvec(i) = resvec(i)+cmatrx_jds(offset1+i)*
     $                       locvec(colind_jds(offset1+i))
                     enddo
                  endif
               endif
               
            enddo
            ! remainder loop
            do diag=zmaxSTART+1,zmax
               diagLen =  min(blocklen,(jds_ptr(diag+1)-
     $              jds_ptr(diag)-(block_start-1)))
               if(diagLen .gt. 0) then
                  offset = jds_ptr(diag)-1
!DEC$ IVDEP
!DEC$ VECTOR ALWAYS
!DEC$ VECTOR ALIGNED
                  do i=block_start, block_start+diagLen-1
                     resvec(i) = resvec(i)+cmatrx_jds(offset+i)*
     $                    locvec(colind_jds(offset+i))
                  enddo
               endif
            enddo
            
         enddo
         goto 99
      endif

      write(*,*) " mode = ",mode," fuer JDS nicht implementiert "
      stop

 99   continue

      return
      end

      subroutine FortranCRSc(nnp, anznnel, resvec , locvec, 
     $     cmatrx_crs, index_crs, rowoffset )
      implicit none

      integer nnp               ! TOM: res->nRows
      integer anznnel           ! TOM: cr->nEnts

      double complex resvec(nnp)
                                ! TOM: res->val[]
      double complex locvec(nnp)
                                ! TOM: vec->val[]

      double complex cmatrx_crs(anznnel)
                                ! cr->val
      integer index_crs(anznnel)       ! cr->col
      integer rowoffset(nnp+1)       ! cr->rowOffset
      
      integer i,j,start,end

      double complex tmp
   
      !write(*,*) 'in Fortran_CRS'

!$OMP PARALLEL DO private(tmp, start, end) schedule(runtime)
      do i=1, nnp

         tmp   = 0.d0
         start = rowoffset(i)+1
         end   = rowoffset(i+1)

!DEC$ VECTOR ALWAYS
!DEC$ IVDEP
         do j= start , end
            tmp = tmp + cmatrx_crs(j) * locvec(index_crs(j))
         enddo

         resvec(i)=tmp

      enddo
      
      return
      end

      subroutine FortranCRSAXPYc(nnp, anznnel, resvec , locvec, 
     $     cmatrx_crs, index_crs, rowoffset )
      implicit none

      integer nnp               ! TOM: res->nRows
      integer anznnel           ! TOM: cr->nEnts

      double complex resvec(nnp)
                                ! TOM: res->val[]
      double complex locvec(nnp)
                                ! TOM: vec->val[]

      double complex cmatrx_crs(anznnel)
                                ! cr->val
      integer index_crs(anznnel)       ! cr->col
      integer rowoffset(nnp+1)       ! cr->rowOffset
      
      integer i,j,start,end

      double complex tmp
   
      !write(*,*) 'in Fortran_CRS'

!$OMP PARALLEL DO private(tmp, start, end) schedule(runtime)
      do i=1, nnp

         tmp   = 0.d0
         start = rowoffset(i)+1
         end   = rowoffset(i+1)

!DEC$ VECTOR ALWAYS
!DEC$ IVDEP
         do j= start , end
            tmp = tmp + cmatrx_crs(j) * locvec(index_crs(j))
         enddo

         resvec(i) = resvec(i) + tmp

      enddo
      
      return
      end
     
      subroutine FortranCRS(nnp, anznnel, resvec , locvec, 
     $     cmatrx_crs, index_crs, rowoffset )
      implicit none

      integer nnp               ! TOM: res->nRows
      integer anznnel           ! TOM: cr->nEnts

      double precision resvec(nnp)
                                ! TOM: res->val[]
      double precision locvec(nnp)
                                ! TOM: vec->val[]

      double precision cmatrx_crs(anznnel)
                                ! cr->val
      integer index_crs(anznnel)       ! cr->col
      integer rowoffset(nnp+1)       ! cr->rowOffset
      
      integer i,j,start,end

      double precision tmp
   
      !write(*,*) 'in Fortran_CRS'

!$OMP PARALLEL DO private(tmp, start, end) schedule(runtime)
      do i=1, nnp

         tmp   = 0.d0
         start = rowoffset(i)+1
         end   = rowoffset(i+1)

!DEC$ VECTOR ALWAYS
!DEC$ IVDEP
         do j= start , end
            tmp = tmp + cmatrx_crs(j) * locvec(index_crs(j))
         enddo

         resvec(i)=tmp

      enddo
      
      return
      end

      subroutine FortranCRSAXPY(nnp, anznnel, resvec , locvec, 
     $     cmatrx_crs, index_crs, rowoffset )
      implicit none

      integer nnp               ! TOM: res->nRows
      integer anznnel           ! TOM: cr->nEnts

      double precision resvec(nnp)
                                ! TOM: res->val[]
      double precision locvec(nnp)
                                ! TOM: vec->val[]

      double precision cmatrx_crs(anznnel)
                                ! cr->val
      integer index_crs(anznnel)       ! cr->col
      integer rowoffset(nnp+1)       ! cr->rowOffset
      
      integer i,j,start,end

      double precision tmp
   
      !write(*,*) 'in Fortran_CRS'

!$OMP PARALLEL DO private(tmp, start, end) schedule(runtime)
      do i=1, nnp

         tmp   = 0.d0
         start = rowoffset(i)+1
         end   = rowoffset(i+1)

!DEC$ VECTOR ALWAYS
!DEC$ IVDEP
         do j= start , end
            tmp = tmp + cmatrx_crs(j) * locvec(index_crs(j))
         enddo

         resvec(i) = resvec(i) + tmp

      enddo
      
      return
      end

      subroutine FortranCRScf(nnp, anznnel, resvec , locvec, 
     $     cmatrx_crs, index_crs, rowoffset )
      implicit none

      integer nnp               ! TOM: res->nRows
      integer anznnel           ! TOM: cr->nEnts

      complex resvec(nnp)
                                ! TOM: res->val[]
      complex locvec(nnp)
                                ! TOM: vec->val[]

      complex cmatrx_crs(anznnel)
                                ! cr->val
      integer index_crs(anznnel)       ! cr->col
      integer rowoffset(nnp+1)       ! cr->rowOffset
      
      integer i,j,start,end

      complex tmp
   
      !write(*,*) 'in Fortran_CRS'

!$OMP PARALLEL DO private(tmp, start, end) schedule(runtime)
      do i=1, nnp

         tmp   = 0.d0
         start = rowoffset(i)+1
         end   = rowoffset(i+1)

!DEC$ VECTOR ALWAYS
!DEC$ IVDEP
         do j= start , end
            tmp = tmp + cmatrx_crs(j) * locvec(index_crs(j))
         enddo

         resvec(i)=tmp

      enddo
      
      return
      end

      subroutine FortranCRSAXPYcf(nnp, anznnel, resvec , locvec, 
     $     cmatrx_crs, index_crs, rowoffset )
      implicit none

      integer nnp               ! TOM: res->nRows
      integer anznnel           ! TOM: cr->nEnts

      complex resvec(nnp)
                                ! TOM: res->val[]
      complex locvec(nnp)
                                ! TOM: vec->val[]

      complex cmatrx_crs(anznnel)
                                ! cr->val
      integer index_crs(anznnel)       ! cr->col
      integer rowoffset(nnp+1)       ! cr->rowOffset
      
      integer i,j,start,end

      complex tmp
   
      !write(*,*) 'in Fortran_CRS'

!$OMP PARALLEL DO private(tmp, start, end) schedule(runtime)
      do i=1, nnp

         tmp   = 0.d0
         start = rowoffset(i)+1
         end   = rowoffset(i+1)

!DEC$ VECTOR ALWAYS
!DEC$ IVDEP
         do j= start , end
            tmp = tmp + cmatrx_crs(j) * locvec(index_crs(j))
         enddo

         resvec(i) = resvec(i) + tmp

      enddo
      
      return
      end
      
      subroutine FortranCRSf(nnp, anznnel, resvec , locvec, 
     $     cmatrx_crs, index_crs, rowoffset )
      implicit none

      integer nnp               ! TOM: res->nRows
      integer anznnel           ! TOM: cr->nEnts

      real resvec(nnp)
                                ! TOM: res->val[]
      real locvec(nnp)
                                ! TOM: vec->val[]

      real cmatrx_crs(anznnel)
                                ! cr->val
      integer index_crs(anznnel)       ! cr->col
      integer rowoffset(nnp+1)       ! cr->rowOffset
      
      integer i,j,start,end

      real tmp
   
      !write(*,*) 'in Fortran_CRS'

!$OMP PARALLEL DO private(tmp, start, end) schedule(runtime)
      do i=1, nnp

         tmp   = 0.d0
         start = rowoffset(i)+1
         end   = rowoffset(i+1)

!DEC$ VECTOR ALWAYS
!DEC$ IVDEP
         do j= start , end
            tmp = tmp + cmatrx_crs(j) * locvec(index_crs(j))
         enddo

         resvec(i)=tmp

      enddo
      
      return
      end

      subroutine FortranCRSAXPYf(nnp, anznnel, resvec , locvec, 
     $     cmatrx_crs, index_crs, rowoffset )
      implicit none

      integer nnp               ! TOM: res->nRows
      integer anznnel           ! TOM: cr->nEnts

      real resvec(nnp)
                                ! TOM: res->val[]
      real locvec(nnp)
                                ! TOM: vec->val[]

      real cmatrx_crs(anznnel)
                                ! cr->val
      integer index_crs(anznnel)       ! cr->col
      integer rowoffset(nnp+1)       ! cr->rowOffset
      
      integer i,j,start,end

      real tmp
   
      !write(*,*) 'in Fortran_CRS'

!$OMP PARALLEL DO private(tmp, start, end) schedule(runtime)
      do i=1, nnp

         tmp   = 0.d0
         start = rowoffset(i)+1
         end   = rowoffset(i+1)

!DEC$ VECTOR ALWAYS
!DEC$ IVDEP
         do j= start , end
            tmp = tmp + cmatrx_crs(j) * locvec(index_crs(j))
         enddo

         resvec(i) = resvec(i) + tmp

      enddo
      
      return
      end

