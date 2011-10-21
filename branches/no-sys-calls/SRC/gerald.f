C
C     DOUBLE Werte --> Matrixeintraege
C
      subroutine SaveMatV( filnam , vals , size ,io_base_ptr)

      implicit none

      include 'mpif.h' 
      include 'my_mpi.inc'      ! --> myid , numprocs
      include 'GLOBDIM.inc'     ! wird nicht benötigt

      integer size
      real*8 vals(size)


c----------------
c     defines fuer filename 
c     enthaelt insbesondere den Pfad zum
c     jeweiligen parallelen FS
c
      include 'def_pio.inc' (siehe unten )
c
c----------------
      character*256 filnam

      integer FH                ! File-handle
      INTEGER (KIND=MPI_OFFSET_KIND) io_base_ptr, prec_v_8
      INTEGER (KIND=MPI_OFFSET_KIND) offset
      INTEGER STATUS( MPI_STATUS_SIZE)


      real*8 fsize             ! Groesse des Files in Byte

      integer i , error
      real*8 starttime, totaltime1, totaltime2, tmp

      call MPI_BARRIER(MPI_COMM_WORLD,ierr)      

c
c     check if pio has been activated
c
      if( .not. pioflag) then
         if(myid.eq.0) then
            write(*,*) ' Matrix ',filnam,' kann nicht gesichert werden !'
         endif
         return
      endif
c
c     construct filename including piopath
c
      write(ioname,889) iopath,filnam
      if(myid.eq.0) then
         write(*,*)
         write(*,'(a,a)') " Schreibe R8s nach --> ",ioname
         write(*,*)
      endif
C     
C     Oeffne File => SCHREIBEN
C     
      call MPI_File_open( MPI_COMM_WORLD ,  ioname , MPI_MODE_WRONLY + MPI_MODE_CREATE , 
     $     MPI_INFO_NULL , FH , error )
      call MPI_BARRIER(MPI_COMM_WORLD, error )      
      
      starttime = MPI_Wtime()
C     
C     Offset in Einheiten von etype
C     
c      dim_el_8=V_DIM_EL1 - 1
c      dim_ph_8=V_DIM_PH
c      prec_v_8=MY_VECPREC

      offset  =io_base_ptr * 8

c      write(*,'(i5,a,i12,x,i12)') myid," offset = ",offset,io_base_ptr

      call MPI_File_seek( FH , offset , MPI_SEEK_SET , error )
         
      call MPI_File_write( FH , vals , size , 
     $     MPI_DOUBLE_PRECISION , STATUS , error)
         

      
      totaltime1 = MPI_Wtime() - starttime
      call MPI_BARRIER(MPI_COMM_WORLD, error )      
      totaltime2 = MPI_Wtime() - starttime
      
      call MPI_File_close( FH , error )
      
      tmp=dble(size)
      call MPI_ALLREDUCE(tmp, fsize ,1 ,MPI_DOUBLE_PRECISION ,MPI_SUM,MPI_COMM_WORLD, ierr )
      
      fsize = fsize* 8

      if(myid.eq.0) write(*,'(i3,3(a,g20.12,x))')
     $     myid,' WRITE Time1 = ',totaltime1 ,
     $     ' Time2 = ',totaltime2,' Rate = ',
     $     (fsize/(dble(1024*1024)*totaltime2))
      
      call MPI_BARRIER(MPI_COMM_WORLD, error )      
      
      
      return
      end


C
C     INTEGER Werte --> MatrixINDIZES
C
      subroutine SaveMatI( filnam , ivals , size ,io_base_ptr)

      implicit none

      include 'mpif.h'
      include 'my_mpi.inc'
      include 'GLOBDIM.inc'

      integer size
      integer ivals(size)


c----------------
c     defines fuer filename 
c     enthaelt insbesondere den Pfad zum
c     jeweiligen parallelen FS
c
      include 'def_pio.inc'
c
c----------------
      character*256 filnam

      integer FH                ! File-handle
      INTEGER (KIND=MPI_OFFSET_KIND) io_base_ptr, prec_v_8
      INTEGER (KIND=MPI_OFFSET_KIND) offset
      INTEGER STATUS( MPI_STATUS_SIZE)


      real*8 fsize             ! Groesse des Files in Byte

      integer i , error
      real*8 starttime, totaltime1, totaltime2, tmp

      call MPI_BARRIER(MPI_COMM_WORLD,ierr)      

c
c     check if pio has been activated
c
      if( .not. pioflag) then
         if(myid.eq.0) then
            write(*,*) ' Matrix ',filnam,' kann nicht gesichert werden !'
         endif
         return
      endif
c
c     construct filename including piopath
c
      write(ioname,889) iopath,filnam
      if(myid.eq.0) then
         write(*,*)
         write(*,'(a,a)') " Schreibe I4s nach --> ",ioname
         write(*,*)
      endif
C     
C     Oeffne File => SCHREIBEN
C     
      call MPI_File_open( MPI_COMM_WORLD ,  ioname , MPI_MODE_WRONLY + MPI_MODE_CREATE , 
     $     MPI_INFO_NULL , FH , error )
      call MPI_BARRIER(MPI_COMM_WORLD, error )      
      
      starttime = MPI_Wtime()
C     
C     Offset in Einheiten von etype
C     
c      dim_el_8=V_DIM_EL1 - 1
c      dim_ph_8=V_DIM_PH
c      prec_v_8=MY_VECPREC

      offset  =io_base_ptr * 4  ! INTEGER

c      write(*,'(i5,a,i12,x,i12)') myid," offset = ",offset,io_base_ptr

      call MPI_File_seek( FH , offset , MPI_SEEK_SET , error )
         
      call MPI_File_write( FH , ivals , size , 
     $     MPI_INTEGER , STATUS , error)
         

      
      totaltime1 = MPI_Wtime() - starttime
      call MPI_BARRIER(MPI_COMM_WORLD, error )      
      totaltime2 = MPI_Wtime() - starttime
      
      call MPI_File_close( FH , error )
      
      tmp=dble(size)
      call MPI_ALLREDUCE(tmp, fsize ,1 ,MPI_DOUBLE_PRECISION ,MPI_SUM,MPI_COMM_WORLD, ierr )
      
      fsize = fsize* 4

      if(myid.eq.0) write(*,'(i3,3(a,g20.12,x))')
     $     myid,' WRITE Time1 = ',totaltime1 ,
     $     ' Time2 = ',totaltime2,' Rate = ',
     $     (fsize/(dble(1024*1024)*totaltime2))
      
      call MPI_BARRIER(MPI_COMM_WORLD, error )      
      
      
      return
      end
C
C     READ INTEGER Werte --> MatrixINDIZES
C
      subroutine ReadMatI( filnam , ivals , size ,io_base_ptr)

      implicit none

      include 'mpif.h'
      include 'my_mpi.inc'
      include 'GLOBDIM.inc'

      integer size
      integer ivals(size)


c----------------
c     defines fuer filename 
c     enthaelt insbesondere den Pfad zum
c     jeweiligen parallelen FS
c
      include 'def_pio.inc'
c
c----------------
      character*256 filnam

      integer FH                ! File-handle
      INTEGER (KIND=MPI_OFFSET_KIND) io_base_ptr
      INTEGER (KIND=MPI_OFFSET_KIND) offset
      INTEGER STATUS( MPI_STATUS_SIZE)


      real*8 fsize             ! Groesse des Files in Byte

      integer i , error
      real*8 starttime, totaltime1, totaltime2, tmp

      call MPI_BARRIER(MPI_COMM_WORLD,ierr)      

c
c     check if pio has been activated --> in def_pio.inc
c
      if( .not. pioflag) then
         if(myid.eq.0) then
            write(*,*) ' Matrix ',filnam,' kann nicht gelesen werden !'
         endif
         return
      endif
c
c     construct filename including iopath (aus def_pio.inc)
c
      write(ioname,889) iopath,filnam
      if(myid.eq.0) then
         write(*,*)
         write(*,'(a,a)') " Lese I4s von --> ",ioname
         write(*,*)
      endif
C     
C     Oeffne File => SCHREIBEN
C     
      call MPI_File_open( MPI_COMM_WORLD ,  ioname , MPI_MODE_RDONLY, 
     $     MPI_INFO_NULL , FH , error )

      call MPI_BARRIER(MPI_COMM_WORLD, error )      
      
      starttime = MPI_Wtime()
C     
C     Offset in Einheiten von etype -->  BYTE
C     
      offset  =io_base_ptr * 4  ! 4 --> MPI_INTEGER

      call MPI_File_seek( FH , offset , MPI_SEEK_SET , error )
         
      call MPI_File_read( FH , ivals , size , 
     $     MPI_INTEGER , STATUS , error)
         
      totaltime1 = MPI_Wtime() - starttime
      call MPI_BARRIER(MPI_COMM_WORLD, error )      
      totaltime2 = MPI_Wtime() - starttime
      
      call MPI_File_close( FH , error )
      
      tmp=dble(size)
      call MPI_ALLREDUCE(tmp, fsize ,1 ,MPI_DOUBLE_PRECISION ,MPI_SUM,MPI_COMM_WORLD, ierr )
      
      fsize = fsize* 4

      if(myid.eq.0) write(*,'(i3,3(a,g20.12,x))')
     $     myid,' READ Time1 = ',totaltime1 ,
     $     ' Time2 = ',totaltime2,' Rate = ',
     $     (fsize/(dble(1024*1024)*totaltime2))
      
      call MPI_BARRIER(MPI_COMM_WORLD, error )      
      
      
      return
      end


      def_pio.inc:


 889  format(a28,a225)
      parameter(iopath='/home/vault/unrz/unrz87/pio/')
      parameter(pioflag=.true.)


