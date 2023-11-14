module modmatrix
use debug
implicit none

! =================================================================
!
! This module implements a matrix type and the relevant 
!  single-matrix basic operations: trace and adjoint 
! Two utility functions are also defined: one for random
!  initialization, and one for printing out the matrix in a file
! 
! -----------------------------------------------------------------
!
! TYPES:
! 
! complex8_matrix
! 
!
!           Members
!           | shape (integer, dimension(2)) Stores the matrix shape
!
!           | vals (complex*8, dimension(:,:), allocatable) Stores the matrix element values
!
!
!           Interfaces
!           | .Adj. returns the adjoint matrix
!
!           | .Tr. returns the trace if the matrix is square
!
!           | .randInit. returns a random matrix with the selected shape
! 
!
!
!
! FUNCTIONS:
!
! c8mat_adj(mat)
!       Provides the functionality to the .Adj. interface
!
!           Inputs
!           | mat (complex8_matrix) The matrix of which to calculate the adjoint
!
!
!
!
! c8mat_init(shape)
!       Provides the functionality to the .randInit. interface
!
!           Inputs
!           | shape (integer, dimension(2)) Specifies the matrix shape to be used
!
!
!
!
! c8mat_tr(mat)
!       Provides the functionality to the .Tr. interface
!
!           Inputs
!           | mat (complex8_matrix) The matrix of which to calculate the trace
!
!
!
!
! SUBROUTINES:
!
! c8mat_print2file(mat, filename)
!       Print the matrix to file
!
!           Inputs
!           | mat (complex8_matrix) The matrix to be printed
! 
!           | filename (character) The file in which to write the matrix (used in append mode)
!

type complex8_matrix
    ! store the shape of the matrix, to facilitate checks
    integer, dimension(2) :: shape
    
    ! store the matrix element values
    complex*8, dimension(:,:), allocatable :: vals
    
end type
    
    interface operator(.Adj.)
        module procedure c8mat_adj
    end interface
    
    interface operator(.Tr.)
        module procedure c8mat_tr
    end interface
    
    interface operator(.randInit.)
        module procedure c8mat_init
    end interface
contains

function c8mat_adj(mat) result(adj)
    type(complex8_matrix), intent(in) :: mat
    type(complex8_matrix) :: adj 
    
    adj%shape(1) = mat%shape(2)
    adj%shape(2) = mat%shape(1)
    
    allocate(adj%vals(adj%shape(1), adj%shape(2)))
    
    adj%vals = conjg(transpose(mat%vals))
end function

function c8mat_tr(mat) result(tr)
    type(complex8_matrix), intent(in) :: mat
    complex*8 :: tr
    integer :: i
    character(len=35) :: str
    
    if (mat%shape(1) /= mat%shape(2)) then
        write(str, "(a22,2i4)") "matrix is not square: ", mat%shape(1), mat%shape(2)
        call checkpoint(str, .true.)
        error stop
    end if
    
    tr = complex(0d0, 0d0) ! initialize
    
    do i=1,mat%shape(1)
        tr = tr + mat%vals(i,i)
    end do
end function

subroutine c8mat_print2file(mat, filename)
    type(complex8_matrix), intent(in) :: mat
    character (len=*), intent(in) :: filename
    character (len=32):: fmt
    character (len=(21*mat%shape(2))) :: str
    integer :: i
    
    ! first write the format string, the pattern must be reproduced a number of
    !  times (at least) equal to the column count
    write(fmt, "(a,i4,27a)") '(', mat%shape(2),'(SP,F7.4,SP,F7.4,"i", 6x))'
    do i=1,mat%shape(1)
        ! for each row, append the values to file
        write (str,fmt) mat%vals(i,:)
        call checkpoint(text=str, debug=.true., filename=filename)
    end do
end subroutine

function c8mat_init(shape) result(mat)
    integer, dimension(2), intent(in) :: shape
    type(complex8_matrix) :: mat
    real*8, dimension(shape(1),shape(2)) :: a,b
    mat%shape = shape
    allocate(mat%vals(shape(1), shape(2)))
    ! changing here the function used to generate real, imag part will impact the complex number
    !  distribution; as it is, they are confined in the (0,1) range for both real and imag part
    call random_number(a)
    call random_number(b)
    mat%vals = cmplx(a,b)
end function
end module
