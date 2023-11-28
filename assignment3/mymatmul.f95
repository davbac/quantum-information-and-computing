program main
    use debug
    implicit none;
    character (len=4) :: rstring, dstring
    integer :: runs
    integer :: d 
    integer i,j
    real :: t, t1, t2
    
    if (command_argument_count()>=1) then 
        call get_command_argument(1, dstring)
        !call checkpoint(dstring, .true.)
        read(dstring,*) d
        if (command_argument_count()>=2) then 
            call get_command_argument(2, rstring)
            !call checkpoint(rstring, .true.)
            read(rstring,*) runs
        else
            runs=1
        end if 
    else 
        stop
    end if
    
    
    do j=1,runs
!           get time in seconds
        call get_times(d, t, t1, t2)
        write (*,*) t, t1, t2
    end do

contains

! =================================================================
!
! This module implements two matrix-matrix multiplication functions
!  and a function to get the timings for those and the builtin
!  matmul function given a matrix size
!
! -----------------------------------------------------------------
!
! SUBROUTINES:
!
! mymatmul1(m1, m2, m3)
!
!
!           Inputs
!           | m1, m2 (matrices): the two matrices to multiply
!
!           Outputs
!           | m3 (matrix): the output matrix
!
!
!
!
! mymatmul2(m1, m2, m3)
!
!
!           Inputs
!           | m1, m2 (matrices): the two matrices to multiply
!
!           Outputs
!           | m3 (matrix): the output matrix
!
!
!
!
! get_times(s, t, t1, t2)
!
!
!           Inputs
!           | s (integer) matrix size
!
!           Outputs
!           | t (real) time taken by matmul
!   
!           | t1 (teal) time taken by mymatmul1
!
!           | t2 (real) time taken by mymatmul2


subroutine mymatmul1 (m1, m2, m3)
    real, dimension(:,:), intent(in) :: m1, m2
    real, dimension(:,:), intent(out):: m3
    integer :: i,j,k
    integer, dimension(2) :: shape1, shape2, shape3
    character (len=40) :: str
    shape1=shape(m1)
    shape2=shape(m2)
    shape3=shape(m3)
    
    ! first check that the height of m2 equals the width of m1, then that the target matrix dimensions are the correct ones
    if (shape2(1) /= shape1(2) .or. shape3(1) /= shape1(1) .or. shape3(2) /= shape2(2)) then
        call checkpoint(text="dimension mismatch", debug=.true.)
        write (str, '(I4, I4,"   --",I4,I4,"   --",I4,I4)') shape2(1), shape1(2), shape3(1), shape1(1), shape3(2), shape2(2)
        call checkpoint(text=str, debug=.true.)
        error stop
    end if
    
    ! perform the row-by-column multiplication
    do i=1,shape1(1)
        do j=1,shape2(2)
            m3(i,j)=0 !shouldn't be needed, jic memory isnt clean
            do k=1,shape1(2)
                m3(i,j)=m3(i,j)+m1(i,k)*m2(k,j)
            end do
        end do
    end do
    
    
    
end subroutine mymatmul1

subroutine mymatmul2 (m1, m2, m3)
    real, dimension(:,:), intent(in) :: m1, m2
    real, dimension(:,:), intent(out):: m3
    integer :: i,j,k
    integer, dimension(2) :: shape1, shape2, shape3
    character (len=40) :: str
    shape1=shape(m1)
    shape2=shape(m2)
    shape3=shape(m3)
    
    ! first check that the height of m2 equals the width of m1, then that the target matrix dimensions are the correct ones
    if (shape2(1) /= shape1(2) .or. shape3(1) /= shape1(1) .or. shape3(2) /= shape2(2)) then
        call checkpoint(text="dimension mismatch", debug=.true.)
        write (str, '(I4, I4,"   --",I4,I4,"   --",I4,I4)') shape2(1), shape1(2), shape3(1), shape1(1), shape3(2), shape2(2)
        call checkpoint(text=str, debug=.true.)
        error stop
    end if
    
    ! add cycle to clean memory
    do i=1,shape3(1)
        do j=1,shape3(2)
            m3(i,j)=0
        end do
    end do
    
    ! perform the row-by-column multiplication
    do k=1,shape1(2)
        do i=1,shape1(1)
            do j=1,shape2(2)            
                m3(i,j)=m3(i,j)+m1(i,k)*m2(k,j)
            end do
        end do
    end do
    
    
    
end subroutine mymatmul2

subroutine get_times (s, t, t1, t2) 
    integer, intent(in) :: s
    real, intent(out) :: t,t1,t2
    
    ! create the matrices that will be used for multiplication
    real, dimension(s,s) :: m1
    real, dimension(s,s) :: m2
    real, dimension(s,s) :: m3, m4, m5
    ! the result matrices are stored in different variables to allow
    ! checking the result of the different subroutines
    
    real :: start, finish
    
    ! fill the matrices with random numbers
    call random_number(m1)
    call random_number(m2)
    
    ! time the different functions
    call cpu_time(start)
    call mymatmul1(m1, m2, m3) 
    call cpu_time(finish)
    
    t1 = finish-start
    
    call cpu_time(start)
    m4 = matmul(m1,m2)
    call cpu_time(finish)
    
    t = finish-start
        
    call cpu_time(start)
    call mymatmul2(m1,m2,m5)
    call cpu_time(finish)
    
    t2 = finish-start
    
end subroutine

end program main
