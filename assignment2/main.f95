program main
    use modmatrix
    implicit none
    
    type(complex8_matrix) :: mat , adj
    
    mat = .randInit.((/3,3/))
    
    adj = .Adj.mat
    
    print '(9(SP,F7.4,SP,F7.4,"i", 6x))' , mat%vals
    print '(9(SP,F7.4,SP,F7.4,"i", 6x))' , adj%vals
    print *, .Tr.mat
    print *, .Tr.adj
    
    call c8mat_print2file(mat, "mat.txt")
    
end program
