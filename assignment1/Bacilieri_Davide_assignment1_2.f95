program main
    implicit none
    
    integer*2 :: a, b
    integer*4 :: c, d
    real :: e,f
    real*8 :: g,h
    
    print *, "Testing number precision"
    print *, "integer*2 case"
    
    a=2000000   ! -fno-range-check needed for this case, as huge(a) is 32767
    b=1
    print *, a+b
    
    print *, "integer*4 case"
    c=2000000
    d=1
    print *, c+d
    
    print *, "real case"
    e=4*atan(1.)*1e32
    f=sqrt(2.)*1e21
    print *, e+f
    
    print *, "real*2 case"
    g=4*atan(1.)*1e32
    h=sqrt(2.)*1e21
    print *, g+h
    
    print *, "for reference: pi is ", e,g, "respectively in the two cases"
    
end program main
