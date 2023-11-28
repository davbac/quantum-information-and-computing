import scipy.sparse as sps
import scipy.sparse.linalg as las
import numpy as np
from scipy.special import hermite

# helper function to get the factorial of a number
### np.fact is not stable as its return type is the same as the input, ie short int
##### Using native python works around thanks to automatic retyping of the variable to hold the value
def fact(x):
    res = 1
    for i in range(2,x+1):
        res*=i
    return res


def harmonic_eig(N=500, j=40, eigval_max_dev=None, eigvec_max_dev=None, K_order=1):
    # N is the number of points per unit distance
    # j gives the number of eigenvalues and eigenfunctions to compute
    
    # take some standard values that work well
    omega = 2
    interval = (-5, 5) #left, right limit in unit distance ####### needs int values as it is for now
    
    npoints = N*(interval[1]-interval[0])+1 # total number of points
    
    # the scipy sparse algorithms take care of both the storage (via the diags() function)
    #  and the diagonalization (via the linalg.eigsh() function.
    ### diags() takes a list of diagonals and a list of the indices at which they are to be placed, 
    ###  with 0 the principal, the range being from -N+1 to N-1
    ##### each diagonal should be a np.array with length = N-k where k is its index
    ### linalg.eigsh() only works for hermitian matrices;
    ##### to work with any matrix, use linalg.eigs()
    
    
    # Kinetic part
    if K_order == 1:
        K = sps.diags([2*np.ones(npoints), -1*np.ones(npoints-1), -1*np.ones(npoints-1)], [0,+1,-1]) *(N**2)/2
    elif K_order == 2:
        K = sps.diags([1.25*np.ones(npoints), -(4/3)*np.ones(npoints-1), -(4/3)*np.ones(npoints-1), (1/12)*np.ones(npoints-2), (1/12)*np.ones(npoints-2)], [0,+1,-1,+2,-2]) *(N**2)/2
    
    # points used to calculate the potential
    x = np.linspace(*interval, npoints)
    
    # Potential part
    V = sps.diags([(omega**2)*0.5*x**2], [0])
    # V = sps.diags([U(x)], [0]) ## given a generic U(x:np.array) -> np.array
    
    
    H = K+V
    
    # find numeric eigenvalues & eigenvectors
    vals, vecs = las.eigsh(H, k=j, which="SA")
    
    #print(np.diff(vals), vals[0]) #spacings, first value
    
    # the eigenvectors are not normalized -> calculate the norms and renormalize
    norms = [sum(vecs[:,i]**2)/(npoints) for i in range(j)]
    for i in range(j):
        vecs[:,i]/=norms[i]**0.5
    
    
    
    # 1/sqrt(2^n n!) (omega/pi)^1/4 e^(-omega/2 x^2) hermite(n)(sqrt(omega)*x)
    exact = lambda n: lambda x: (((10/(fact(n)*(2**n)))**0.5)*((omega/np.pi)**0.25)*np.exp(-0.5*omega*x**2)*hermite(n)(x*omega**0.5))
    # needed to add a sqrt(10) term to get normalized functions, might be due to
    # some difference in the definition of the hermite functions
    
    
    # different printing options:
    # - if there is no max deviation for both values and vectors, print all deviations
    # - else, only print the value of n at which either threshold is passed
    
    if eigval_max_dev is None and eigvec_max_dev is None:
        print("n\terror on eigenvalue\t\tmax error on square eigenfunction")
    
    for i in range(j):
        f=exact(i)
        max_diff = max(np.abs(f(x)**2 - vecs[:,i]**2))
        val_diff = np.abs(vals[i]- (omega*(i+0.5)))
        
        if eigval_max_dev is None and eigvec_max_dev is None:
            print(i, val_diff, "", max_diff, sep="\t")
        elif eigval_max_dev is not None and val_diff>eigval_max_dev:
            print("First function with a deviation from the exact eigenvalue larger than "+
                  str(eigval_max_dev)+" is for n="+str(i))
            break
        elif eigvec_max_dev is not None and max_diff>eigvec_max_dev:
            print("First function with a deviation from the exact eigenvector larger than "+
                  str(eigvec_max_dev)+" is for n="+str(i))
            break
    
    # added return to ease plotting
    return x, vecs


# example of usage
if __name__=="__main__":
    import matplotlib.pyplot as plt
    x,y=harmonic_eig(j=4)
    plt.plot(x,y)
