import scipy.sparse as sps
import scipy.sparse.linalg as las
import numpy as np

def harmonic_eig(npoints=5001, j=40, K_order=1, interval=(-5,5), omega=2):
    # N is the number of points per unit distance
    # j gives the number of eigenvalues and eigenfunctions to compute
    
    # points per unit length, -1 is to acount for the one on the border
    N = (npoints-1) / (interval[1]-interval[0])
    
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
    norms = [sum(vecs[:,i]**2)/N for i in range(j)]
    for i in range(j):
        vecs[:,i]/=norms[i]**0.5
    
    return x, vecs


# example of usage
if __name__=="__main__":
    import matplotlib.pyplot as plt
    x,y=harmonic_eig(j=4)
    plt.plot(x,y)
