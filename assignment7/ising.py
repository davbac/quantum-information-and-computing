import numpy as np
from composite import *

sigma_x = np.array([[0,1],[1,0]])
sigma_z = np.array([[1,0],[0,-1]])

def ising_hamiltonian(N,l):
    H = np.zeros(shape=(2**N,2**N))
    
    for i in range(N):
        H += pad(sigma_z, N,i) * l
    
    for i in range(N-1):
        H += pad(separable(sigma_x,sigma_x),N-1,i,dim=2)
    
    return H


if __name__=="__main__":
    """ ## use for timing checks with %timeit
    from sys import argv
    try:
        _, N, l = argv
        N = int(N)
        l = int(l)
    except:
        raise Exception("The script expects two command line arguments giving the number of subsystems and the strength of the field (N, lambda)")
    
    if N>14:
        raise Exception("Not enough memory available for the computation; you'd need {:e} bytes just to store the matrix using float32".format(4*2**(2*N)))
    
    if N>11: #execution time goes over the minute timescale at 12
        s = input("Are you sure? This may take a long time to execute; if in doubt try with a lower N. (y/n)")
        if s.lower()[0] == n:
            exit()
    
    vals, vecs = np.linalg.eigh(ising_hamiltonian(N, l))
    print(vals, vecs)
    """
    
    ## use for plotting spectrum
    import matplotlib.pyplot as plt
    k = 10 # number of levels to consider
    dl = 0.1 # spacing over lambda
    TOL = 1e-6 # tolerance for eogenvalue spacing
    
    lambdas = -1*np.arange(dl,3, dl)
    values = np.zeros(shape=(8,len(lambdas),k))
    
    
    for N in range(3,11):
        print(N)
        for i in range(len(lambdas)):
            vals,_ = np.linalg.eigh(ising_hamiltonian(N,lambdas[i]))
            #vals = np.unique(np.floor(vals/TOL).astype(int))*TOL # only keep unique values
            vals = sorted(vals)[:k]
            values[N-3,i,:len(vals)] = vals
            
    fig, ax = plt.subplots(nrows=2,ncols=4, sharex=True, sharey=True)
    
    for i in range(8):
        for j in range(k):
            ax[i//4][i%4].plot(lambdas,values[i,:,j], label=str(k))
        ax[i//4][i%4].set_title("N="+str(i+3))
    
    ax[1][0].set_xlabel("\u03bb")
    ax[1][0].set_ylabel("eigenvalues")
    plt.show()
    
