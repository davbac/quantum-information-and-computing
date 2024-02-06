import numpy as np
from composite import *

sigma_x = np.array([[0,1],[1,0]])
sigma_z = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])

def ising_hamiltonian(N,l):
    H = np.zeros(shape=(2**N,2**N))
    
    for i in range(N):
        H += pad(sigma_z, N,i) * l
    
    for i in range(N-1):
        H += pad(separable(sigma_x,sigma_x),N-1,i,dim=2)
    
    return H


def realspace_rg(n0, n_iter, l=-1):
    H = ising_hamiltonian(n0, l)
    projectors = []
    
    m = 2**n0 # keep a number of eigenvectors equal to the initial condition
    
    HintA = pad(sigma_x,n0,n0-1)
    HintB = pad(sigma_x,n0,0)
    I_n0  = separable(*([I]*n0))
    
    gvals = []
    
    for i in range(n_iter):        
        new_H = pad(H,2,0) + pad(H,2,1) + separable(HintA, HintB)
        vals, vecs = np.linalg.eigh(new_H)
        projectors.append(vecs[:,:m])
        
        #H = projectors[-1].T.conj() @ new_H @ projectors[-1]
        H = np.diag(vals[:m])
        gvals.append(vals[0])
        """
        HintA and HintB are the two halves of the interaction matrix.
        
        At each step, in the computational basis the interaction matrix can be written as itself at the previous step kronecker-multiplied by the appropriate number of identity matrices; by keeping track of the representation of this product-of-identities factor in the changing basis given by SVD, we can just use this iterative proceure to reduce the number of calculations needed.
        
        The product-of-identities itself can be found iteratively by kronecker-multiplying its previous value by itself. But since identity remains identity under an orthonormal change of basis, and projecting only finds the identity in the subspace, this results in yet another identity.
        """
        HintA = projectors[-1].T.conj() @ separable(I_n0, HintA) @ projectors[-1]
        HintB = projectors[-1].T.conj() @ separable(HintB, I_n0) @ projectors[-1]
    
    return H, projectors, gvals
    
"""
### does not work, returns a flat eigenvalue across lambdas for the ground state; check later
def infinite_dmrg(n0, n_iter, l=-1):
    H = ising_hamiltonian(n0,l)
    projectors = []
    
    m = 2**n0
    
    I_n0 = separable(*([I]*n0))
    HintA = pad(sigma_x,n0,n0-1)
    HintB = pad(sigma_x,n0,0)
    
    
    for i in range(n_iter):
        new_H = pad(separable(H,I),2,0) + pad(separable(I,H),2,1) + separable(I_n0,l*sigma_z,I,I_n0) + separable(I_n0,I,l*sigma_z,I_n0) + separable(HintA,sigma_x,I,I_n0) + separable(I_n0, I, sigma_x, HintB) + separable(I_n0,sigma_x,sigma_x,I_n0)
        
        _, vecs = np.linalg.eigh(new_H)
        
        state = vecs[0].reshape(m*2,m*2)
        rhoA = np.tensordot(state, state.conj(), ([1],[1]))
        _, vecs = np.linalg.eigh(rhoA)
        projectors.append(vecs[:,:m])
        H = separable(H,I) + separable(I_n0, l*sigma_z) + separable(HintA, sigma_x)
        
        H = projectors[-1].T.conj() @ H @ projectors[-1] 
        HintA = projectors[-1].T.conj() @ separable(I_n0,sigma_x) @ projectors[-1]
        HintB = projectors[-1].T.conj() @ separable(sigma_x,I_n0) @ projectors[-1]
        
    
    return H, projectors
"""

def int_to_hex(v): 
    # helper func, returns the hex string corresponding to an int in 0-255
    strings=[str(i) for i in range(10)] + ["a","b","c","d","e","f"]
    return strings[v//16]+strings[v%16]
    

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    lam = np.linspace(-3,-0,31)
    g_rs = []
    niter = 40
    n0 = 4
    
    # get values for each lambda
    for l in lam:
        print("{:.2}".format(l))
        _, _, gvals = realspace_rg(n0,niter,l)
        g_rs.append(gvals)
    
    
    #plot
    g_rs=np.array(g_rs)
    fig, ax=plt.subplots(ncols=2)
    #alphas = [int_to_hex(66+(190*i//niter)) for i in range(niter)]
    col=[int(235*1.3**(niter-i-1)/1.3**(niter-1)) for i in range(niter)]
    rgb = mpl.colormaps["CMRmap"](col)[np.newaxis, :, :3][0]
    print(col)
    for i in range(niter):
        ax[0].plot(lam,g_rs[:,i]/(n0*2**(i+1)), c=rgb[i])#c=("#ff0000"+alphas[i]))
        ax[1].plot(lam,g_rs[:,i]/(n0*2**(i+1) -1), c=rgb[i])#c=("#ff0000"+alphas[i]))
    ax[0].set_ylabel("E/N")
    ax[1].set_ylabel("E/(N-1)")
    ax[0].set_xlabel("\u03bb")
    ax[1].set_xlabel("\u03bb")
    fig.suptitle("ground level energy at different system sizes")
    fig.tight_layout()
    plt.show()
    
        
