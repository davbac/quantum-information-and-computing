from qtealeaves.emulator.mps_simulator import MPS
from qtealeaves.tensors import QteaTensor as QteaT
from qtealeaves.convergence_parameters.conv_params import TNConvergenceParameters as TNConvPar
import numpy as np
from copy import deepcopy

def print_mps(mps): #helper func, to check the values 
    print([i.elem for i in mps.tensors])
    print("\n")
    
def print_mpd(mpd):
    print([[i.elem for i in j] for j in mpd])
    print("\n")


def mpd_from_mps(mps):
    my_mps = deepcopy(mps)
    N = my_mps.num_sites
    if np.all(np.array(my_mps.local_dim) == my_mps.local_dim[0]):
        d = my_mps.local_dim[0]
    else:
        raise Exception("Needs an MPS with the same local dimension on all sites. Tested only on d=2")
    
    my_mps._convergence_parameters.sim_params["max_bond_dimension"]=2
    # we need the isometry at the first site
    ## move it all the way to make sure the mps is all reduced to bond dimension 2
    my_mps.iso_towards(len(my_mps)-1)
    my_mps.iso_towards(0, trunc=True)
    my_mps.normalize()
    
    my_mpd = []
    
    # start from G[0]
    G = QteaT((d,d,d,d)) # initialized to zero
    G.elem[0,0,:,:]=my_mps[0].copy().elem # set the first part equal to the MPS  
    for i in range(d):
        for j in range(d):
            if j==0 and i==0: # don't reset the first part
                continue
            M = QteaT((d,d), ctrl="R") #random init
            
            # take random matrix, keep only parts perpendicular to the ones that are already set, renormalize
            ##the parts that aren't set yet will just give a null contribution to this process
            for k in range(d):
                for h in range(d):
                    M.elem -= G.elem[k,h,:,:]*np.vdot(G.elem[k,h,:,:],M.elem)
                    M.normalize()
            G.elem[i,j,:,:] = M.elem
    
    my_mpd.append(G.transpose((1,0,2,3)))
    
    # same for G[1:N-2]
    for n in range(1,N-1):
        G = QteaT((d,d,d,d))
        G.elem[0,:,:,:]=my_mps[n].copy().elem
        
        for i in range(1,d):
            for j in range(d):
                M = QteaT((d,d), ctrl="R")
                for k in range(d):
                    for h in range(d):
                        M.elem -= G.elem[k,h,:,:]*np.vdot(G.elem[k,h,:,:], M.elem)
                        M.normalize()
                G.elem[i,j,:,:] = M.elem
        
        my_mpd.append(G.transpose((1,0,2,3)))
    
    # now for G[N-1] 
    ##remove extra leg with dimension 1 to avoid errors in apply_one_site_operator
    my_mpd.append(my_mps[N-1].copy().remove_dummy_link(2))
    
    return my_mpd

def rev_apply_mpd(mpd, mps=None, chi=2, d=2):
    if mps is None:
        N = len(mpd[0])
        mps = MPS(N, TNConvPar(max_bond_dimension=chi), initialize="vacuum", local_dim=d)
    else:
        mps = deepcopy(mps)
        N = len(mps)
    
    mps.normalize()
    for i in range(len(mpd)):
        mps.apply_one_site_operator(mpd[i][N-1],N-1)
        for n in reversed(range(N-1)):
            mps.apply_two_site_operator(mpd[i][n],n)

    mps.normalize()
    mps.iso_towards(0)
    return mps
    
def apply_mpd(mpd, mps=None, chi=2, d=2):
    if mps is None:
        N = len(mpd[0])
        mps = MPS(N, TNConvPar(max_bond_dimension=chi), initialize="vacuum", local_dim=d)
    else:
        mps = deepcopy(mps)
        N = len(mps)
    
    mps.normalize()
    for i in reversed(range(len(mpd))):
        for n in range(N-1):
            mps.apply_two_site_operator(mpd[i][n].transpose((2,3,0,1)).conj(),n) 
    
        mps.apply_one_site_operator(mpd[i][N-1].transpose((1,0)).conj(),N-1)
    mps.normalize()
    mps.iso_towards(0)
    return mps

    
if __name__ == "__main__":
    d=2
    chi=2
    N=8
    my_mps = MPS(N, TNConvPar(max_bond_dimension=chi), initialize="random", local_dim=d)
    my_mps.normalize()
    my_mpd = []
    my_mpd.append(mpd_from_mps(my_mps))
    
    
    control_mps = MPS(N, TNConvPar(max_bond_dimension=chi), initialize="vacuum", local_dim=d)
    control_mps.normalize()
    
    print(np.abs(my_mps.contract(apply_mpd(my_mpd, control_mps))))
    print(np.abs(control_mps.contract(rev_apply_mpd(my_mpd, my_mps))))
    
