from qtealeaves.emulator.mps_simulator import MPS
from qtealeaves.tensors import QteaTensor as QteaT
from qtealeaves.convergence_parameters.conv_params import TNConvergenceParameters as TNConvPar
import numpy as np

def print_mps(mps): #helper func, to check the values 
    print([i.elem for i in mps.tensors])
    print("\n")
    
def print_mpd(mpd):
    print([i.elem for i in mpd])
    print("\n")


H = QteaT.from_elem_array(np.array([[1,1],[1,-1]])/(2**0.5))
CNOT = QteaT.from_elem_array(np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]).reshape((2,2,2,2)))
NCNOT = QteaT.from_elem_array(np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]]).reshape((2,2,2,2)))
HY = QteaT.from_elem_array(np.array([[1,1j],[1,-1j]])/(2**0.5))

N = 8 # number of sites
d = 2 # local dimension
chi = 2 # bond dimension
control = "A" # switch between GHZ and random 

if control=="R":
    # random
    my_mps = MPS(N, TNConvPar(max_bond_dimension=chi), initialize="random", local_dim=d)

elif control=="G":
    # GHZ
    my_mps = MPS(N, TNConvPar(max_bond_dimension=chi), initialize="vacuum", local_dim=d)
    my_mps.apply_one_site_operator(H,0)
    for n in range(N-1):
        my_mps.apply_two_site_operator(CNOT,n)

elif control=="A":
    # alternative GHZ - "simple" non-symmetric state
    my_mps = MPS(N, TNConvPar(max_bond_dimension=chi), initialize="vacuum", local_dim=d)
    my_mps.apply_one_site_operator(H,0)
    for n in range(N-1):
        my_mps.apply_two_site_operator(NCNOT,n)    


my_mps.normalize()
#print_mps(my_mps)

my_mps.iso_towards(0) 
#my_mps.iso_towards(N-1)
print_mps(my_mps)

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

my_mpd.append(G)

# same for G[1-6]
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
    
    my_mpd.append(G)

# now for G[7] 
##remove extra leg with dimension 1 to avoid errors in apply_one_site_operator
my_mpd.append(my_mps[N-1].copy().remove_dummy_link(2))


## now check results

control_mps = MPS(N, TNConvPar(max_bond_dimension=chi), initialize="vacuum", local_dim=d)
control_mps.normalize()
# initialized to zero on all sites

# apply the MPD a gate at a time, see notes for how the circuit is built

my_mps.apply_one_site_operator(my_mpd[N-1].transpose((1,0)),N-1)
for n in reversed(range(N-1)):
    my_mps.apply_two_site_operator(my_mpd[n],n,swap=True)

my_mps.normalize()

#print_mpd(my_mpd)
print_mps(my_mps)

# print fidelity
print(np.abs(control_mps.contract(my_mps))) 

# GHZ : quite low, ~ 0.3 - 0.5
# random : very low, ~ 0.02
