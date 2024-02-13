from qtealeaves.emulator.mps_simulator import MPS
from qtealeaves.convergence_parameters.conv_params import TNConvergenceParameters as TNConvPar
from basic import *
from qtealeaves.tensors import QteaTensor as QteaT

def full_analytic_mpd(mps, D):
    my_mpd = []
    for i in range(D):
        temp_mps = rev_apply_mpd(my_mpd,mps)
        my_mpd.append(mpd_from_mps(temp_mps))
    
    return my_mpd

def optimize_mpd(mpd, mps, layers=None, l_rate=0.005, tol=0.05, maxiter=1000): 
    mpd = deepcopy(mpd)
    if layers is None:
        # by default only optimize the last layer of the MPD 
        layers = (len(mpd)-1,)
        
    N = mps.num_sites
    chi = mps._convergence_parameters.sim_params["max_bond_dimension"]
    if np.all(np.array(mps.local_dim) == mps.local_dim[0]):
        d = mps.local_dim[0]
    else:
        raise Exception("Needs an MPS with the same local dimension on all sites. Tested only on d=2")
    
    control_mps = MPS(N, TNConvPar(max_bond_dimension=chi), initialize="vacuum", local_dim=d)
    control_mps.normalize()
    
    II = QteaT.from_elem_array(np.eye(d**2).reshape((d,d,d,d)))
    I = QteaT.from_elem_array(np.eye(d))
    
    fid = []
    for count in range(maxiter):
        fid.append(np.abs(mps.contract(apply_mpd(mpd, control_mps))))
        if fid[-1]>1-tol:
            break
        for l in layers:
            mps_a = apply_mpd(mpd[:l],control_mps)
            mps_b = rev_apply_mpd(mpd[l+1:], mps)
            for n in range(N):
                mpd_a = [II for _ in range(n+1)]
                mpd_a.extend(mpd[l][n+1:])
                if n == N-1:
                    mpd_a[-1]=I 
                
                mpd_b = mpd[l][:n]
                mpd_b.extend([II for _ in range(N-n-1)])
                mpd_b.append(I)
                
                mps_a_n = apply_mpd([mpd_a], mps_a)
                mps_b_n = rev_apply_mpd([mpd_b], mps_b)
                
                if n==N-1:
                    f = mps_a_n.contract(mps_b_n, (0,N-1)).remove_dummy_link(0)
                    dtens = mps_a_n[N-1].remove_dummy_link(2).tensordot(f, (0,0)).tensordot(mps_b_n[N-1].conj().remove_dummy_link(2), (1,0))
                    #dtens.transpose((1,0))
                    
                else: 
                    if n<N-2: #move the legs we don't want to contract out of the contraction range
                        mps_a_n.swap_qubits((n, N-2))
                        mps_a_n.swap_qubits((n+1, N-1))
                        mps_b_n.swap_qubits((n, N-2))
                        mps_b_n.swap_qubits((n+1, N-1))
                
                    f = mps_a_n.contract(mps_b_n, (0,N-2)).remove_dummy_link(0) 
                    dtens = mps_a_n[N-2].tensordot(f, (0,0)).tensordot(mps_b_n[N-2].conj(), (2,0))
                    dtens = mps_a_n[N-1].remove_dummy_link(2).tensordot(dtens, (0,1)).tensordot(mps_b_n[N-1].conj().remove_dummy_link(2), (3,0))
                    
                    #dtens.transpose((2,3,0,1))
                    
                
                mpd[l][n].elem += dtens.elem*l_rate
    
    return mpd, fid

if __name__=="__main__":
    d=2
    chi=8
    N=8
    D=2
    my_mps = MPS(N, TNConvPar(max_bond_dimension=chi), initialize="random", local_dim=d)
    my_mps.normalize()
    
    my_mpd = full_analytic_mpd(my_mps, D)
    
    #my_mpd = optimize_mpd(my_mpd, my_mps, maxiter=100)
    
    control_mps = MPS(N, TNConvPar(max_bond_dimension=chi), initialize="vacuum", local_dim=d)
    control_mps.normalize()
    
    print(-np.log(np.abs(my_mps.contract(apply_mpd(my_mpd, control_mps))))/N)
    print(-np.log(np.abs(control_mps.contract(rev_apply_mpd(my_mpd, my_mps))))/N)
    
    my_mpd, fid = optimize_mpd(my_mpd, my_mps, layers=(0,1), maxiter=400)
    
    print()
    print(-np.log(np.abs(my_mps.contract(apply_mpd(my_mpd, control_mps))))/N)
    print(-np.log(np.abs(control_mps.contract(rev_apply_mpd(my_mpd, my_mps))))/N)
    
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(fid)
    ax[1].plot(-np.log(fid)/N)
    plt.show()
