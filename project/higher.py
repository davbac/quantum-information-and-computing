from qtealeaves.emulator import MPS
from qtealeaves.convergence_parameters import TNConvergenceParameters as TNConvPar
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
        contr = mps.contract(apply_mpd(mpd, control_mps))
        fid.append(np.abs(contr))
        if fid[-1]>1-tol:
            break
        
        new_mpd = deepcopy(mpd)
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
                
                if n==N-1: # two-leg tensor case
                    c = mps_a_n.contract(mps_b_n, (0,N-1)).remove_dummy_link(0)
                    f = mps_a_n[N-1].remove_dummy_link(2).tensordot(c, (0,0)).tensordot(mps_b_n[N-1].conj().remove_dummy_link(2), (1,0))
                    f = f.transpose((1,0))
                    
                elif n==N-2: # right edge case
                    c = mps_a_n.contract(mps_b_n, (0,N-2)).remove_dummy_link(0) 
                    twoq_a = mps_a_n[N-2].tensordot(mps_a_n[N-1].remove_dummy_link(2), (2,0))
                    twoq_b = mps_b_n[N-2].tensordot(mps_b_n[N-1].remove_dummy_link(2), (2,0))
                    f = twoq_a.tensordot(c.tensordot(twoq_b.conj(), (1,0)), (0,0))
                    f = f.transpose((2,3,0,1))
                    
                elif n==0: # left edge case
                    c = mps_a_n.contract(mps_b_n, (N-1,1)).remove_dummy_link(2) 
                    twoq_a = mps_a_n[0].remove_dummy_link(0).tensordot(mps_a_n[1], (1,0))
                    twoq_b = mps_b_n[0].remove_dummy_link(0).tensordot(mps_b_n[1], (1,0))
                    f = twoq_a.tensordot(c, (2,0)).tensordot(twoq_b.conj(), (2,2))
                    f = f.transpose((2,3,0,1))
                
                else: # in between
                    c1 = mps_a_n.contract(mps_b_n, (0,n)).remove_dummy_link(0) 
                    c2 = mps_a_n.contract(mps_b_n, (N-1,n+1)).remove_dummy_link(2)
                    twoq_a = mps_a_n[n].tensordot(mps_a_n[n+1], (2,0))
                    twoq_b = mps_b_n[n].tensordot(mps_b_n[n+1], (2,0))
                    f = twoq_a.tensordot(c2, (3,0)).tensordot(c1.tensordot(twoq_b.conj(), (1,0)), ([0,3],[0,3]))
                    f = f.transpose((2,3,0,1))
                
                """ # gradient ascent kinda thing, not working
                new_mpd[l][n].elem += f.elem*l_rate*contr.conjugate()
                
                #new_mpd[l][n].normalize() #nope, not normalized, it needs to be _unitary_
                if n==N-1:
                    t_left, t_right, _, _ = new_mpd[l][n].split_svd([0],[1], conv_params=TNConvPar(), no_truncation=True)
                    new_mpd[l][n] = t_left.tensordot(t_right, (1,0))
                else:
                    t_left, t_right, _, _ = new_mpd[l][n].split_svd([0,1],[2,3], conv_params=TNConvPar(), no_truncation=True)
                    new_mpd[l][n] = t_left.tensordot(t_right, (2,0))
                """
                if n!=N-1:
                    # reshape tensor in matrix form in the 4-leg cases
                    f = f.reshape((d**2, d**2))
                
                # make it unitary
                t_left, t_right, _, _ = f.split_svd([0],[1], conv_params=TNConvPar(), no_truncation=True)
                unew = t_left @ t_right
                
                # get "old" tensor, reshape if needed
                if n!=N-1:
                    u = new_mpd[l][n].reshape((d**2,d**2))
                else:
                    u = new_mpd[l][n]
                
                # find adj of the old tensor
                uadj = u.transpose((1,0)).conj()
                
                # calculate (uadj@unew)**lrate
                tens = uadj @ unew
                t_left, t_right, singvals, _ = tens.split_svd([0],[1], conv_params=TNConvPar(), no_truncation=True)
                singvals = singvals ** l_rate
                tens = t_left @ QteaT.from_elem_array(np.diag(singvals)) @ t_right
                
                # u' = u (uadj@unew)**lrate
                uprime = u @ tens
                
                # reshape if needed, assign
                if n!=N-1:
                    uprime = uprime.reshape((d,d,d,d))
                
                new_mpd[l][n] = uprime
                
                    
        mpd=new_mpd
    
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
    
    my_mpd, fid = optimize_mpd(my_mpd, my_mps, layers=(0,1), maxiter=100, l_rate=0.06)
    
    print()
    print(-np.log(np.abs(my_mps.contract(apply_mpd(my_mpd, control_mps))))/N)
    print(-np.log(np.abs(control_mps.contract(rev_apply_mpd(my_mpd, my_mps))))/N)
    
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(fid)
    ax[1].plot(-np.log(fid)/N)
    plt.show()
