from qtealeaves.emulator import MPS
from qtealeaves.convergence_parameters import TNConvergenceParameters as TNConvPar
from basic import *
from qtealeaves.tensors import QteaTensor as QteaT
from scipy import linalg

def full_analytic_mpd(mps, D):
    my_mpd = []
    for i in range(D):
        temp_mps = apply_mpd(my_mpd,mps, rev=True)
        my_mpd.append(mpd_from_mps(temp_mps))
    
    return my_mpd

def optimize_mpd(mpd, mps, layers=None, l_rate=0.001, tol=0.05, maxiter=1000): 
    # a few parameters
    N=mps.num_sites
    chi=mps._convergence_parameters.sim_params["max_bond_dimension"]
    d=np.random.choice(mps.local_dim)
    if not np.allclose(mps.local_dim,d):
        raise Exception("This algorithm only works with the same physical dimension across loci")
    
    # define some useful quantities
    control_mps = MPS(N, TNConvPar(max_bond_dimension=chi), initialize="vacuum", local_dim=int(d))
    I = QteaT.from_elem_array(np.eye(d))
    II = QteaT.from_elem_array(np.eye(d**2).reshape((d,d,d,d)))
    
    fid = []
    
    for it in range(maxiter): #start iterating
        #check convergence
        contr = control_mps.contract(apply_mpd(mpd, mps, rev=True))
        fid.append(np.abs(contr))
        if np.abs(contr)>1-tol:
            break
        
        if it%10 == 9:
            print(it, "\t", fid[-1])
        
        new_mpd = deepcopy(mpd)
        
        for l in layers:
            ## Get the result of the circuit "up to the given layer" on both sides
            
            mpd_f = mpd[l+1:] # "forward" part, applied to control_mps
            mpd_r = mpd[:l]   # "reverse" part, applied to the given mps
            
            #mps_f = apply_mpd(mpd_f, control_mps)
            #mps_r = apply_mpd(mpd_r, mps, rev=True)
            
            for n in range(N):
                mpd_r_n = [[II for _ in range(n+1)] + mpd[l][n+1:]]
                if n==N-1:
                    mpd_r_n[0][-1]=I 
                
                mpd_f_n = [mpd[l][:n] + [II for _ in range(N-n-1)] + [I]]
                
                
                mpd_mid = [II for _ in range(n)] + [mpd[l][n]] + [II for _ in range(N-n-1)]
                if n!=N-1:
                    mpd_mid[-1]=I
                
                if not np.allclose(abs(mps.contract(apply_mpd(mpd_r+mpd_r_n+[mpd_mid]+mpd_f_n+mpd_f, control_mps))), abs(contr)):
                    print("Error in building mpd parts:",abs(mps.contract(apply_mpd(mpd_r+mpd_r_n+[mpd_mid]+mpd_f_n+mpd_f, control_mps))), abs(contr))
                
                #mps_f_n = apply_mpd(mpd_f_n, mps_f, adj=True, rev=True)
                #mps_r_n = apply_mpd(mpd_r_n, mps_r, rev=True)
                
                mps_f_n = apply_mpd(mpd_f_n+mpd_f, control_mps, chi=10)
                mps_r_n = apply_mpd(mpd_r+mpd_r_n, mps, rev=True, chi=10)
                
                mps_f_n.iso_towards(n) # not needed
                mps_r_n.iso_towards(n) # (probably)
                
                if n>0:
                    c1 = mps_r_n.contract(mps_f_n, (0,n)).remove_dummy_link(0)
                else: 
                    c1 = mps_r_n[0].eye_like(mps_r_n[0].links[0])
                if n<N-2:
                    c2 = mps_r_n.contract(mps_f_n, (N-1, n+1)).remove_dummy_link(2)
                else:
                    c2 = mps_r_n[N-1].eye_like(mps_r_n[N-1].links[2])
                    
                
                if n!=N-1:
                    q_f = QteaT.__matmul__(*mps_f_n[n:n+2])
                    q_r = QteaT.__matmul__(*mps_r_n[n:n+2])
                else:
                    q_f = mps_f_n[n]
                    q_r = mps_r_n[n]
                
                #c1 = c1.transpose((1,0))
                #c2 = c2.transpose((1,0))
                
                f = c1.tensordot(q_r, (0,0)).tensordot(c2, (-1,0)).tensordot(q_f.conj(), ([0,-1],[0,-1]))
                
                
                
                if n!=N-1:
                    f = f.reshape((d**2, d**2))
                    u = mpd[l][n].reshape((d**2, d**2))
                else:
                    u = mpd[l][n]
                
                f = f.transpose((1,0))
                
                if not np.allclose(abs(f.tensordot(u, ([0,1],[0,1])).elem),abs(contr)):
                    print("Error in computing f:", abs(f.tensordot(u, ([0,1],[0,1])).elem),abs(contr))
                
                f = f.conj()
                
                f_l, f_r, _, _ = f.split_svd([0],[1], conv_params=TNConvPar(), no_truncation=True)
                f = f_l @ f_r 
                
                
                m = u.transpose((1,0)).conj() @ f
                """
                m_l, m_r, vals, _ = m.split_svd([0],[1], conv_params=TNConvPar(), no_truncation=True)
                
                vals = QteaT.from_elem_array(np.diag(vals**l_rate))
                m = m_l @ vals @ m_r"""
                m.elem = linalg.fractional_matrix_power(m.elem, l_rate)
                uprime = u @ m
                
                ### ATTENTION: vals all have |lambda|=1 , exponentiation changes only phase
                # aka uprime is equal to unew, at least for the real case [unew kept name "f" in code]
                
                ### vals seem to be only +1 even for complex case
                
                ## using scipy linalg works
                
                #print(u.elem, uprime.elem, f.elem, "\n", sep="\n")
                
                if n!=N-1:
                    uprime = uprime.reshape((d,d,d,d))
                
                new_mpd[l][n] = uprime
        
        mpd = new_mpd
                
    return mpd, fid

if __name__=="__main__":
    d=2
    chi=8
    N=4
    D=2
    from qtealeaves.tensors import TensorBackend
    my_mps = MPS(N, TNConvPar(max_bond_dimension=chi), initialize="random", local_dim=d)#, tensor_backend=TensorBackend(dtype="D"))
    my_mps.normalize()
    
    my_mpd = full_analytic_mpd(my_mps, D)
    
    control_mps = MPS(N, TNConvPar(max_bond_dimension=chi), initialize="vacuum", local_dim=d)#, tensor_backend=TensorBackend(dtype="D"))
    control_mps.normalize()
    
    print(-np.log(np.abs(my_mps.contract(apply_mpd(my_mpd, control_mps))))/N)
    print(-np.log(np.abs(control_mps.contract(apply_mpd(my_mpd, my_mps, rev=True))))/N)
    
    my_mpd, fid = optimize_mpd(my_mpd, my_mps, layers=(0,1), maxiter=50, l_rate=0.2, tol=1e-10)
    
    print()
    print(-np.log(np.abs(my_mps.contract(apply_mpd(my_mpd, control_mps))))/N)
    print(-np.log(np.abs(control_mps.contract(apply_mpd(my_mpd, my_mps, rev=True))))/N)
    
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(fid)
    ax[1].plot(-np.log(fid)/N)
    plt.show()
