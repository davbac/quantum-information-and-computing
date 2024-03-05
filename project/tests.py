from higher import *
from QuantumIsing_1d_groundstate import main as write_ising

modes = ["i", "a", "o", "s", "b", "f", "c"]
couplings = np.linspace(0.2,2,10)
#D = 4


N=8
J=1
#g=0.5

f = open("graph.txt", "w")

#for g in couplings:
for D in range(8,13):
    print(D)
    #write_ising(N,J,g)
    #tensor_array = np.load("my_file.npy", allow_pickle=True)
    #maxchi = np.max([[i.elem.shape] for i in tensor_array])
    maxchi = 16
    #my_mps = MPS(N, TNConvPar(max_bond_dimension=int(maxchi)),local_dim=2)
    my_mps = MPS(N, TNConvPar(max_bond_dimension=int(maxchi)),local_dim=2, initialize="random")
    #my_mps._tensors = list(tensor_array)
    my_mps.normalize()
    control_mps = MPS(N, TNConvPar(max_bond_dimension=int(maxchi)), initialize="vacuum", local_dim=2)
    control_mps.normalize()
    
    
    for ctrl in modes:
        mpd = get_mpd(my_mps, D, ctrl=ctrl, tol=1e-8)
        f.write(",".join([str(i) for i in [D, ctrl, -np.log(abs(control_mps.contract(apply_mpd(mpd, my_mps, rev=True))))/N]])+"\n")

f.close()
