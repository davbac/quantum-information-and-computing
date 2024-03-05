# This code is part of qtealeaves.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Ground state simulation of the quantum Ising model.
===================================================

"""
import numpy as np
import numpy.linalg as nla
import qtealeaves as qtl
from qtealeaves.models import get_quantum_ising_1d

# Keys are L, g, symmetry sector
ref_osmps = {
    (8, 0.5, None) : -7.64059255359009,
    (8, 0.5, 0) : -7.64059255359008,
    (8, 0.5, 1) : -7.63473266438249
}

# ED code : no symmetry, E_0, E_1
# -7.640592553590071 -7.634732664382486

def main(L,J,g):
    """
    Main method for the ground state simulation of 1d quantum
    Ising model.
    """
    do_disentanglers = False

    # Available: None, 0, 1
    symmetry_sector = None

    input_folder = lambda params : '.qi_input_L%d'%(params['L'])
    output_folder = lambda params : '.qi_output_L%d'%(params['L'])

    model, my_ops = get_quantum_ising_1d()

    my_conv = qtl.convergence_parameters.TNConvergenceParameters(max_iter=7, max_bond_dimension=20)

    # Ensure ID is the first matrix
    #if(symmetry_sector is not None): del my_ops.ops['id']

    my_obs = qtl.observables.TNObservables()
    my_obs += qtl.observables.TNObsLocal('sz', 'sz')
    if(symmetry_sector is None):
        my_obs += qtl.observables.TNObsLocal('sx', 'sx')

    my_obs += qtl.observables.TNState2File("my_file", "U")
    # Set variables TTN vs aTTN
    de_loc = np.array([[1], [2]]) if(do_disentanglers) else None
    tn_type = 6 #2 if(do_disentanglers) else 1

    # Set backend according to symmetry_sector
    tensor_backend = 2 if(symmetry_sector is None) else 1

    simulation = qtl.QuantumGreenTeaSimulation(model, my_ops, my_conv, my_obs,
                                    tn_type=tn_type,
                                    tensor_backend=tensor_backend,
                                    disentangler=de_loc,
                                    folder_name_input=input_folder,
                                    folder_name_output=output_folder,
                                    has_log_file=False
    )

    params = []

    params.append({
        'L' : L,
        'J' : J,
        'g' : g
        })
    if(symmetry_sector is not None):
        params[-1]['SymmetrySectors'] = [symmetry_sector]
        params[-1]['SymmetryGenerators'] = [np.array([[0, 0], [0, 1]])]
        params[-1]['SymmetryTypes'] = ['Z2']

    for elem in params:
        energy_0, energy_1 = nla.eigh(model.build_ham(my_ops, elem))[0][:2]
        simulation.run(elem, delete_existing_folder=True)

        results = simulation.get_static_obs(elem)
        print('TN energies E0    ', results['energy'])
        print('ED energies E0, E1', energy_0, energy_1)

    return


if(__name__ == '__main__'):
    main(8, 1, 0.5)
