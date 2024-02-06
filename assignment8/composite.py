import numpy as np

def separable(*sub):
    """Builds a separable state from the single-subsystem states.
    The subsystem states are just given as separate arguments to the function.
        --- also works for density matrices, operators and such
    
    In the separable case, this is _NOT_ the optimal representation [as a matter of fact, the `sub` list is]
    """
    
    if len(sub)==0: # in case the function is called with no arguments, raise an error
        raise Exception("Can't build a state from nothing")
    elif len(sub)==1: # if a single state is given, return it
        return sub[0]
    
    res = np.kron(sub[0],sub[1]) # put the first two states together
    
    if len(sub)>2: # if there's more, multiply them in one by one
        for i in range(2,len(sub)):
            res = np.kron(res, sub[i])
    
    return res


def random_state(D): #example routine to generate a random state for a dimension D system
    # generate complex numbers with random magnitude and phase
    res = np.random.random(size=(D,)) * np.exp(2.j *np.pi* (np.random.random(size=(D,)))-0.5 )
    res /= np.abs(res**2).sum()**0.5 # normalize the vector
    return res

def random_separable(D,N, single_state_gen=random_state):
    """Returns a list of N arrays representing the state of each D-dimensional subsystem.
    If used as `separable(*random_separable)`, you'll get the full system state"""
    return [single_state_gen(D) for i in range(N)]

def random_composite(D,N):
    """Returns a generic random vector in the multi-body hilbert space given by N bodies with subspace dimension D each
    """
    res = np.random.random(size=(D**N,))*np.exp(2.j*np.pi*(np.random.random(size=(D**N,))-0.5))
    res /= np.abs(res**2).sum()**0.5 # normalize the vector
    return res


def get_density_mat(*states):
    """Builds a density matrix from the corresponding state
    Using the list-of-substates representation returns a list-of-submatrices representation
    """
    if len(states)==0: # in case the function is called with no arguments, raise an error
        raise Exception("Pass at least one state")
    elif len(states)==1:
        return np.outer(states[0], states[0].conj())
    else:
        return [np.outer(s,s.conj()) for s in states]
        

def trace_out_subs(mat, ind, dims):
    """Function to trace out a specific subsystem from the density matrix.
    Inputs:
        mat : original density matrix
        ind : index of the subsystem to trace out
        dims: dimensions of the various subsystems
    
    Outputs:  the matrix for the remaining subsystems
    """
    if np.prod(dims) != mat.shape[0]:
        raise Exception("The subsystem dimensions given do not match the density matrix dimension")
    
    pre = int(np.prod(dims[:ind]))
    dim = dims[ind]
    post = int(np.prod(dims[(ind+1):]))
    #print(pre, dim, post)
    
    res = np.zeros(shape=(pre*post, pre*post), dtype="complex128")
    for i in range(dim):
        indices = [n for n in range(mat.shape[0]) if (n//post)%dim == i]
        #print(indices, mat[indices,:][:,indices])
        res += (mat[indices,:][:, indices]).reshape(dim, dim)
    
    return res


def pad(mat, N=2, ind=0, pad_with=None, dim=None):
    """Function to pad operators, by default with identities
    Inputs:
        mat : matrix to be padded
        N   : (optional) number of subsystems to consider; 2 by default
        ind : (optional) the index of the subsystem mat acts on; 0 by default
        pad_with : (optional) the matrix to use for padding; the identity of dimension dim by default
        dim : (optional) the dimension of each subspace; by default, the dimension of mat
        
    Outputs: the padded operator, in its full matrix form
    """
    if pad_with is None:
        if dim is None:
            if mat.shape[0] == mat.shape[1]:
                dim = mat.shape[0]
            else:
                raise Exception("Cannot infer subspace dimension")
        pad_with = np.diag(np.ones(dim))
    
    return separable(*([pad_with]*ind), mat, *([pad_with]*(N-ind-1)) )

if __name__=="__main__":
    ## test: the trace of the density matrix built from a separable state is the same as the
    ##      density matrix of the subsystem
    
    s = random_separable(2,2)
    rho = get_density_mat(*s)
    r = separable(*rho)
    assert np.isclose(trace_out_subs(r,0,(2,2)) , rho[1]).all()
    
    
    
    ## assignment requirements:
    D = 3
    N = 4
    
    s = random_separable(D,N) # gives random separable state
    
    state = random_composite(D,N) # gives a random state in the composite space
    
    r = get_density_mat(*s) # finds separate density matrix for the separable case
    rho = get_density_mat(state) # finds the density matrix in the full space
    
    rho_1 = trace_out_subs(rho, 0, (9,9)) # finds the density matrix of the right part of the system
    # since it was defined with 4 loci with dimension 3, if we want to split in half we must give the 
    #  subsystem dimensions as 3**2 (=9) each
    
    print(rho, rho_1)
