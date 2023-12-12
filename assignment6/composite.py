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
    if np.prod(dims) != mat.shape[0]:
        raise Exception("The subsystem dimensions given do not match the denisty matrix dimension")
    
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


if __name__=="__main__":
    s = random_separable(2,2)
    rho = get_density_mat(*s)
    r = separable(*rho)
    print(np.isclose(trace_out_subs(r,0,(2,2)) , rho[1]).all())
