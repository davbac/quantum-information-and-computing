import numpy as np
from scipy.special import hermite


def timestep(wf, U_K, U_V2):
    """Perform a single timestep evolution
    Arguments:
        wf : the wavefunction, in real space
        U_K : the kinetic evolution operator, in momentum space
        U_V2 : the potential evolution operator, with dt halved
    
    Returns the evolved waveform in real space
    """
    # apply half-timed potential operator
    wf *= U_V2
    # go to mpmentum spacce
    wf_k = np.fft.fft(wf)
    # apply kinetic operator
    wf_k *= U_K
    # return to real space
    wf = np.fft.ifft(wf_k)
    # apply second half of potential operator
    wf *= U_V2
    # return results
    return wf

Vnull = lambda t: lambda x: np.zeros(x.shape)
Knull = lambda k: 0.5*k**2

def evolve(wf, K=Knull, V=Vnull, T=1, xmax=5, xsteps=1024, tsteps=1e5, norm_tol=1e-5, sample_t=1e2):
    """Evolve a given wavefunction for a time T, using a potential V(t)(x) and a kinetic energy K(k)
    
    Arguments:
        wf : the wavefunction, given as a complex numpy array
        
        K (optional) : the shape of the kinetic energy wrt the momentum
                        If not given,  uses 0.5*k**2
        
        V (optional) : the shape of the potential, as a function of time
                        Must be in the form pf a funciton of t that returns a function of x
        
        T (optional) : the time scale, 1 if not given
        
        xmax (optional) : the width of real space to consider (goes from -xmax to xmax); standard is 5
        
        xsteps (optional) : the amount of steps in real space to use; standard is 1024
        
        tsteps (optional) : the amount of timesteps to use; standard is 1e5
        
        norm_tol (optional) : the tolerance before requiring renormalization; standard is 1e-5
        
        sample_t (optional) : the number of steps before saving an intermediate result; standard is 100
    
    Returns a list with the sampled values of wf
    """
    
    # create x, k grids
    x = np.linspace(-xmax, xmax, xsteps)
    k = np.concatenate(( np.arange(0,xsteps/2), np.arange(-xsteps/2,0))) * np.pi/xmax
    
    # delta time
    dt = T/tsteps
    
    # kinetic evolution operator    ## defined here since it doesn't chenge in time
    U_K = np.exp(-1.j * K(k) * dt)
    
    results = []
    
    for i in range(int(tsteps)):
        # time at step i
        t = i*dt
        
        # potential evolution operator      ## defined with the half-timed version already
        U_V2 = np.exp(-0.5j * V(t)(x) * dt) 
        
        # evolve
        wf = timestep(wf, U_K, U_V2)
        
        # check normalization, correct if needed
        norm = np.sum(np.abs(wf)**2)*2*xmax/xsteps
        if abs(norm - 1) > norm_tol:
            wf /= norm**0.5
            print("renormalization needed at step ",i,"; value was ",norm)
        
        # save intermediate results
        if not i%sample_t:
            results.append(wf)
    
    return results


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from matplotlib import animation
    ## take the definition of the ground state wavefunction from last time
    def fact(x):
        res = 1
        for i in range(2,x+1):
            res*=i
        return res
    
    omega=1
    exact = lambda n: lambda x: (((1/(fact(n)*(2**n)))**0.5)*((omega/np.pi)**0.25)*np.exp(-0.5*omega*x**2)*hermite(n)(x*omega**0.5))
    
    x = np.linspace(-5, 5, 1024)
    wf = exact(0)(x)
    wf = wf.astype("complex128")
    
    # define moving potential
    def V(t):
        def pot(x):
            return (x-t)**2
        return pot
        
    #evolve
    res = evolve(wf, V=V)
    
    
    #plot
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    wf, = ax.plot(x, np.abs(res[0]**2).real, color="blue")
    pot, = ax2.plot(x, V(0)(x), color="red")
    lines=[wf, pot]
    
    ax.set_ylim(0, 1.1*np.max(np.abs(res)**2))
    
    def animate(i):
        lines[0].set_data(x,np.abs(res[i]**2).real)
        lines[1].set_data(x, V(i/1000)(x))
        return lines
    
    anim = animation.FuncAnimation(fig, animate, interval=0.5, frames=len(res), blit=True)
    
    ax.set_ylabel("density")
    ax2.set_ylabel("potential")
    plt.show()
    
    
    
    ### now for k space
    k = np.concatenate(( np.arange(0,1024/2), np.arange(-1024/2,0))) * np.pi/5
    res_k = [np.fft.fft(res[i]) for i in range(len(res))]
    fig, ax = plt.subplots()
    line, = ax.plot(k, np.abs(res_k[0]**2).real, color="blue")
    
    
    def k_animate(i):
        line.set_data(k,np.abs(res_k[i]**2).real)
        return line,
    
    anim = animation.FuncAnimation(fig, k_animate, interval=0.1, frames=len(res), blit=True)
    
    ax.set_ylim(0, 1.1*np.max(np.abs(res_k)**2))
    ax.set_xlim(-10,10)
    
    plt.show()
