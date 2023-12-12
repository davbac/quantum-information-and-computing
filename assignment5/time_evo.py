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
        wf                  : the wavefunction, given as a complex numpy array
        
        K (optional)        : the shape of the kinetic energy wrt the momentum
                                If not given,  uses 0.5*k**2
        
        V (optional)        : the shape of the potential, as a function of time
                                Must be in the form of a function of t that returns a function of x
                                If not given,  uses a flat zero potential
        
        T (optional)        : the time scale, 1 if not given
        
        xmax (optional)     : the width of real space to consider (goes from -xmax to xmax); standard is 5
        
        xsteps (optional)   : the amount of steps in real space to use; standard is 1024
        
        tsteps (optional)   : the amount of timesteps to use; standard is 1e5
        
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
            if i!=0: 
                #the normalization of the input wavefunction might not be perfect;
                #don't print in that case
                print("renormalization needed at step ",i,"; value was ",norm, sep="")
        
        # save intermediate results
        if not i%sample_t:
            results.append(wf)
    
    return results


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from eig import harmonic_eig
    
    ## set parameters
    T=10
    xmax=10
    xsteps=2048
    tsteps=1e6
    sample_t=1e4
    omega=1
    
    
    # get waveform from the procedure defined last time
    x, wf = harmonic_eig(npoints=xsteps, j=1, interval=(-xmax,xmax), omega=omega)
    if wf[xsteps//2,0] <0: # flip if the function is returned with a  - sign
        wf *= -1
    wf = wf.astype("complex128").reshape((2048,))
    
    
    # define moving potential
    def V(t):
        def pot(x):
            return (x-(t/T))**2
        return pot
        
    #evolve
    res = evolve(wf, V=V, sample_t=sample_t, T=T, tsteps=tsteps, xmax=xmax, xsteps=xsteps)
    #print(len(res))
    
    #find mean, std of the position at each time
    meanpos=np.array([np.mean(x*(np.abs(res[i]**2).real))*2*xmax for i in range(len(res))])
    stdpos=np.array([np.mean(x**2*(np.abs(res[i]**2).real))*2*xmax-meanpos[i]**2 for i in range(len(res))])
    t=np.linspace(0,T,len(res))
    
    
    #plot
    plt.fill_between(t,meanpos-stdpos,meanpos+stdpos)
    plt.plot(t,t/T, label="potential's center", color="red")
    plt.plot(t,meanpos, label="mean position", color="orange")
    
    plt.legend()
    plt.show()
    
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    wf, = ax.plot(x, np.abs(res[0]**2).real, color="blue")
    pot, = ax2.plot(x, V(0)(x), color="red")
    mean = ax.vlines(meanpos[0], -0.5,1.5)
    lines=[wf, pot, mean]
    
    ax.set_ylim(0, 1.1*np.max(np.abs(res)**2))
    
    def animate(i):
        lines[0].set_data(x,np.abs(res[i]**2).real)
        lines[1].set_data(x, V(i*T*sample_t/tsteps)(x))
        lines[2].set_segments([np.array([[meanpos[i], -0.5],[meanpos[i],1.5]])])
        return lines
    
    anim = animation.FuncAnimation(fig, animate, interval=0.1, frames=len(res), blit=True)
    
    ax.set_ylabel("density")
    ax2.set_ylabel("potential")
    
    ### save
    writer = animation.PillowWriter(fps=120,
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)
    anim.save('real_space.gif', writer=writer)
    ### show instead
    #plt.show()
    
    
    
    ### now for k space
    k = np.concatenate(( np.arange(0,xsteps/2), np.arange(-xsteps/2,0))) * np.pi/xmax
    res_k = [np.fft.fft(res[i]) for i in range(len(res))]
    fig, ax = plt.subplots()
    line, = ax.plot(k, np.abs(res_k[0]**2).real, color="blue")
    
    
    def k_animate(i):
        o = np.argsort(k) #ordering
        line.set_data(k[o],(np.abs(res_k[i]**2).real)[o])
        return line,
    
    anim = animation.FuncAnimation(fig, k_animate, interval=0.1, frames=len(res), blit=True)
    
    ax.set_ylim(0, 1.1*np.max(np.abs(res_k)**2))
    ax.set_xlim(-10,10)
    
    ### save
    writer = animation.PillowWriter(fps=120,
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)
    anim.save('momentum_space.gif', writer=writer)
    ### show instead
    #plt.show()
