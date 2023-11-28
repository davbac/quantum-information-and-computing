import numpy as np
from scipy import linalg, stats, optimize
import matplotlib.pyplot as plt

N = 1500 #matrix size

# generate matrix
A = np.random.uniform(-1, 1, size=(N,N)) + 1.j * np.random.uniform(-1, 1, size=(N,N))

# make hermitian
for i in range(1,N):
    for j in range(i+1,N):
        A[i][j]=0

A += A.conj().T

for i in range(N):
    A[i][i] /= 2

# find eigenvalues
eigenvals, _ = linalg.eig(A) ## the second element is the eigenvector list

# find spacings
s = np.diff(sorted(eigenvals.real))

# normalize spacings
s = s / s.mean()
s = s[s<5] # cut higher values, to have better results from the binning

# calculate the values and the bin extrema
vals, breaks = np.histogram(s, bins=100, density=True)

# to plot them manually afterwards
plotbreaks = []
plotvals = []

for i in breaks:
    plotbreaks.append(i)
    plotbreaks.append(i)

for i in vals:
    plotvals.append(i)
    plotvals.append(i)

del plotbreaks[0]
del plotbreaks[-1]


# fit with ax^alpha * exp(bx^beta) -> log(y) = log(a) + alpha log(x) + b exp(beta log(x))
x=(breaks[1:]+breaks[:-1])/2 # center point of the bin
y=np.copy(vals) # we don't touch the vals array in case we need it later
y[y==0]=min(y[y!=0]) #remove 0s to work in log-log scale without errors

# cut out the tail of the distribution from the last point in which there are two consecutive bins 
## with more than two counts
last = None
for i in reversed(range(len(y))):
    if y[i]>2*min(y) and y[i-1]>2*min(y):
        last=i+1
        break

x=x[:last]
y=y[:last]

# function to be fitted     ## first parameter is log(a)
def func(lx, la,alpha,b,beta):
    return la+alpha*lx+b*np.exp(beta*lx)


## first generic approximation, found manually
#plt.plot(np.log(x), np.log(y))
#yplot=np.exp(func(np.log(x),3, 2.5, -3, 1))
#plt.plot(np.log(x),np.log(yplot))
#plt.title("first approx")
#plt.show()

# optimizazion of the parameters; pcov is the covariance matrix
popt, pcov = optimize.curve_fit(func, np.log(x), np.log(y), p0=(3, 2.5, -3, 1)) 
print(popt)

# compute the function's values using the optimal parameters
yplot=np.exp(func(np.log(x), *popt))
yplot/= np.sum(yplot)*(x[1]-x[0]) #normalize

# plot
plt.plot(x, y)
plt.plot(x,yplot)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("s")
plt.ylabel("density")
plt.title("Fit in log-log scale")
plt.show()


eig_b = np.random.uniform(-1,1,size=N)
s_b = np.diff(sorted(eig_b))
s_b /= s_b.mean()
s_b = s_b[s_b<5]

plt.hist(s_b, bins=100, density=True, facecolor="white", edgecolor="#3366ffaa", label="real diagonal matrix")
plt.plot(plotbreaks, plotvals, marker=None, linestyle="solid", label="random hermitian matrix")
plt.plot(x, yplot, label="fit for random hermitian")
plt.title("Histogram of the normalized separations")
plt.xlabel("s")
plt.ylabel("density")
plt.legend()
plt.show()
