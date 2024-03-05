import matplotlib.pyplot as plt 
import pandas as pd
#import matplotlib as mpl

#mpl.rcParams['text.usetex'] = True

data = pd.read_csv("graph.txt")

for mode in ["a","c","f","i","o","s","b"]:
    pdata=data[data["mode"]==mode]
    #pdata = pdata.sort_values("g")
    plt.plot(pdata["D"], pdata["fid"], label=mode)
    
plt.yscale("log")
plt.xlabel("D")
plt.legend()
plt.title(r"$F_D$ for random state with N=8")
plt.show()
