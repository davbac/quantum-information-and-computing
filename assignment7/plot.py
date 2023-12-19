import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("times.txt")

def weighted_sum(df):
    new_df=pd.DataFrame({"N":[0],"time":[0.],"sd":[0.]})
    new_df["N"] = int(df["N"].mean())
    new_df["sd"]=1/((1/df["sd"]**2).sum()) **0.5
    new_df["time"]=(df["time"]/(df["sd"]**2)).sum() *new_df["sd"]**2
    return new_df[["N","time","sd"]]

data = data.groupby("N").apply(weighted_sum)

coeffs = np.polyfit(data["N"], np.log(data["time"]), 1, w=data["sd"]/data["time"])
yfit = np.exp(data["N"]*coeffs[0]+coeffs[1])

ax = data.plot("N","time")
ax.plot(data["N"], yfit, label="coeffs: "+str(coeffs))
ax.set_yscale("log")
ax.set_ylabel("time [\u03bcs]")
ax.legend()
plt.show()
