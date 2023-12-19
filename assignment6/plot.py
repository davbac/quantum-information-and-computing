import pandas as pd
import matplotlib.pyplot as plt 

data = pd.read_csv("times.txt")

part1 = data[data["type"]=="s"]
part2 = data[data["type"]=="c"]
part3 = data[data["type"]=="f"]

fig,ax = plt.subplots()

#part1.plot("N", "time", ax=ax, label="separable state")
#part2.plot("N", "time", ax=ax, label="complete state")
#part3.plot("N", "time", ax=ax, label="complete state from separable")

ax.fill_between(part1["N"], part1["time"]-part1["sd"], part1["time"]+part1["sd"], label="separable state")
ax.fill_between(part2["N"], part2["time"]-part2["sd"], part2["time"]+part2["sd"], label="complete state")
ax.fill_between(part3["N"], part3["time"]-part3["sd"], part3["time"]+part3["sd"], label="complete state from separable")

ax.set_yscale("log")
ax.set_xlabel("N")
ax.set_ylabel("time")

plt.legend(loc="upper left")
plt.show()
