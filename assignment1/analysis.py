import matplotlib.pyplot as plt 
import numpy as np

runs=10

f = open("t.csv")
datastrings = [s for s in f.read().strip().split(" ") if s!=""]
f.close()
data = []
for i in range(len(datastrings)//runs):
    data.append([])
    for j in range(runs):
        data[-1].append(float(datastrings[j+runs*i]))

f = open("t1.csv")
datastrings = [s for s in f.read().strip().split(" ") if s!=""]
f.close()
data1 = []
for i in range(len(datastrings)//runs):
    data1.append([])
    for j in range(runs):
        data1[-1].append(float(datastrings[j+runs*i]))

f = open("t2.csv")
datastrings = [s for s in f.read().strip().split(" ") if s!=""]
f.close()
data2 = []
for i in range(len(datastrings)//runs):
    data2.append([])
    for j in range(runs):
        data2[-1].append(float(datastrings[j+runs*i]))

del datastrings

means=np.mean(data, axis=1)
means1=np.mean(data1, axis=1)
means2=np.mean(data2, axis=1)

x=np.arange(len(means))+10 #10 is mindim

sds=np.std(data, axis=1)
sds1=np.std(data1, axis=1)
sds2=np.std(data2, axis=1)

plt.plot(x, means, label="t")
plt.plot(x, means1, label="t1")
plt.plot(x, means2, label="t2")

plt.fill_between(x, means-sds, means+sds)
plt.fill_between(x, means1-sds1, means1+sds1)
plt.fill_between(x, means2-sds2, means2+sds2)

coeffs = np.polyfit(np.log(x), np.log(means), 1)
y=np.exp(coeffs[0]*np.log(x)+coeffs[1])
plt.plot(x,y,color="red")
print(coeffs[0])

coeffs = np.polyfit(np.log(x), np.log(means1), 1)
y=np.exp(coeffs[0]*np.log(x)+coeffs[1])
plt.plot(x,y,color="cyan")
print(coeffs[0])

coeffs = np.polyfit(np.log(x), np.log(means2), 1)
y=np.exp(coeffs[0]*np.log(x)+coeffs[1])
plt.plot(x,y,color="pink")
print(coeffs[0])

plt.yscale("log")
plt.xscale("log")

plt.legend()
plt.show()
