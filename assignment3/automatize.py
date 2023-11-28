import sys 
from multiprocessing import Pool
import subprocess
import os.path

# dimensions of matrix to use
Ns = [10,20,50,100,200,500,1000,2000]


def parallel_execute(Ns):
    #ready the results array
    results=[""]*len(Ns)
    a_results=[]
    
    # do in parallel, to speed up
    with Pool() as p:
        for i in Ns:
            a_res = p.apply_async(subprocess.run, args=["./mymatmul.x "+str(i)], kwds={"text":True, "capture_output":True, "shell":True})
            a_results.append(a_res)
        
        for i in range(len(a_results)):
            #a_results[i].wait()
            results[i]=a_results[i].get().stdout.strip().split(" ")
            results[i] = [float(j) for j in results[i] if j!=""]
    
    # write to file
    paths = ["./t.csv", "./t1.csv", "./t2.csv"]
    for i, p in enumerate(paths):
        file_exists = os.path.exists(p)
        
        f = open(p, "a")
        
        if not file_exists:
            f.write(", ".join([str(j) for j in Ns]))
            f.write("\n")
        
        f.write(", ".join([str(results[j][i]) for j in range(len(Ns))]) )
        f.write("\n")
        
        f.close()


for i in range(13):
    parallel_execute(Ns)
