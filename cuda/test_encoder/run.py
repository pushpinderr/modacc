import os
import sys
import subprocess
import numpy as np
import pandas as pd 

nq = [1,2,4,8,12,16,24]
MIN_ARGS = 4
assert len(sys.argv) > MIN_ARGS, "executable path or mandatory args missing"

cmd = ""
for arg in sys.argv[1:]:
    cmd += arg+" "
op=sys.argv[1]
print(op)
for qs in nq:
    subprocess.getoutput(f"{cmd} {qs}")
    print(f"{cmd} {qs}")
coarse = float(open("queue_size=1.txt").read())
os.remove("queue_size=1.txt")
fine = []

for qs in nq[1:]:
    f = open(f"queue_size={qs}.txt")
    fine.append(float(f.read()))
    f.close()
    # os.remove(f"queue_size={qs}.txt")

for qs,t in zip(nq, [coarse]+fine):
    print(f"t({qs})={t}")

speedups = coarse/np.array(fine)
for i,speed_up in enumerate(speedups):
    print(f"speed up is {speed_up} for queue_size={nq[i+1]}")

df = []
for qs,su in zip(nq[1:], speedups):
    df.append({ "queue_size" : qs, "speed_up" : su })
output_file=op+".csv"
pd.DataFrame(df).to_csv(output_file, index=False)
