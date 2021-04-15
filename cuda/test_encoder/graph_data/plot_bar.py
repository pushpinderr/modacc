import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

assert len(sys.argv) > 1, "need to supply filename"

dfs = []
for fname in sys.argv[1:]:
    dfs.append(pd.read_csv(fname))
    try:
        del dfs[-1]["Unnamed: 0"]
    except:
        pass 
    # print(dfs[-1])

col = list(dfs[0].columns)
y = np.array([np.array(df[col[1]]) for df in dfs])
y = 100*np.mean(y, axis=0)-100
x = [i for i in range(len(y))]

plt.bar(x,y)
plt.title("Percent speedup of fine-grained implementation with queue size (averaged over 5 runs)")
plt.xlabel(col[0])
plt.ylabel(col[1])
plt.xticks(x, labels=list(dfs[0][col[0]]))
plt.show()