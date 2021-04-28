import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

assert len(sys.argv) > 1, "need to supply filename"

dfs = []
for fname in [sys.argv[1]]:
    dfs.append(pd.read_csv(fname))
    try:
        del dfs[-1]["Unnamed: 0"]
    except:
        pass 
    # print(dfs[-1])

col = list(dfs[0].columns)
y = np.array([np.array(df[col[1]]) for df in dfs])
y = 100-100*np.mean(y, axis=0)#-100
x = [i for i in range(len(y))]

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

fig, ax = plt.subplots()
bars = ax.bar(x,y,color='green')
autolabel(bars)
plt.title(f"Fine-grained vs Coarse-grained (batch size={sys.argv[2]})")
plt.xlabel("number of queues")
plt.ylabel("Percentage of slow down (%)")
plt.xticks(x, labels=list(dfs[0][col[0]]))
step_size = 20
if len(sys.argv) > 3:
    step_size = int(sys.argv[3])
# yl = [i*step_size for i in range(int(max(y)/step_size)+2)]
# print(yl)
# plt.yticks(yl, labels=yl)
plt.savefig(f"norm_{sys.argv[2]}.png")