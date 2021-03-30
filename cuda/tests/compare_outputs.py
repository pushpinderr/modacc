import sys
import json
import pandas as pd 
l1 = json.load(open("../dump/input_norm.json"))
l2 = json.load(open("../dump/bert_norm.json"))

try:
    eps = sys.argv[1]
except:
    eps = 1e-5
print(f"using tolerance of {eps} while comparing")

match_ctr = 0
mismatches = []
index = 0
for x_1,x_2 in zip(l1,l2):
    if abs(x_1-x_2) <= eps:  
        match_ctr += 1
    index += 1
    mismatches.append((index, x_1, x_2))

print(f"{match_ctr} elements match out of {len(l1)}.")
if match_ctr == len(l1):
    print(f"perfect match!")
else: 
    print(f"there are {len(l1)-match_ctr} matches!")
    df = []
    for i,py,cu in mismatches:
        df.append({"index" : i, "expected" : py, "actual" : cu})
    
    df = pd.DataFrame(df)
    df.to_csv("../dump/mismatches.csv", index=False)
