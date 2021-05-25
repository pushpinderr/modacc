from matplotlib import pyplot as plt
import numpy as np
import sys
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
plt.rcParams['font.size']='20'

# layer_dict={"layer_norm":0.041658,"qkv_linear":0.037400,"launch_bias_add_transform_0213":0.009271,"attention_scores":0.001583,"softmax":0.129802,"attention_probability_dropout":0.020485,"attention_context":0.003015,"attention_out_linear":0.007071,"attention_output_dropout":0.002689,"attention_layer_norm":0.011060,"1st_feed_forward_layer":0.009706,"gelu":0.010536,"2nd_feed_forward_layer":0.050024,"layer_output_dropout":0.002705}
layer_dict ={}
encoder_file_contents=open(sys.argv[1],'r').readlines()
for line in encoder_file_contents:
	k,v=line.strip("\n").split(":")
	layer_dict[k]=float(v)
x = []
y = []

for l in layer_dict:
	layer_dict[l]=layer_dict[l]*1000
layer_type_dict={}
layer_type_dict["Norm"]=layer_dict["layer_norm"]+layer_dict["attention_layer_norm"]
layer_type_dict["GEMM"]=layer_dict["qkv_linear"]+layer_dict["attention_out_linear"]+layer_dict["1st_feed_forward_layer"]+layer_dict["2nd_feed_forward_layer"]
layer_type_dict["stridedGEMM"]=layer_dict["attention_scores"]+layer_dict["attention_context"]
layer_type_dict["Dropout"]=layer_dict["attention_probability_dropout"]+layer_dict["attention_output_dropout"]+layer_dict["layer_output_dropout"]
layer_type_dict["Softmax"]=layer_dict["softmax"]
layer_type_dict["GELU"]=layer_dict["gelu"]
layer_type_dict["Mem"]=layer_dict["launch_bias_add_transform_0213"]+layer_dict["launch_transform4d_0213"]

for l in layer_type_dict:
	x.append(l)
	y.append(layer_type_dict[l])

ax.pie(y, labels = x,autopct='%1.2f%%')
plt.show()

