import matplotlib.pyplot as plt
import sys


def plot_bar(filename):
    

    exec_file = open(filename,'r').readlines()
    stream_ids = [1,2,4,8,12,16]
    num_plots=len(stream_ids)
    counter = 0
    filename = filename.split("/")[1]
    filename = filename[:-4]+".pdf"
    for i in range(0,num_plots):
        x=[]
        y=[]
        for j in range(0,stream_ids[i]):
            line = exec_file[counter]
            stream_id,latency = line.strip("\n").split(":")
            x.append(int(stream_id.split(" ")[1]))
            y.append(float(latency))
            counter +=1
        plt.barh(x, y, color='green')
        plt.yticks(x, [str(x_i) for x_i in x])
        plt.xlabel("Execution Time")
        plt.ylabel("Stream IDs")
        plot_filename = "./plots/num_streams="+str(stream_ids[i])+"_"+filename
        plt.savefig(plot_filename, bbox_inches='tight')
    

def plot_line(filename):
    x = []
    y = []
    exec_file = open(filename,'r').readlines()
    for line in exec_file:
        stream_id,latency = line.strip("\n").split(":")
        x.append(int(stream_id))
        y.append(float(latency))

    plt.xticks(x, ["1/"+str(x_i) for x_i in x])    
    plt.xlabel("Fraction")
    plt.ylabel("Copy Time")
    plt.plot(x,y)
    plt.show()

if __name__ == "__main__":
    if(sys.argv[1]=="wallclock"):

        speedup_file = open(sys.argv[2]).readlines()
        x=[2,4,8,12,16]
        y = []
        for line in speedup_file:
            seq,asyn = line.strip("\n").split(" ")
            speedup = float(seq)/float(asyn)
            y.append(speedup)
            filename = sys.argv[2].split("/")[1]
            filename = "./plots/"+filename[:-4]+".pdf"
            

        plt.xlabel("Number of Streams")
        plt.ylabel("Speedup (Fine vs Coarse)")
        plt.plot(x,y)
        plt.savefig(filename, bbox_inches='tight')

    if(sys.argv[1]=="stream"): 
        plot_bar(sys.argv[2])   




