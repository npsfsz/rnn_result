from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet,UnsupervisedDataSet
from pybrain.structure import LinearLayer
import numpy
import sys

#cassandra_new

window = 200
ds = SupervisedDataSet(window, window)

start_file=int(sys.argv[1])
chunk_size=int(sys.argv[2])

training_data=list()
#print start_file,chunk_size
temp_start_file=start_file
for x in range(1, chunk_size+1):
    filename="./signal3/signal"+str(temp_start_file)+".txt"
    #print filename
    #print start_file,x
    temp_start_file=int(start_file)+int(x)
    training_data.append(numpy.loadtxt(filename))

#here training done, now build the network

#print len(training_data)
data_points=len(training_data)
for x in range(0, data_points-1):
    #print x
    ds.addSample(training_data[x],training_data[x+1])

net = buildNetwork(window, window-1, window, outclass=LinearLayer,bias=True, recurrent=True)                                                                   
trainer = BackpropTrainer(net, ds)                                              
trainer.trainEpochs(100)                                                        
                                                                                
ts = UnsupervisedDataSet(window,)                                               
#add the sample to be predicted                                                 
ts.addSample(training_data[data_points-1])                                                            
                                                                                
result = net.activateOnDataset(ts)  

filename="./temp_signals/"+sys.argv[1]+sys.argv[2]
target = open(filename, 'w+')

for elem in result[0]:
    target.write(str(elem)+"\n")

target.close()
#for elem in result[0]:                                                        
#  print elem  

'''
filename = "./"+sys.argv[1]+"/cassandra_new"


train1 = numpy.loadtxt(filename+"1")
train2 = numpy.loadtxt(filename+"2")
train3 = numpy.loadtxt(filename+"3")
train4 = numpy.loadtxt(filename+"4")



ds.addSample(train1,train2)
ds.addSample(train2,train3)
ds.addSample(train3,train4)

 
 
net = buildNetwork(window, window-1, window, outclass=LinearLayer,bias=True, recurrent=True)
trainer = BackpropTrainer(net, ds)
trainer.trainEpochs(100)

ts = UnsupervisedDataSet(window,)
#add the sample to be predicted
ts.addSample(train3)

result = net.activateOnDataset(ts)
#for elem in result[0]:
#        print elem 
'''
