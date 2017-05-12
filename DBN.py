import numpy as np
import pandas as pd
import sys
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt

if len(sys.argv)%2 == 1:
    print (len(sys.argv))
    print ("usage: python main.py [data_file] ['DPQ'] [value]")
    exit()

pr = {}
for i in range((int(len(sys.argv)/2-1))):
    pr[sys.argv[2 + 2*i]]=int(sys.argv[2 + 2*i + 1])

data = pd.read_csv(sys.argv[1], ",")
data_size = len(data)

# pr = {}
# data = pd.read_csv('data.csv') #"fisrm.csv"
# data_size = len(data)
model = BayesianModel()
list_edges = [('TQ', 'DFT'), ('DPQ', 'DI'), ('C','DI'),('DI','DFT'),('DI','RD'),('DFT','RD'),('RD','DFO'),('OU','DFO')]

model.add_edges_from(list_edges)
model.fit(data, estimator_type = BayesianEstimator, prior_type = "BDeu", equivalent_sample_size = 10)
for edge in model.edges():
    print(edge)
    print("\n")
infer = VariableElimination(model)

nodes = model.nodes()
Distribution = {}

for key in pr.keys():
    Distribution[key] = [1 - abs(np.sign(pr[key] - i)) for i in range(5)]
    nodes.remove(key)
    print('pr done')

for key in nodes:
    Distribution[key] = infer.query([key], evidence = pr)[key].values
    print('done' + key)

print(Distribution['DPQ'])
plt.subplot(4, 2, 1)
plt.bar([1,2,3,4,5], Distribution['DPQ'])
plt.xticks([1.5,2.5,3.5,4.5,5.5], ['very low','low','medium','high','very high'])
plt.title("design process quality")
# plt.xlabel('number')
plt.ylabel('probability')

plt.subplot(4, 2, 2)
plt.bar([1,2,3,4,5],Distribution['C'])
plt.xticks([1.5,2.5,3.5,4.5,5.5], ['very low','low','medium','high','very high'])
plt.title("complexity")
# plt.xlabel('number')
plt.ylabel('probability')


plt.subplot(4, 2, 3)
plt.bar([1,2,3,4,5],Distribution ['TQ'])
plt.xticks([1.5,2.5,3.5,4.5,5.5], ['very low','low','medium','high','very high'])
plt.title("Test quality")
# plt.xlabel('number')
plt.ylabel('probability')

plt.subplot(4, 2,4)
plt.plot(Distribution ['DI'])
plt.title("defects inserted")
# plt.xlabel('number')
plt.ylabel('probability')

plt.subplot(4, 2,5)
plt.plot(Distribution ['DFT'])
plt.title("defects found in testing")
# plt.xlabel('number')
plt.ylabel('probability')

plt.subplot(4, 2,6)
plt.plot(Distribution['RD'])
plt.title("residual defects")
# plt.xlabel('number')
plt.ylabel('probability')

plt.subplot(4, 2,7)
plt.bar([1,2,3,4,5],Distribution ['OU'])
plt.xticks([1.5,2.5,3.5,4.5,5.5], ['very low','low','medium','high','very high'])
plt.title("operational usage")
# plt.xlabel('number')
plt.ylabel('probability')

plt.subplot(4, 2,8)
plt.plot(Distribution ['DFO'])
plt.title("defects found in operation")
# plt.xlabel('number')
plt.ylabel('probability')

plt.subplots_adjust(hspace = 0.5)
plt.show()
