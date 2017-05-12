import numpy as np
import pandas as pd
import sys
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt

# Check input
if len(sys.argv)%2 == 1:
    print (len(sys.argv))
    print ("usage: python main.py [data_file] ['DPQ'] [value]")
    exit()
# Get environment variables

#priority
# pr={
#     'DPQ':4
# }

pr = {}
for i in range((int(len(sys.argv)/2-1))):
    pr[sys.argv[2+2*i]]=int(sys.argv[2+2*i+1])

# generate data
data = pd.read_csv(sys.argv[1],",") #"fisrm.csv"
data_size = len(data)

# xac dinh phan phoi cho cac node o input layer
# cpd_tq = TabularCPD(variable='TQ', variable_card=5, values=[[0.2, 0.2,0.2,0.2,0.2]])
# cpd_c = TabularCPD(variable='C', variable_card=5, values=[[0.2, 0.2,0.2,0.2,0.2]])
# cpd_dpq = TabularCPD(variable='DPQ', variable_card=5, values=[[0.2, 0.2,0.2,0.2,0.2]])
# cpd_ou = TabularCPD(variable='OU', variable_card=5, values=[[0.2, 0.2,0.2,0.2,0.2]])

# dinh nghia cau truc mang bayes

# TQ: Test quality, DPQ: design process quality, C: complexity, DI: defects inserted, DFT: defects found intesting
# RD: desidual defects, OU: operational usage, DFO: defects found in operation
model = BayesianModel([('TQ', 'DFT'), ('DPQ', 'DI'), ('C','DI'),('DI','DFT'),('DI','RD'),('DFT','RD'),('RD','DFO'),('OU','DFO')])

model.fit(data, estimator_type=BayesianEstimator, prior_type="BDeu",equivalent_sample_size=10) # default equivalent_sample_size=5
# for cpd in model.get_cpds():
#     print(cpd)
# print model.get_cpds()[2]

infer = VariableElimination(model)
DI_distribution = infer.query(['DI']) ['DI'].values
# DI_distribution = infer.query(['DI'], evidence={'DPQ': 2, 'C': 3, 'TQ': 4,'OU':1})['DI'].values
max_DI = np.argmax(DI_distribution)
print (max_DI)
print (infer.query(['DPQ']) ['DPQ'])
print (model.get_cpds()[1])

plt.figure(1)
# plt.subplot(4, 2, 2)
# plt.plot(DI_distribution)
# plt.title("defects inserted")
# plt.xlabel('number')
# plt.ylabel('probability')

nodes = ['DPQ','C','TQ','DI','DFT','RD','OU','DFO']
Distribution = {}
# print np.sign(0)
for key in pr.keys():
    Distribution[key]=[1-abs(np.sign(pr[key]-i)) for i in range(5)]
    nodes.remove(key)

for key in nodes:
    Distribution[key]=infer.query([key], evidence=pr)[key].values
# Distribution['C']=infer.query(['C'], evidence=pr)['C'].values
# Distribution['DI']=infer.query(['DI'], evidence=pr)['DI'].values
# Distribution['DFT']=infer.query(['DFT'], evidence=pr)['DFT'].values
# Distribution['RD']=infer.query(['RD'], evidence=pr)['RD'].values
# Distribution['DFO']=infer.query(['DFO'], evidence=pr)['DFO'].values
# Distribution['DPQ']=[0,0,0,0,1]
# Distribution['TQ']=[0,0,0,0,1]
# Distribution['OU']=[0,0,0,0,1]
# Distribution = infer.query(['DPQ','C','TQ','DI','DFT','RD','OU','DFO'], evidence={'DPQ': 2, 'TQ': 4,'OU':1})

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
