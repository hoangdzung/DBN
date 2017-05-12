import numpy as np
import pandas as pd
import sys
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt

argv = 'main.py fisrm_data_2.csv DPQ 0 C 4 TQ 0'
argv = argv.split()

#pr = {'C': 4, 'DPQ': 0, 'TQ': 0}
pr = {}
for i in range(len(argv)/2-1):
    pr[argv[2+2*i]]=int(argv[2+2*i+1])

#read input file
data = pd.read_csv('fisrm.csv')
#data_size = 10000
data_size = len(data)

#define our Bayesian model
model = BayesianModel([('TQ', 'DFT'), ('DPQ', 'DI'), ('C','DI'),('DI','DFT'),('DI','RD'),('DFT','RD'),('RD','DFO'),('OU','DFO')])
#learn all the conditional probabilities for each node in our models using input data as training set
model.fit(data, estimator_type=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)

# #http://pgmpy.org/inference.html
inference = VariableElimination(model)
#
# nodes = ['DPQ','C','TQ','DI','DFT','RD','OU','DFO']
# Distribution = {}
# for key in pr.keys():
#     Distribution[key]=[1-abs(np.sign(pr[key]-i)) for i in range(5)]
#     #Distribution = {'C': [0, 0, 0, 0, 1], 'DPQ': [1, 0, 0, 0, 0], 'TQ': [1, 0, 0, 0, 0]}
#     nodes.remove(key)
#     #nodes = ['DI', 'DFT', 'RD', 'OU', 'DFO']
#
# for key in nodes:
#     Distribution[key]=inference.query([key], evidence=pr)[key].values

mm = model.to_markov_model()
mm.nodes()
mm.edges()

