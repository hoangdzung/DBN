import numpy as np
import pandas as pd
import sys
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt

model = DynamicBayesianNetwork()

list_edges = [(('DPQ', 0), ('DI', 0)), (('C', 0), ('DI', 0))]

for i in range(3):
    list_edges += [(('DI', i), ('DFT', i)),
    			(('TQ', i), ('DFT', i)),
    			(('DFT', i), ('RD', i)),
    			(('RD', i), ('DFO', i)),
    			(('OU', i), ('DFO', i))]
    if (i == 2):
    	break
    list_edges += [(('RD', i), ('DI', i + 1)),
                (('TQ', i), ('TQ', i + 1)),
                (('OU', i), ('OU', i + 1))]

model.add_edges_from(list_edges)
print(model.edges())
print(model.nodes())

