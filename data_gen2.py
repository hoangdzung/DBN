import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def truncated_normal_randomly_generate(mu = 15, sigma = 30, size = 1000, TQ_data = [], C_data = []):
	random_list = []
	for i in range(size):
		# mu = mu * (C_data[i] + 1) * (4 - TQ_data[i])
		while (True):
			x = np.random.normal(mu, sigma)
			if (x >= 0):
				random_list.append(int(x))
				print(i)
				break
	return np.array(random_list)

def binomial_randomly_generate(n_data = [], p_data = []):
	random_list = []
	for i in range(len(n_data)):
		x = np.random.binomial(n_data[i], (float)(p_data[i]) / 4)
		random_list.append(int(x))
	return np.array(random_list)

def found_probability_generate(n_data = [], p_data = []):
	random_list = []
	for i in range(len(n_data)):
		sigma = 0.001
		if p_data[i] == 0:
			mu = 0.01
		elif p_data[i] == 1:
			mu = 0.1
		elif p_data[i] == 2:
			mu = 0.2
		elif p_data[i] == 3:
			mu = 0.35
		elif p_data[i] == 4:
			mu = 0.5
		
		p = np.random.normal(mu, sigma)
		if p < 0:
			p = 0
		elif p > 1:
			p = 1

		random_list.append(int(n_data[i] * p))
	return np.array(random_list)

def finding_quality_generate(prior_data = []):
	random_list = []
	for i in range(len(prior_data)):
		x = math.ceil(np.random.normal(prior_data[i], 1))
		if x > 4:
			x = 4
		elif x < 0:
			x = 0
		random_list.append(x)
	return np.array(random_list)


def randomly_int(data = []):
	random_list = [np.random.randint(low = 0, high = i + 1) for i in data]	
	return np.array(random_list)

data_size = 100

data = pd.DataFrame(np.random.randint(low = 0, high = 5, size = (data_size, 4)),
                                    columns = ['TQ0','DPQ0', 'C0', 'OU0'])

data['DI0'] = truncated_normal_randomly_generate(size = data_size, TQ_data = data['TQ0'], C_data = data['C0'])
print ('DI0 generated')
# data['DFT0'] = binomial_randomly_generate(n_data = data['DI0'], p_data = data['TQ0'])
# data['DFT0'] =  randomly_int(data['DI0'])
data['DFT0'] = found_probability_generate(data['DI0'], data['TQ0'])
print ('DFT0 generated')
data['RD0'] = data['DI0'] - data['DFT0']
print ('RD0 generated')
# data['DFO0'] = binomial_randomly_generate(n_data = data['RD0'], p_data = data['OU0'])
# data['DFO0'] =  randomly_int(data['RD0'])
data['DFO0'] = found_probability_generate(data['RD0'], data['OU0'])
print ('DFO0 generated')

data['DPQ1'] = data['DPQ0']
print ('DPQ1 generated')
data['C1'] = data['C0']
print ('C1 generated')
data['TQ1'] = finding_quality_generate(data['TQ0'])
print ('TQ1 generated')
data['OU1'] = finding_quality_generate(data['OU0'])
print ('OU1 generated')
data['DI1'] = data['RD0']
print ('DI1 generated')
# data['DFT1'] = binomial_randomly_generate(n_data = data['DI1'], p_data = data['TQ1'])
# data['DFT1'] =  randomly_int(data['DI1'])
data['DFT1'] = found_probability_generate(data['DI1'], data['TQ1'])
print ('DFT1 generated')
data['RD1'] = data['DI1'] - data['DFT1']
print ('RD1 generated')
# data['DFO1'] = binomial_randomly_generate(n_data = data['RD1'], p_data = data['OU1'])
# data['DFO1'] =  randomly_int(data['RD1'])
data['DFO1'] = found_probability_generate(data['RD1'], data['OU1'])
print ('DFO1 generated')

data['DPQ2'] = data['DPQ1']
print ('DPQ2 generated')
data['C2'] = data['C1']
print ('C2 generated')
data['TQ2'] = finding_quality_generate(data['TQ1'])
print ('TQ2 generated')
data['OU2'] = finding_quality_generate(data['OU1'])
print ('OU2 generated')
data['DI2'] = data['RD1']
print ('DI2 generated')
#data['DFT2'] = binomial_randomly_generate(n_data = data['DI2'], p_data = data['TQ2'])
# data['DFT2'] =  randomly_int(data['DI2'])
data['DFT2'] = found_probability_generate(data['DI2'], data['TQ2'])
print ('DFT2 generated')
data['RD2'] = data['DI2'] - data['DFT2']
print ('RD2 generated')
#data['DFO2'] = binomial_randomly_generate(n_data = data['RD2'], p_data = data['OU2'])
# data['DFO2'] =  randomly_int(data['RD2'])
data['DFO2'] = found_probability_generate(data['RD2'], data['OU2'])
print ('DFO2 generated')

data.to_csv("data2.csv", index = False)
