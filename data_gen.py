import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def truncated_normal_randomly_generate(mu = 15, sigma = 30, size = 1000, TQ_data = [], C_data = []):
	random_list = []
	for i in range(size):
		mu = (float)(mu) / 16 * (C_data[i] + 0.5) * (4.5 - TQ_data[i])
		while (True):
			x = np.random.normal(mu, sigma)
			if (x >= 0):
				random_list.append(int(x))
				print(i)
				break
	return np.array(random_list)

# def binomial_randomly_generate(n_data = [], p_data = []):
# 	random_list = []
# 	for i in range(len(n_data)):
# 		x = np.random.binomial(n_data[i], (float)(p_data[i]) / 4)
# 		random_list.append(int(x))
# 	return np.array(random_list)
# data_size = 1000
def binomial_randomly_generate(n_data = [], p_data = []):
	random_list = []
	for i in range(len(n_data)):
		# x = np.random.binomial(n_data[i], (float)(p_data[i]) / 4)
		# random_list.append(int(x))
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
		while (True):
			p = np.random.normal(mu, sigma)
			if (0 <= p <= 1):
				random_list.append(int(n_data[i] * p))
				print(i)
				break
	return np.array(random_list)

data_size = 5000
data = pd.DataFrame(np.random.randint(low = 0, high = 5, size = (data_size, 4)),
                                    columns = ['TQ','DPQ', 'C', 'OU'])

data['DI'] = truncated_normal_randomly_generate(size = data_size, TQ_data = data['TQ'], C_data = data['C'])
print ('DI generated')
data['DFT'] = binomial_randomly_generate(n_data = data['DI'], p_data = data['DPQ'])
print ('DFT generated')
data['RD'] = data['DI'] - data['DFT']
print ('RD generated')
data['DFO'] = binomial_randomly_generate(n_data = data['RD'], p_data = data['OU'])
print ('DFO generated')

data.to_csv("data.csv", index = False)
