import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def positive_normal_random_gen_1(mu = 15, sigma = 30, size = 1000):
    count = 0
    ran_list = []
    while (count < size):
        a = np.random.normal(mu, sigma)
        if a >= 0:
            ran_list.append(int(a))
            count += 1
    return np.array(ran_list)

def positive_normal_random_gen_2(mu = 15, sigma = 30, size = 1000, data = []):
    count = 0
    ran_list = []
    while (count < size):
        a = np.random.normal(mu, sigma)
        if ((a >= 0)&(a <= data[count])):
            ran_list.append(int(a))
            count += 1
            if (count % 100 == 99):
                print(count)
    return np.array(ran_list)

data_size = 10

data = pd.DataFrame(np.random.randint(low = 0, high = 5, size = (data_size, 10)),
                                    columns = ['TQ0','DPQ0', 'C0', 'OU0',
                                            'TQ1','DPQ1', 'OU1',
                                            'TQ2','DPQ2', 'OU2'])
data['C1'] = data['C0']
data['C2'] = data['C0']

data['DI0'] = positive_normal_random_gen_1(mu = 15, sigma = 30, size = data_size)
print ('DI0 done')
data['DFT0'] = positive_normal_random_gen_2(mu = -60, sigma = 35, size = data_size, data = data['DI0'])
print ('DFT0 done')
data['RD0'] = data['DI0'] - data['DFT0']
print ('RD0 done')
data['DFO0'] = positive_normal_random_gen_2(mu = -60, sigma = 35, size = data_size, data = data['RD0'])
print ('DFO0 done')

data['DI1'] = data['RD0']
print ('DI1 done')
data['DFT1'] = positive_normal_random_gen_2(mu = -60, sigma = 35, size = data_size, data = data['DI1'])
print ('DFT1 done')
data['RD1'] = data['DI1'] - data['DFT1']
print ('RD1 done')
data['DFO1'] = positive_normal_random_gen_2(mu = -60, sigma = 35, size = data_size, data = data['RD1'])
print ('DFO1 done')

data['DI2'] = data['RD1']
print ('DI2 done')
data['DFT2'] = positive_normal_random_gen_2(mu = -60, sigma = 35, size = data_size, data = data['DI2'])
print ('DFT2 done')
data['RD2'] = data['DI2'] - data['DFT2']
print ('RD2 done')
data['DFO2'] = positive_normal_random_gen_2(mu = -60, sigma = 35, size = data_size, data = data['RD2'])
print ('DFO2 done')

data.to_csv("data.csv", index = False)
