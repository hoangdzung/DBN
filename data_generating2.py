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

data_size = 100

data = pd.DataFrame(np.random.randint(low = 0, high = 5, size = (data_size, 4)),
                                    columns = ['TQ','DPQ', 'C', 'OU'])

data['DI'] = positive_normal_random_gen_1(mu = 15, sigma = 30, size = data_size)
print ('DI done')
data['DFT'] = positive_normal_random_gen_2(mu = -60, sigma = 35, size = data_size, data = data['DI'])
print ('DFT done')
data['RD'] = data['DI'] - data['DFT']
print ('RD done')
data['DFO'] = positive_normal_random_gen_2(mu = -60, sigma = 35, size = data_size, data = data['RD'])
print ('DFO done')

data.to_csv("data2.csv", index = False)
