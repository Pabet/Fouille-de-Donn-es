#%%
import random
import numpy 

def simulation_coin(num_exp, num_coins_per_exp):
    distribution = numpy.zeros((1001,), dtype=int)
    coin_toss_results = numpy.zeros((num_exp,), dtype=int)

    for n in range(num_exp):
        for _ in range(num_coins_per_exp):
            coin = random.sample([0,1], 1)
            if coin[0] == 1:
                coin_toss_results[n] += 1

    for i in range(1001):
        for n in range(num_exp):
            ratio = coin_toss_results[n] / num_coins_per_exp
            if i == int(ratio * 1000):
                distribution[i] += 1
    
    return distribution

#print(simulation_coin(1000, 100))

#%%
import math

def proba_normal_var_above(value):
    return 1.0 - (0.5*(1.0+math.erf(value/math.sqrt(2.0))))

#print(proba_normal_var_above(6.2))    

#%%
import math

def proba_sample_mean_above_true_mean_by_at_least(sample, delta):
    mean = 0.0
    for value in sample:
        mean += value
    mean /= len(sample)
    
    variance = 0.0
    for value in sample:
        variance += (value - mean)**2
    variance /= (len(sample)-1)
    ecart_type = math.sqrt(variance)
   
    if ecart_type == 0.0:
        if delta < 0.0:
            return 1
        elif delta == 0:
            return 0.5 
        else:
            return 0   

    #vrai_moyenne = mean - delta

    return proba_normal_var_above(math.sqrt(len(sample))*(delta/ecart_type))
    
#list = [183,160,170,175,178,187,192,163,243]
data = [3,3]
error = 0
#print(proba_sample_mean_above_true_mean_by_at_least(data, error))


#%%
import numpy as np
import math

def standard_percentile(p):
    axis = np.linspace(-4.0, 4.0, num=1000)
    return binary_search_rec(axis, p, -4.0, 4.0) 

def binary_search_rec(array, prob, start, end):
    if start > end:
        return -1

    if start<0 and end<0:
        mid = start + (start - end) / 2
    else:
        mid = (start + end) / 2

    if math.isclose(prob , 1-proba_normal_var_above(mid), rel_tol=1e-02):
        return mid

    #print("left: ", start, "right: ", end , "mid: ", mid, "prob: ", 1-proba_normal_var_above(mid))
    if prob < 1.0-proba_normal_var_above(mid):
        return binary_search_rec(array, prob, start, mid)
    else:
        return binary_search_rec(array, prob, mid, end)       

print(standard_percentile(0.16))
#%%


def confidence_interval_of_mean(sample, pvalue):
    mean = 0.0
    for value in sample:
        mean += value
    mean /= len(sample)    

