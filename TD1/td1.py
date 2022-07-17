#%%
import math
from random import random


def average(lst):
    avg = 0.0
    for i in lst:
        avg+=i
    return avg/len(lst)

#print(average([3, 7, 1, 2, 3]))

#%%
def median (lst):
    next_to_middle = 0
    for i in lst:
        counter = -1
        for j in lst:
            if j<=i:
                counter += 1
            elif j>=i:
                counter -= 1    
        if counter == 0:
            return i
        elif abs(counter) == 1:
            next_to_middle = i        
    return next_to_middle+1       

            
#print(median([0, 8, 9]))
#print(median([7, 2, 3, 10, 3, 30])) 

#print(median([2, 4, 5, 6, 173]))
#print(average([2, 4, 5, 6, 173]))
#%%
def occurences(lst):
    dictionnaire = {}
    for i in lst:
        if i in dictionnaire:
            dictionnaire[i] += 1
        else:
             dictionnaire[i] = 1
    return dictionnaire



#print(occurences([2, 4, 4, 5, 6, 173]))
#%%
def unique(lst):
    new_list = []
    for i in lst:
        is_in = False
        for j in new_list:
            if i == j:
                is_in = True
        if is_in == False:
            new_list.append(i)
    return new_list    
#complexity O(len(lst)**2)                
#print(unique([2, 3, 1, 1, 3, 1, 4, 1, 3]))
#%%
#Exercise 2

def squares(lst):
    new_lst = []
    for i in lst:
        new_lst.append(i*i)
    return new_lst

#print(squares([2, 4, 5, 6, 173]))
#%%

def stddev(lst):
    avg = average(lst)
    standard_deviation = 0.0
    for i in lst:
        standard_deviation += (i-avg)**2
    return math.sqrt(standard_deviation/len(lst))

#print(stddev([5, 4, 7, 4, 4, 2, 5, 9]) )
#print(stddev([20, 20, 20]))
#%%
#Exercise 3

def uniform():
    n = random()
    if n <= 0.5:
        return 0
    else:
        return 1

#print(uniform())

n = 0
avg = 0.0
num_tries = 1000000
while (n < num_tries):
    avg += uniform()
    n += 1
#print(avg/num_tries)

#%%
#3.2
def uniform_2():
    n = random()
    if n <= 1/6:
        return 0
    elif n <= 2/6:
        return 1
    elif n <= 3/6:
        return 2
    elif n <= 4/6:
        return 3
    elif n <= 5/6:
        return 4
    else:
        return 5       
#%%
#3.3

def uniform_3(n):
    p = random()
    return (n*p)

n = 10
avg = 0.0
num_tries = 1000000
while (n < num_tries):
    avg += uniform()
    n += 1
#print(avg/num_tries)
#%%
#3.4
#def exam_success(n, p):




#%%
# Exercise 4
import random
import math 

def monty_hall(change):
    n = random.random()
    if n <= 1/3:
        door = 0
    elif n <= 2/3:
        door = 1
    else:
        door = 2
    #print("candidate picked door", door)

    m = random.random()
    if m <= 1/3:
        reward = 0
    elif m <= 2/3:
        reward = 1
    else:
        reward = 2    
    #print("(reward is behind door", reward, ")")

    if door == reward:
        z = random.random()
        if z <= 1/2:
            if door == 0:
                reveil = 1
            elif door == 1:
                reveil = 0
            else:
                reveil = 0    
        else:
            if door == 0:
                reveil = 2
            elif door == 1:
                reveil = 2
            else:
                reveil = 1  
    else:
        if door + reward == 3:
            reveil = 0    
        if door + reward == 2:
            reveil = 1
        else:
            reveil = 2 
    #print ("the host reveils door", reveil)

    if change:
        if door+reveil==3:
            change = 0
        elif door+reveil==2:
            change = 1
        else:
            change = 2
        #print("test player changes to door", change)
        if change == reward:
            #print("the player wins a car")
            return 1
        else:
            #print("the player loses")
            return 0 

    else:
        #print("the player doesn't change the door")
        if door == reward:
            #print("the player wins a car")
            return 1
        else:
            #print("the player loses")
            return 0 

#monty_hall(True)               
#%%
#4.2

def monty_hall_simulation(n):
    i = 0
    freq_change = 0.0
    freq_nochange = 0.0
    while i < n:
        freq_change += monty_hall(True)
        freq_nochange += monty_hall(False)
        i+=1
    return (freq_change/n, freq_nochange/n)

#print(monty_hall_simulation(10))        