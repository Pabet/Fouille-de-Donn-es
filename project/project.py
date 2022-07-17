#%%
import numpy as np
import math
import random
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd


def calculate_mean(col):
    
    mean = 0.0
    num_col_e_zero = 0
    for i in col:
        mean += i
        if i == 0.0:
            num_col_e_zero += 1    
    print("values equal to zero: " + str(num_col_e_zero))

    return mean/(len(col)-num_col_e_zero)



def analyse_data(csv_data):

    names = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
    data = np.genfromtxt(csv_data, delimiter=",", names=names)
    data = np.delete(data, 0)

    outliers = []

    for i in names: 
        print(i)
        avg = calculate_mean(data[i])
        print("average: " + str(avg))

        abs_diff = []
        for j in range(0, len(data)):
            a_diff = abs(data[i][j]-avg)
            abs_diff.append(a_diff)

        relative_diff = []   
        for k in range(0, len(abs_diff)):
            r_diff = abs_diff[k]/avg
            relative_diff.append(r_diff)
            if(r_diff > 1.0):
                if k not in outliers:
                    outliers.append(k)

        print("relative error to average: ")
        plt.ylabel('relative error [0:1]')
        plt.xlabel('entries')
        plt.plot(relative_diff) 
        plt.savefig('fig'+str(i)+'.png')
        plt.show()

    print(str(len(outliers)) + " outliers have been found")
    print(outliers)

    return outliers


def remove_outliers(filename, new_filename, idx_list):
    idx = 0
    line_count = 0
    f = open(new_filename, 'w')

    for line in open(filename, 'r').readlines()[1:]:
        if idx not in idx_list:
            f.write(line)
            line_count += 1
        idx += 1    
    f.close()

    return line_count


def count_types(filename):

    types = [0] * 8
    for line in open(filename, 'r').readlines()[1:]:
        line_list = line.split(',')
        types[int(line_list[9].rstrip('\n'))] += 1
    del types[0]

    print("occurences of each type: " + str(types))
    return types


def create_graph(y_values):
    labels = ('1', '2', '3', '4', '5', '6', '7')
    y_pos = np.arange(len(y_values))

    plt.bar(y_pos, y_values, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('occurences')
    plt.xlabel('glass type')
    plt.title('occurences of each glass type')
    plt.savefig('count.png')
    plt.show()



def split_input(filename, seed):
 
    random.seed(seed)
    out1 = open("data_set.txt", 'w')
    out2 = open("test_set.txt", 'w')

    for line in open(filename, 'r').readlines():
        if random.randint(0, 1):
            out1.write(line)
        else:
            out2.write(line)




def get_data(filename):
    """
    seperates a data line into data 
    and labels
    """
    data = []
    label = []

    for line in open(filename, 'r').readlines():

        line_list = line.split(',')
        data.append(line_list[:-1])
        label.append(line_list[len(line_list)-1].rstrip('\n'))
        
    return data, label


def train(data, labels, seed, estimators, max_depth):
    """
    train the model on the test data set
    """
    clf = RandomForestClassifier(max_depth=max_depth, random_state=seed, n_estimators=estimators)
    clf.fit(data, labels)

    return clf


def predict(t_list, clf):
    return clf.predict(t_list)   


def eval_classifier(test_data, test_label, clf):
    
    true_positives = 0.0
    false_positives = 0.0
    true_negatives = 0.0
    false_negatives = 0.0

    prediction_list = clf.predict(test_data)

    for i in range(0, len(prediction_list)):

        if(prediction_list[i] == test_label[i]):
            if(test_label[i]):
                true_positives += 1.0
            else:
                false_positives += 1.0
        else:
            if(test_label[i]):
                false_negatives += 1.0
            else:
                true_negatives += 1.0

    return (true_positives + true_negatives)/len(test_data)

 
idx_list = analyse_data("glass.csv")
remove_outliers("glass.csv", "new_glass.txt", idx_list)
types = count_types("new_glass.txt")
create_graph(types)

split_input("new_glass.txt", 10)
train_x, train_y = get_data("data_set.txt")
test_x, test_y = get_data("test_set.txt")


#%%

best_parameters = (0,0,0)

for k in range(1,6): 
    results = []
    for i in range(1,200):
        clf = train(train_x, train_y, 41, i, k)
        r = eval_classifier(test_x, test_y, clf)
        if(r > best_parameters[0]):
            best_parameters = (r, i, k)
        results.append((i,r))

    zip(*results)
    plt.scatter(*zip(*results))
    plt.title("max depth: " + str(k))
    plt.ylabel('accuracy')
    plt.xlabel('numbe of trees')
    plt.savefig('result'+str(k)+'.pdf')
    plt.show()

print("best parameters: " + str(best_parameters))    
#%%



