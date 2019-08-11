import numpy as np
import sys
import csv
import math
data = np.genfromtxt(sys.argv[1], delimiter = ',')
data2 = np.genfromtxt(sys.argv[1], delimiter = ',', dtype = str)
data1 = [x[0:8] for x in data]
label = [x[8] for x in data2]
testing_data = np.genfromtxt(sys.argv[2], delimiter = ',')


"""Euclidean distance function"""
def euclidean(x2,y):
    dist = []
    for x in x2:
	   s = squaret(x, y, 8)
       sq = math.sqrt(s)
       dist.append(sq)
    return distance

def square (x,y num):
	s = 0  
	for i in  num:
		s += (x[i]-y[i])**2
	return s

"""K-nearest neighbour function"""

def knn(data1, k, pt):
    dist = euclidean(data1, pt)
    distance = np.argsort(dist) 
    decisions = np.zeros(2, dtype=float)
    for i in range(k):
        if label[distance[i]] == "yes":
            decisions[0] += 1
        elif label[distance[i]] == "no":
            decisions[1] += 1 
    if (decisions[0] == decisions[1]):
            return decisions[0]
    else:
            return np.argmax(decisions)

def knn_results(testing_data, k):
    result = np.zeros(len(testing_data), dtype=float)
    for i in range(len(testing_data)):
       result[i] = (knn(data1, k, testing_data[i]))
       results = ["" for i in result]
    for i in range(len(results)):
        if result[i] == 0:
            print("yes")
        elif result[i] == 1:
            print("no")

"""Naive-bayes function"""

def nb(pt):
  counts = np.zeros(2)
  for i in range(len(data)):
        if label[i] == "yes":
            counts[0] += 1
        elif label[i] == "no":
            counts[1] += 1

def standard_dev(x, mean, n):
    a = np.zeros(8)
    for line in x:
        for i in range(len(line)):
            a[i] += ((line[i] - mean[i])**2)
    sd = np.zeros(8)
    for j in range(len(a)):
        sd[j] = np.sqrt(a[j]/(n-1))
    return sd
def cal_nb(sd, mean, pt):
    prob = 1.0
    for i in range(len(pt)):
        j = 1/(sd[i] * np.sqrt(2 * math.pi))
        k = -((pt[i] -mean[i])**2)/ (2 * sd[i]**2)
        prob *= (j * math.exp(k))
    return prob


def naivebayes(pt):
    counts = np.zeros(2)
    yes = []
    no = [] 
    sum0 = np.zeros(8)
    sum1 = np.zeros(8)
    for i in range(len(data)):
        if label[i] == "yes":
            yes.append(data1[i])
            for j in range(len(sum0)):
               sum0[j] += data1[i][j]
        elif label[i] == "no":
            no.append(data1[i])
            for j in range(len(sum0)):
               sum1[j] += data1[i][j]
    sum0_mean = [x/len(yes) for x in sum0]
    sum1_mean = [x/len(no) for x in sum1]
 
    sd_yes = standard_dev(yes, sum0_mean, len(yes))
    sd_no = standard_dev(no, sum1_mean, len(no))

    nb0 = cal_nb(sd_yes, sum0_mean, pt)
    nb1 = cal_nb(sd_no, sum1_mean, pt)
    
    nb0_cal = nb0 * (len(yes)/len(data1))
    nb1_cal = nb1 * (len(no)/len(data1))
    if (nb0_cal >= nb1_cal):
       return "yes"
    else:
       return "no"
   

def naivebayes_results(testing_data):
    results = ["" for i in testing_data]
    for i in range(len(testing_data)):
        results[i] = naivebayes(testing_data[i])
        print(results[i])
    
k = ""
if sys.argv[3] == "NB":
       naivebayes_results(testing_data)
else:
    l = list(sys.argv[3])
    for i in range (len(l)):
        if l[i] == 'N':
            break
        else:
            k = k + l[i]
    knn_results(testing_data, int(k))



