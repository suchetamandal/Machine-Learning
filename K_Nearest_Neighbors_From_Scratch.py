
# coding: utf-8

# In[10]:

import numpy as np
from math import sqrt
import warnings
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

dataset = {'k': [[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_featues = [5,7]

for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1],s=100,color = i)
plt.scatter(new_featues[0],new_featues[1])
plt.show()


# In[16]:

def k_nearest_neighbor(data, predict, k = 3):
    if(len(data)>=k):
        warnings.warn('K is set to a value, that is less than input length')
    distances=[]
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    
    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

result = k_nearest_neighbor(dataset,new_featues,k=3);

[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_featues[0],new_featues[1],color=result)
plt.show()


# In[ ]:



