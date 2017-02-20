
# coding: utf-8

# In[7]:

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs= np.array([1,2,3,4,5,6], dtype = np.float64)
ys=np.array([5,4,6,5,6,7], dtype = np.float64)

def best_fit_slope_and_intercept(xs,ys):
    m1 = (mean(xs) * mean(ys)) - mean(xs * ys)
    m2 = (mean(xs) * mean(xs)) - mean(xs * xs)
    
    m = m1/m2
    b = mean(ys) - m*(mean(xs)) 
    return m,b
#To determine accuracy of a best fit linear regression, we can use R-Squared theorem or Co-efficient of Determination
def squared_error(ys_orig, ys_line):
    return sum((ys_orig-ys_line)**2)

def coefficinet_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig,y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)

m,b = best_fit_slope_and_intercept(xs,ys)
regression_line = [(m*x) + b for x in xs]

r_squared = coefficinet_of_determination(ys,regression_line)

plt.scatter(xs,ys)
plt.plot(xs,regression_line)
plt.show()
print(r_squared)


# In[ ]:




# In[ ]:



