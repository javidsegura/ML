import numpy as np 
import pandas as pd


""" THIS MODULE GENERATES RANDOM DATASET
THAT CAN BE USED TO ANALYZE LINEAR REGRESSION

LIMITATIONS: FUNCTION IS STRICTLY INCREASING"""

# H(x) = 700 + 10x + E

n_sample = 100

y_intercept = 700
parameter_1 = 10

x = np.random.rand(n_sample) * 100
error = np.random.rand(n_sample)* 15 #Error is noise

function = parameter_1 * x + y_intercept - error 

df = pd.DataFrame({"x":x,"y":function})

df.to_csv("/Users/javierdominguezsegura/Programming/Python/Drafts/Scikit/Linear regression/Andrew ng/Multiple linear regression/datasets/my_dataset.csv", 
          header=False, index=False) 