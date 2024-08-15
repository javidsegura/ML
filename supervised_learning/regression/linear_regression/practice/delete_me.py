import pandas as pd 
import numpy as np

df = pd.read_csv("/Users/javierdominguezsegura/Programming/Python/Drafts/Scikit/Linear regression/Andrew ng/Multiple linear regression/datasets/restaurants.csv", names = ["Population","Price"])

m = int(len(df) * .95)

df = df[m:]

df.to_csv("practice/df_used.csv", index=False)


x_values = df.drop(["Price"], axis = 1)
y_values = df["Price"].values


m = len(x_values)

x_values = np.c_[np.ones(m), x_values]

thetas = np.zeros(x_values.shape[1])

def cost_function(x,y,theta):
      h = np.dot(x,theta)
      J = (np.sum((h-y)**2)) / (2 * m)
      return J

def gradient_descent(x,y,theta):
      m = len(x)
      for i in range(2):
            h = np.dot(x,theta)
            theta = theta - (0.01/m) * (h-y).dot(x)
      return theta

thetas = gradient_descent(x_values, y_values, thetas) # [-1.37922031  0.68009952]

print(thetas)

def predictions(x,theta):
      h = np.dot(x,theta)
      return h
print(predictions(x_values,thetas))