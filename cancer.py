import numpy as np
import pandas as pd
data=pd.read_csv('cancer1.csv')
print(data)

X = data.iloc[:,:8]
y = data.iloc[:,9]
y = y[:,np.newaxis]
print(X)
print(y)
X.astype(float)

#Mean Normalization
X = (X-np.mean(X))/np.std(X)
print(X)

#Activation function
def sigmoid(s):
    return 1/(1+np.exp(-s))

#derivative fn
def sigmoid_der(s):
    return s*(1-s)

epochs=10000
no_of_hidden=16
no_of_ip=X.shape[1]
lr=0.1
no_of_op=1

wh=np.random.uniform(size=(no_of_ip,no_of_hidden))
bh=np.random.uniform(size=(1,no_of_hidden))
wout=np.random.uniform(size=(no_of_hidden,no_of_op))
bo=np.random.uniform(size=(1,no_of_op))

for i in range(epochs):
    hidden1_ip=np.dot(X,wh)
    hidden1_ip=hidden1_ip + bh
    activation1=sigmoid(hidden1_ip)
    output1=np.dot(activation1,wout)
    output1=output1+bo
    output=sigmoid(output1)
    
    e=y-output
    slope_output=sigmoid_der(output)
    slope_hidden=sigmoid_der(activation1)
    delta_output=e*slope_output
    error_hidden=np.dot(delta_output,wout.T)
    delta_hidden=error_hidden*slope_hidden
    wout=wout+np.dot(activation1.T,delta_output)*lr
    wh=wh+np.dot(X.T,delta_hidden)*lr
    bh=np.sum(delta_output,axis=0)*lr
    bout=np.sum(delta_hidden,axis=0)*lr
    
print(wh)
print(wout)
print(output)




        