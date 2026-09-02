import copy
import pandas as pd
import numpy as np

import pandas as pd



df = pd.read_csv("HistoricalData_1726367135218.csv")

price_columns = ["Close/Last", "High", "Low", "Open"]

for columns in price_columns:
    df[columns] = df[columns].str.replace('$','',regex=False).astype(float)
df["Volume"] = df["Volume"].astype(float)

X_train = df[["Close/Last", "High", "Low", "Open","Volume"]].iloc[:-1]
Y_train = df["Close/Last"].shift(-1).iloc[:-1]

X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_scaled = (X_train - mean) / std
#
# print(X_scaled.mean(axis=0))
# print(X_scaled.std(axis=0))
#%%
def compute_cost(X_train, Y_train,w,b):
    m = X_train.shape[0]
    cost = 0
    for i in range(m):
        f_wb_i = np.dot(X_train[i],w)+b
        cost+=(f_wb_i-Y_train[i])**2
    cost = cost / (2*m)
    return cost

#%%

#%%
def compute_gradient(X,y,w,b):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = (np.dot(X[i],w)+b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j]+ err*X[i,j]
        dj_db = dj_db + err
    dj_db = dj_db/ m
    dj_dw = dj_dw/ m
    return dj_db, dj_dw
#%%
init_w = np.zeros(5,)
init_b = 0
iterations = 1000
alpha = 0.3
w_final,b_final,cost = gradient_descent(X_scaled,Y_train,init_w,init_b,compute_cost,compute_gradient,alpha,iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m = X_train.shape[0]
for i in range(m):
    print(f"prediction:{ np.dot(X_scaled[i],w_final)+b_final:0.2f}, target value:{Y_train[i]}")
print(f"cost: {cost}")
#%%
def gradient_descent(X,y,w_in,b_in, cost_function,gradient_function,alpha,num_iters):
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_db,dj_dw = gradient_function(X,y,w,b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost = cost_function(X,y,w,b)
        print(cost)
    return w,b,cost
#%%

def EnterData(Close,High,low,open,volume):
    X_New_stock = np.array([Close,High,low,open,volume])
    X_norm = (X_New_stock -X_train.mean(axis=0))/X_train.std(axis=0)
    X_stock_predict = np.dot(X_norm,w_final)+b_final
    return X_stock_predict


#%%
Prediction = EnterData(327.51,335.50,323.64,335.00, 28698900)
print(Prediction)
