import copy 
import pandas as pd 
import numpy as np 
 
import pandas as pd 
 
# Load the Tesla historical stock data
df = pd.read_csv("HistoricalData_1726367135218.csv") 
 
# Columns containing stock price information
price_columns = ["Close/Last", "High", "Low", "Open"] 
 
# Remove the '$' symbol and convert price values from strings to floats
for columns in price_columns: 
    df[columns] = df[columns].str.replace('$','',regex=False).astype(float)

# Convert trading volume to float
df["Volume"] = df["Volume"].astype(float) 
 
# X_train contains today's stock information
# We use all rows except the last because we need the next day's price
X_train = df[["Close/Last", "High", "Low", "Open","Volume"]].iloc[:-1]

# Y_train contains the next day's closing price
Y_train = df["Close/Last"].shift(-1).iloc[:-1] 
 
# Convert pandas data into NumPy arrays
X_train = X_train.to_numpy() 
Y_train = Y_train.to_numpy() 
 
# Calculate the mean and standard deviation of each feature
# These values are used to normalize the input data
mean = X_train.mean(axis=0) 
std = X_train.std(axis=0) 
 
# Normalize the training data
X_scaled = (X_train - mean) / std

# Check the mean and standard deviation of the scaled data
# print(X_scaled.mean(axis=0)) 
# print(X_scaled.std(axis=0)) 

# Calculate the cost (Mean Squared Error)
def compute_cost(X_train, Y_train,w,b): 
    m = X_train.shape[0] 
    cost = 0 

    # Calculate the prediction for every training example
    for i in range(m): 
        f_wb_i = np.dot(X_train[i],w)+b 

        # Add the squared prediction error to the total cost
        cost+=(f_wb_i-Y_train[i])**2 

    # Divide by 2m as used in the linear regression cost function
    cost = cost / (2*m) 
    return cost 


# Calculate the gradient for w and b
def compute_gradient(X,y,w,b): 
    m,n = X.shape 

    # Create an array to store the gradient for each weight
    dj_dw = np.zeros((n,)) 

    # Initialize the gradient for b
    dj_db = 0. 

    # Go through every training example
    for i in range(m): 

        # Calculate the prediction error
        err = (np.dot(X[i],w)+b) - y[i] 

        # Calculate the gradient for each feature
        for j in range(n): 
            dj_dw[j] = dj_dw[j]+ err*X[i,j] 

        # Calculate the gradient for b
        dj_db = dj_db + err 

    # Average the gradients over all training examples
    dj_db = dj_db/ m 
    dj_dw = dj_dw/ m 

    return dj_db, dj_dw 


# Gradient descent algorithm
def gradient_descent(X,y,w_in,b_in, cost_function,gradient_function,alpha,num_iters): 

    # Make a copy of the initial weights
    w = copy.deepcopy(w_in) 
    b = b_in 

    # Repeat gradient descent for the specified number of iterations
    for i in range(num_iters): 

        # Calculate the gradients
        dj_db,dj_dw = gradient_function(X,y,w,b) 

        # Update the weights and bias
        w = w - alpha * dj_dw 
        b = b - alpha * dj_db 

        # Calculate the cost after the update
        cost = cost_function(X,y,w,b) 

        # Display the cost for each iteration
        print(cost) 

    return w,b,cost 


# Initialize weights and bias
init_w = np.zeros(5,) 
init_b = 0 

# Number of gradient descent iterations
iterations = 1000 

# Learning rate
alpha = 0.3 

# Train the model using gradient descent
w_final,b_final,cost = gradient_descent(
    X_scaled,
    Y_train,
    init_w,
    init_b,
    compute_cost,
    compute_gradient,
    alpha,
    iterations
) 

# Display the final bias and weights found by gradient descent
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ") 

# Number of training examples
m = X_train.shape[0] 

# Display the prediction and actual target value for every training example
for i in range(m): 
    print(
        f"prediction:{np.dot(X_scaled[i],w_final)+b_final:0.2f}, "
        f"target value:{Y_train[i]}"
    ) 

# Display the final cost of the model
print(f"cost: {cost}") 


# Function used to make a prediction for new stock data
def EnterData(Close,High,low,open,volume): 

    # Create a NumPy array containing the new stock information
    X_New_stock = np.array([Close,High,low,open,volume]) 

    # Normalize the new data using the training data's mean and standard deviation
    X_norm = (X_New_stock - X_train.mean(axis=0))/X_train.std(axis=0) 

    # Calculate the predicted next-day closing price
    X_stock_predict = np.dot(X_norm,w_final)+b_final 

    return X_stock_predict 


# Enter sample Tesla stock information
Prediction = EnterData(
    327.51,     # Close price
    335.50,     # High price
    323.64,     # Low price
    335.00,     # Open price
    28698900    # Trading volume
)

# Display the predicted next-day closing price
print(Prediction)
