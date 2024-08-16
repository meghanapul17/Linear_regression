#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 12:31:17 2023

@author: meghanapuli
"""
import numpy as np
import matplotlib.pyplot as plt

# Load our data set
X_train = np.array([1,2,3,4,5])
y_train = np.array([300,500,700,900,1100]) 

m = len(X_train)

# print("Training set:\n")
# for i in range(m):
#     print(f"X_{i}: {X_train[i]}")
#     print(f"y_{i}: {y_train[i]}")
#     print()
    
plt.scatter(X_train, y_train, marker='x', c='r')
plt.xticks(np.arange(0, 6, 0.5))
plt.yticks(np.arange(0, 1201, 100))

# add labels and title
plt.xlabel('size in 1000 sqft')
plt.ylabel("price in $1000's")
plt.title('Training set(Housing data)')

# display the plot
plt.show()

# list to store the cost 
cost_history = []
num_iterations = []

# compute the prediction of the model
def compute_model_output(X, w, b): 
    f_wb = w * X + b   
    return f_wb

# function to calculate the cost
def compute_cost(X, y, w, b):
    cost = 0  
    for i in range(m):
        #tmp_f_wb = w * X[i] + b
        tmp_f_wb = compute_model_output(X[i], w, b)
        cost += (tmp_f_wb - y[i]) ** 2
        total_cost = 1 / (2 * m) * cost
    return total_cost

# compute gradients
def compute_gradient(X, y, w, b):
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = compute_model_output(X[i], w, b)
        #f_wb = w * X[i] + b 
        dj_dw_i = (f_wb - y[i]) * X[i] 
        dj_db_i = f_wb - y[i] 
        dj_dw += dj_dw_i 
        dj_db += dj_db_i
        
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db
    
# compute the weights
def gradient_descent(X, y, w_in, b_in, alpha, num_iters, gradient_function):
    w = w_in
    b = b_in
    print("\nComputed weights over iterations:")
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w , b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        if i < 100:
            num_iterations.append(i)
            # Compute the cost
            cost = compute_cost(X, y, w, b)
            cost_history.append(cost)
        
        if i % 1000 == 0 or i == num_iters-1:
            print(f"Iteration {i}: ",f"w: {w: 0.3e}, b:{b: 0.5e}","Cost: ",compute_cost(X, y, w, b))
            
    # Plotting the cost against the first 100 iterations
    print("\nCost vs Iterations (Learning Curve)")
    plt.plot(num_iterations, cost_history, marker='o')
    plt.title('Cost vs. Iterations')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost')
    plt.show()
    
    return w,b

# initialize parameters
w_init = 0
b_init = 0

# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2

# run gradient descent
w_final, b_final = gradient_descent(X_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_gradient)

tmp_f_wb = compute_model_output(X_train, w_final, b_final)

# Plot our model prediction
plt.plot(X_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(X_train, y_train, marker='x', c='r',label='Actual Values')

# add labels and title
plt.title("Housing Prices")
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (in 1000s of dollars)')

plt.legend()
plt.xticks(np.arange(0, 6, 0.5))
plt.yticks(np.arange(0, 1201, 100))

# display the plot
print("\nOur model fit")
plt.show()

# compute the cost of the model
cost_of_model = compute_cost(X_train, y_train, w_final, b_final)
print(f"\n(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})") 
print(f"Cost of our model: {cost_of_model}")  

# test the model 
size = float((input)("\nEnter the size of house(in sqft): "))
size = float((size/1000))
esltimated_price = compute_model_output(size, w_final, b_final)
#esltimated_price = w_final * size + b_final
print(f"Estimated price: ${round(esltimated_price)*1000}")

