#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 22:22:04 2023

@author: meghanapuli
"""
import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

m,n = X_train.shape

# list to store the cost 
cost_history = []
num_iterations = []

def compute_cost(X, y, w, b): 
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b         
        cost = cost + (f_wb_i - y[i])**2       
    cost = cost / (2 * m)                         
    return cost

def compute_gradient(X, y, w, b): 
    m,n = X.shape           
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    
    w = w_in  
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        if i < 1000:
            if i < 100:
                num_iterations.append(i)
            # Compute the cost
            cost = compute_cost(X, y, w, b)
            cost_history.append(cost)
        
        if i % 100 == 0 or i == num_iters-1:
            print(f"Iteration {i}: ",f"w: {w}, b:{b}, cost: {cost}")
            
    # Plotting the cost against the first 100 iterations
    print("\nCost vs Iterations (Learning Curve)")
    plt.plot(num_iterations, cost_history[:100], marker='o')
    plt.title('Cost vs. Iterations')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost')
    plt.yticks(np.arange(0, 2600, 250))
    plt.show()  
    
    return w,b

# initialize parameters
initial_w =  np.zeros((n,))
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"\nFinal b,w found by gradient descent: {b_final:0.2f},{w_final}\n")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value/ground_truth: {y_train[i]}")

# test the model
X_test = []
X_test.append(float((input)("\nEnter the size of house(in sqft): ")))
X_test.append(float((input)("\nEnter the no. of bedrooms: ")))
X_test.append(float((input)("\nEnter the no. of floors:  ")))
X_test.append(float((input)("\nEnter the age of the house: ")))

esltimated_price = np.dot(X_test, w_final) + b_final
print(f"Estimated price: ${round(esltimated_price)*1000}")