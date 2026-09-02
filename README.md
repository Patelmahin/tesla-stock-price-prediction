# 📈 Tesla Stock Price Prediction

A machine learning project that uses **multivariable linear regression from scratch** with Python and NumPy to predict the **next day's Tesla (TSLA) closing price** using historical stock-market data.

## 🧠 Overview

This project implements a linear regression model without using a machine-learning library such as Scikit-learn.

The model uses five features from Tesla's historical stock data:

* **Close/Last**
* **High**
* **Low**
* **Open**
* **Volume**

The model then predicts the **following day's closing price**.

## ⚙️ Machine Learning Approach

The project follows these steps:

1. Load Tesla historical stock data using Pandas
2. Clean and convert stock-price data into numerical values
3. Select input features and the next day's closing price as the target
4. Normalize the input features
5. Initialize model parameters
6. Calculate the cost using Mean Squared Error
7. Calculate gradients
8. Optimize the parameters using gradient descent
9. Use the trained model to predict a future closing price

## 📐 Model

The prediction is calculated using multivariable linear regression:

$$
f_{w,b}(x) = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

The model minimizes the cost function:

$$
J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)})-y^{(i)})^2
$$

The parameters are optimized using **gradient descent**.

## 🛠️ Technologies

* Python
* NumPy
* Pandas
* Linear Regression
* Gradient Descent

## 📊 Dataset

The project uses historical **Tesla (TSLA)** stock-market data containing daily price and trading-volume information.

## 🎯 What I Learned

Through this project, I practiced:

* Implementing linear regression from scratch
* Implementing a cost function
* Computing gradients
* Understanding gradient descent
* Feature normalization
* Working with real-world datasets
* Using NumPy for machine-learning calculations
* Using Pandas for data preprocessing

## ⚠️ Disclaimer

This project is an educational implementation of machine-learning concepts. It is **not intended to provide reliable financial predictions or investment advice**.

