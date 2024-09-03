---
title: 'Analysis Walkthrough: Supervised Classification with Bank Churn Data'
date: 2024-09-03
permalink: /posts/2024/09/churn_sklearn/
tags:
  - machine learning
  - classification
  - statistics
---

This post provides a walkthrough demonstrating how to use the `sklearn` package in Python to tune and evaluate multiple supervised classification methods, such as logistic regression and extreme gradient boosting (XGBoost) to predict whether bank customers will close their account. The dataset comes from a past [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s4e1/data) and contains several variables, including credit score, gender, and age.

Code to produce this blog post can be found in [this](https://github.com/trgrimm/churn_analysis) GitHub repository.

------------------------------------------------------------------------

# Data description

Data for this analysis comes from a previous [Kaggle playground competition](https://www.kaggle.com/competitions/playground-series-s4e1/data) titled "Binary Classification with a Bank Churn Dataset".

Bank churn, which is also known as customer attrition, is when customers end their relationship with the bank (close their accounts). Predicting churn is essential to allow the bank to take action to retain customers. The cost of acquiring a new customer is almost always higher than retaining an existing customer[^1].


**Analysis Goal:** Predict the probability that a customer `Exited` (probability of churn).

Train and test datsets are provided by Kaggle, and we want to minimize the area under the curve (AUC) of the [receiver operating characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) (ROC) curve, which is also known as the "ROC AUC score".

For this analysis, I'll walk through the following steps:
1. Exploratory data analysis
2. Model building (tuning and evaluation)
3. Prediction on new data

# Exploratory data analysis (EDA)

For this analysis, we will be looking exclusively at the training dataset provided by Kaggle until we make our final predictions on the testing data. This mimics a real-life scenario where the future "new" data a model will be used on is not available until after model training, tuning, and selection.

First, we import necessary functions, load in the data, and evaluate the structure of the training dataset:


```python
# Import necessary libraries/functions

# Data wrangling and computation
import pandas as pd
import numpy as np
# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn import model_selection
import sklearn.metrics as metrics
# Plotting
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, facet_wrap, geom_histogram, labs
import seaborn as sns

```


```python
# Load in train and test sets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```


```python
# Look at the first few rows of the training data
train.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>15674932</td>
      <td>Okwudilichukwu</td>
      <td>668</td>
      <td>France</td>
      <td>Male</td>
      <td>33.0</td>
      <td>3</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>181449.97</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>15749177</td>
      <td>Okwudiliolisa</td>
      <td>627</td>
      <td>France</td>
      <td>Male</td>
      <td>33.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>49503.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>15694510</td>
      <td>Hsueh</td>
      <td>678</td>
      <td>France</td>
      <td>Male</td>
      <td>40.0</td>
      <td>10</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>184866.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>15741417</td>
      <td>Kao</td>
      <td>581</td>
      <td>France</td>
      <td>Male</td>
      <td>34.0</td>
      <td>2</td>
      <td>148882.54</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>84560.88</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>15766172</td>
      <td>Chiemenam</td>
      <td>716</td>
      <td>Spain</td>
      <td>Male</td>
      <td>33.0</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>15068.83</td>
      <td>0</td>
    </tr>
  </tbody>
</table>




```python
# Check column types, see if any rows have null values
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 165034 entries, 0 to 165033
    Data columns (total 14 columns):
     #   Column           Non-Null Count   Dtype  
    ---  ------           --------------   -----  
     0   id               165034 non-null  int64  
     1   CustomerId       165034 non-null  int64  
     2   Surname          165034 non-null  object 
     3   CreditScore      165034 non-null  int64  
     4   Geography        165034 non-null  object 
     5   Gender           165034 non-null  object 
     6   Age              165034 non-null  float64
     7   Tenure           165034 non-null  int64  
     8   Balance          165034 non-null  float64
     9   NumOfProducts    165034 non-null  int64  
     10  HasCrCard        165034 non-null  float64
     11  IsActiveMember   165034 non-null  float64
     12  EstimatedSalary  165034 non-null  float64
     13  Exited           165034 non-null  int64  
    dtypes: float64(5), int64(6), object(3)
    memory usage: 17.6+ MB



```python
# Check how many missing values are in the training data
train.apply(lambda x: x.isna().sum())
```




    id                 0
    CustomerId         0
    Surname            0
    CreditScore        0
    Geography          0
    Gender             0
    Age                0
    Tenure             0
    Balance            0
    NumOfProducts      0
    HasCrCard          0
    IsActiveMember     0
    EstimatedSalary    0
    Exited             0
    dtype: int64



There are no null or missing values. We also have the following predictor variables (features): 

**Categorical**
* `Geography`: customer's country of residence (France, Spain, or Germany)
* `Gender`: customer's gender (Male or Female)

**Quantitative**
* `CreditScore`: customer's credit score (numerical score)
* `Age`: customer's age (years)
* `Tenure`: number of years a customer has had an account with the bank
* `Balance`: customer's account balance
* `NumOfProducts`: number of bank products used by the customer (e.g., savings account, credit card)
* `EstimatedSalary`: customer's estimated salary


**Logical**
* `HasCrCard`: whether the customer has a credit card (1 = yes, 0 = no)
* `IsActiveMember`: whether the customer is an active member (1 = yes, 0 = no)

**Miscellaneous** (not useful)
* `id`: Row number
* `CustomerID`: unique identifier for each customer
* `Surname`: customer's last name

And the target (response) variable we want to predict: `Exited`.


```python
# Basic summary statistics for each column in train
train.drop(columns=['id', 'CustomerId', 'Surname']).describe()

```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CreditScore</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>165034.000000</td>
      <td>165034.000000</td>
      <td>165034.000000</td>
      <td>165034.000000</td>
      <td>165034.000000</td>
      <td>165034.000000</td>
      <td>165034.000000</td>
      <td>165034.000000</td>
      <td>165034.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>656.454373</td>
      <td>38.125888</td>
      <td>5.020353</td>
      <td>55478.086689</td>
      <td>1.554455</td>
      <td>0.753954</td>
      <td>0.497770</td>
      <td>112574.822734</td>
      <td>0.211599</td>
    </tr>
    <tr>
      <th>std</th>
      <td>80.103340</td>
      <td>8.867205</td>
      <td>2.806159</td>
      <td>62817.663278</td>
      <td>0.547154</td>
      <td>0.430707</td>
      <td>0.499997</td>
      <td>50292.865585</td>
      <td>0.408443</td>
    </tr>
    <tr>
      <th>min</th>
      <td>350.000000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.580000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>597.000000</td>
      <td>32.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>74637.570000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>659.000000</td>
      <td>37.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>117948.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>710.000000</td>
      <td>42.000000</td>
      <td>7.000000</td>
      <td>119939.517500</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>155152.467500</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>850.000000</td>
      <td>92.000000</td>
      <td>10.000000</td>
      <td>250898.090000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>199992.480000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>



From the summary statistics, we see that overall churn is about 21% (about 21% of people exited).

To get an idea of the differences in values of each variable for those who closed their accounts and for those who did not, we can group by `Exited` and compute the means and standard deviations across various categories:


```python
# Look at means and standard deviations of different variables for the people that closed their accounts and for those who did not
train.drop(columns = ['id', 'CustomerId', 'Surname', 'Geography', 'Gender']).groupby('Exited').agg(['mean', 'std']).round(2)
```

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">CreditScore</th>
      <th colspan="2" halign="left">Age</th>
      <th colspan="2" halign="left">Tenure</th>
      <th colspan="2" halign="left">Balance</th>
      <th colspan="2" halign="left">NumOfProducts</th>
      <th colspan="2" halign="left">HasCrCard</th>
      <th colspan="2" halign="left">IsActiveMember</th>
      <th colspan="2" halign="left">EstimatedSalary</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>Exited</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>657.59</td>
      <td>79.79</td>
      <td>36.56</td>
      <td>8.15</td>
      <td>5.05</td>
      <td>2.80</td>
      <td>51255.81</td>
      <td>62189.98</td>
      <td>1.62</td>
      <td>0.49</td>
      <td>0.76</td>
      <td>0.43</td>
      <td>0.55</td>
      <td>0.50</td>
      <td>112084.29</td>
      <td>50214.66</td>
    </tr>
    <tr>
      <th>1</th>
      <td>652.22</td>
      <td>81.14</td>
      <td>43.96</td>
      <td>9.00</td>
      <td>4.91</td>
      <td>2.83</td>
      <td>71209.98</td>
      <td>62646.69</td>
      <td>1.33</td>
      <td>0.66</td>
      <td>0.74</td>
      <td>0.44</td>
      <td>0.29</td>
      <td>0.46</td>
      <td>114402.50</td>
      <td>50542.03</td>
    </tr>
  </tbody>
</table>



In general, most of the variables have similar values between customers with `Exited` = 1 and customers with `Exited` = 0. However, the average age is slightly higher for those with `Exited` = 1, and `IsActiveMember` is higher for customers who did not close their accounts (55% for `Exited` = 0) than for those who did (29% for `Exited` = 1), indicating that people may be more likely to close their account if they are not an active member.

To further explore this data, let's create some histograms of each variable with a different color for the customers with `Exited` = 1 and `Exited` = 0.


```python
# Make a df for plotting, change "Exited" to a "category" type to assist with plotting
train_plot = train.astype({'Exited': 'category'})

# Plot quantitative variables
(
    # Reshape the data to facilitate plotting with ggplot
    pd.melt(train_plot[['CreditScore', 'Age' ,'Balance', 'EstimatedSalary', 'Exited']], id_vars = 'Exited')
    >> ggplot() +
        geom_histogram(aes('value', fill = 'Exited')) +
        facet_wrap('variable', scales = 'free') +
        labs(x = 'Category', y = 'Count', title = 'Histograms of Quantitative Variables')
)
```

<p align="center">
<img src="https://github.com/user-attachments/assets/10582f3d-dc7b-4eb5-8dc8-998221a7d0d8" width="600">
</p>

Let's make a plot of the correlation matrix for the 6 continuous quantitative predictor variables:


```python
# Plot categorical variables and quantitative variables with few categories (NumOfProducts, Tenure)
(
    pd.melt(train_plot[['Gender', 'Geography', 'HasCrCard', 'IsActiveMember', 'NumOfProducts', 'Exited', 'Tenure']], id_vars = 'Exited')
    >> ggplot() +
        geom_histogram(aes('value', fill = 'Exited'), binwidth = .5) +
        facet_wrap('variable', scales = 'free') +
        labs(x = 'Value', y = 'Count', title = 'Histograms of Categorical Variables', subtitle = '(and Quantitative Variables with Few Categories)')
)
```

<p align="center">
<img src="https://github.com/user-attachments/assets/b92d167b-2921-4d5c-a682-4f63495bc3e0" width="600">
</p>


In general, the shapes of the distributions of the variables is similar. However, there is a noticeable difference in `Age`, where the majority of customers with `Exited` = 0 tend to be younger, while the majority of customers with `Exited` = 1 tend to be older, which is similar to what was observed based on the summary statistics.

Now, let's look at correlations between the quantitative predictors to determine if any substantial multicollinearity is present:


```python
sns.heatmap(train.drop(columns = ['id', 'CustomerId', 'Surname', 'Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'Exited']).corr(), annot = True)
plt.title('Correlation Plot')
plt.show()
```


<p align="center">
<img src="https://github.com/user-attachments/assets/20429360-03a6-42b7-b422-173ed3fb7f2d" width="600">
</p>


Most of the predictors are uncorrelated with one another, but there appears to be a weak linear relationship between `Balance` and `NumOfProducts`.

# Feature engineering

A common step in many machine learning projects is to "engineer" new features. The goal is to create new predictor columns that will provide better model predictions of the target variable, `Exited`. Often, new features are created using domain expertise.

A naive approach to feature engineering is to simply augment the feature (predictor) matrix with polynomials of the quantitative variables. For simplicity, I'll only add the following:

* a squared `EstimatedSalary` column, denoted `EstimatedSalary2`
* a squared `CreditScore` column, denoted `CreditScore2`
* a squared `Balance` column, denoted `Balance2`


```python
train['EstimatedSalary2'] = train['EstimatedSalary']**2
train['CreditScore2'] = train['CreditScore']**2
train['Balance2'] = train['Balance']**2

# The new variables are now part of the training data
train.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
      <th>EstimatedSalary2</th>
      <th>CreditScore2</th>
      <th>Balance2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>15674932</td>
      <td>Okwudilichukwu</td>
      <td>668</td>
      <td>France</td>
      <td>Male</td>
      <td>33.0</td>
      <td>3</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>181449.97</td>
      <td>0</td>
      <td>3.292409e+10</td>
      <td>446224</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>15749177</td>
      <td>Okwudiliolisa</td>
      <td>627</td>
      <td>France</td>
      <td>Male</td>
      <td>33.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>49503.50</td>
      <td>0</td>
      <td>2.450597e+09</td>
      <td>393129</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>15694510</td>
      <td>Hsueh</td>
      <td>678</td>
      <td>France</td>
      <td>Male</td>
      <td>40.0</td>
      <td>10</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>184866.69</td>
      <td>0</td>
      <td>3.417569e+10</td>
      <td>459684</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>15741417</td>
      <td>Kao</td>
      <td>581</td>
      <td>France</td>
      <td>Male</td>
      <td>34.0</td>
      <td>2</td>
      <td>148882.54</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>84560.88</td>
      <td>0</td>
      <td>7.150542e+09</td>
      <td>337561</td>
      <td>2.216601e+10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>15766172</td>
      <td>Chiemenam</td>
      <td>716</td>
      <td>Spain</td>
      <td>Male</td>
      <td>33.0</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>15068.83</td>
      <td>0</td>
      <td>2.270696e+08</td>
      <td>512656</td>
      <td>0.000000e+00</td>
    </tr>
  </tbody>
</table>



# Model building, tuning, and selection


We will evaluate the performance of 6 different classification methods on predicting `Exited` in the training data:
* logistic regression (LR)
* k-nearest neighbors (KNN)
* support vector machine (SVM)
* random forest (RF)
* extreme gradient boosting (XGBoost)
* CatBoost (CB)

Below, each method will be tuned for optimal performance, and its performance will be evaluated in terms of ROC AUC. Then, the overall performance of each model will be compared, and the best model will be selected for final predictions on the testing data.

## Set up the training data

Perform the following:

Create the predictor `X_train` matrix:
* apply one hot encoding to the categorical variables
* for SVM, center and scale the numerical predictors

Create the target `y_train` vector.


```python
# Import additional functions for preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
```


```python
# Creating the X_train matrix:

# Remove columns that are not predictors
X_train = train.drop(columns=['id', 'CustomerId','Surname', 'Exited'])

# Separate out the categorical columns for one hot encoding
X_train_encoding = X_train[['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']]

# Create a matrix of the one hot encoded categorical columns
encoder = OneHotEncoder(sparse_output = False)
X_train_encoded = encoder.fit_transform(X_train_encoding)

# Check the names of the columns that were created via one hot encoding
encoder.get_feature_names_out()

# The encoded matrix is currently a numpy array. Change this to a data frame
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder.get_feature_names_out())

# Combine the numerical columns with the categorical columns to create a single X_train data frame
X_train = pd.concat([X_train.drop(columns = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']).reset_index(drop = True), X_train_encoded], axis = 1)

# Make sure everything looks correct
X_train.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CreditScore</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>EstimatedSalary</th>
      <th>EstimatedSalary2</th>
      <th>CreditScore2</th>
      <th>Balance2</th>
      <th>Geography_France</th>
      <th>Geography_Germany</th>
      <th>Geography_Spain</th>
      <th>Gender_Female</th>
      <th>Gender_Male</th>
      <th>HasCrCard_0.0</th>
      <th>HasCrCard_1.0</th>
      <th>IsActiveMember_0.0</th>
      <th>IsActiveMember_1.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>668</td>
      <td>33.0</td>
      <td>3</td>
      <td>0.00</td>
      <td>2</td>
      <td>181449.97</td>
      <td>3.292409e+10</td>
      <td>446224</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>627</td>
      <td>33.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>49503.50</td>
      <td>2.450597e+09</td>
      <td>393129</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>678</td>
      <td>40.0</td>
      <td>10</td>
      <td>0.00</td>
      <td>2</td>
      <td>184866.69</td>
      <td>3.417569e+10</td>
      <td>459684</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>581</td>
      <td>34.0</td>
      <td>2</td>
      <td>148882.54</td>
      <td>1</td>
      <td>84560.88</td>
      <td>7.150542e+09</td>
      <td>337561</td>
      <td>2.216601e+10</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>716</td>
      <td>33.0</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>15068.83</td>
      <td>2.270696e+08</td>
      <td>512656</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>




```python
y_train = train['Exited']
```

## Logistic Regression (LR)


```python
X_train.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CreditScore</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>EstimatedSalary</th>
      <th>EstimatedSalary2</th>
      <th>CreditScore2</th>
      <th>Balance2</th>
      <th>Geography_France</th>
      <th>Geography_Germany</th>
      <th>Geography_Spain</th>
      <th>Gender_Female</th>
      <th>Gender_Male</th>
      <th>HasCrCard_0.0</th>
      <th>HasCrCard_1.0</th>
      <th>IsActiveMember_0.0</th>
      <th>IsActiveMember_1.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>668</td>
      <td>33.0</td>
      <td>3</td>
      <td>0.00</td>
      <td>2</td>
      <td>181449.97</td>
      <td>3.292409e+10</td>
      <td>446224</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>627</td>
      <td>33.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>49503.50</td>
      <td>2.450597e+09</td>
      <td>393129</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>678</td>
      <td>40.0</td>
      <td>10</td>
      <td>0.00</td>
      <td>2</td>
      <td>184866.69</td>
      <td>3.417569e+10</td>
      <td>459684</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>581</td>
      <td>34.0</td>
      <td>2</td>
      <td>148882.54</td>
      <td>1</td>
      <td>84560.88</td>
      <td>7.150542e+09</td>
      <td>337561</td>
      <td>2.216601e+10</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>716</td>
      <td>33.0</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>15068.83</td>
      <td>2.270696e+08</td>
      <td>512656</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>




```python
# Initialize the logistic regression model (L1 regularization)
log_reg = LogisticRegression(solver = 'liblinear', penalty = 'l1')

# Set up 10-fold cross-validation so that we use the same 10 folds to evaluate all models.
kfold = model_selection.KFold(n_splits = 10, shuffle = True, random_state = 24)

# Perform 10-fold CV to evaluate logistic regression model performance
log_reg_auc = model_selection.cross_val_score(log_reg, X_train, y_train, cv = kfold, scoring = 'roc_auc')

# Logistic regression achieves a ROC AUC of about 81.8%
np.mean(log_reg_auc)
```

    np.float64(0.8177866487039773)



## K-Nearest Neighbors (KNN)


```python
knn = KNeighborsClassifier()
# Set up a grid of hyperparameters to tune
knn_parameters = {'n_neighbors': [500, 1000, 2500, 5000],
                  'weights': ['uniform', 'distance']}

# Perform 10-fold cross-validation to obtain ROC AUC scores for each combination of hyperparameters
knn_cv = model_selection.GridSearchCV(knn, knn_parameters, scoring = 'roc_auc', cv = kfold, n_jobs = 4) # use 4 cores in parallel to expedite tuning
knn_cv.fit(X_train, y_train)

```


```python
# Make a dataframe to display the roc auc score for each combination of hyperparameters (sorted from best to worst)
pd.concat([pd.DataFrame(knn_cv.cv_results_['params']),
           pd.DataFrame(knn_cv.cv_results_['mean_test_score'],
                        columns = ['roc auc'])], axis = 1).sort_values('roc auc', ascending = False).head(5)

# The best KNN model uses n_neighbors = 500, weights = uniform, and achieves a ROC AUC score of 60.1%
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_neighbors</th>
      <th>weights</th>
      <th>roc auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>500</td>
      <td>uniform</td>
      <td>0.601276</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>uniform</td>
      <td>0.600648</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2500</td>
      <td>uniform</td>
      <td>0.599931</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5000</td>
      <td>uniform</td>
      <td>0.598491</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5000</td>
      <td>distance</td>
      <td>0.572966</td>
    </tr>
  </tbody>
</table>



## Support Vector Machine Classifier (SVM)


```python
svc = SVC()
svc_parameters = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10]},
    {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
]

svc_cv = model_selection.GridSearchCV(svc, svc_parameters, scoring = 'roc_auc', cv = kfold, n_jobs = 4)

# SVC is sensitive to the scale of the data, so let's first scale our continuous predictors
scaler = StandardScaler()

# Select the continuous features
X_train_continuous = X_train[['CreditScore', 'Age', 'Tenure', 'Balance',
                              'NumOfProducts', 'EstimatedSalary', 'EstimatedSalary2', 'CreditScore2', 'Balance2']]

# Apply scaling, turn into a dataframe
X_train_continuous_scaled = pd.DataFrame(scaler.fit_transform(X_train_continuous),
                                         columns = ['CreditScore', 'Age', 'Tenure', 'Balance',
                                                    'NumOfProducts', 'EstimatedSalary', 'EstimatedSalary2', 'CreditScore2', 'Balance2'])

# Combine all columns together to create the final training data
X_train_svc = pd.concat([X_train_continuous_scaled,
                         X_train.drop(columns = ['CreditScore', 'Age', 'Tenure', 'Balance',
                                                 'NumOfProducts', 'EstimatedSalary', 'EstimatedSalary2', 'CreditScore2', 'Balance2'])], axis = 1)

# Fit the SVC
svc_cv.fit(X_train_svc, y_train)
```



```python
pd.concat([pd.DataFrame(svc_cv.cv_results_['params']),
           pd.DataFrame(svc_cv.cv_results_['mean_test_score'],
                        columns = ['roc auc'])], axis = 1).sort_values('roc auc', ascending = False).head(5)

# The best SVM model uses C = 0.1, kernel = rbf, gamma = 0.1
# and achieves a ROC AUC score of 82.5%
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C</th>
      <th>kernel</th>
      <th>gamma</th>
      <th>roc auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0.1</td>
      <td>rbf</td>
      <td>0.1</td>
      <td>0.825011</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.1</td>
      <td>rbf</td>
      <td>1.0</td>
      <td>0.822602</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.1</td>
      <td>linear</td>
      <td>NaN</td>
      <td>0.814939</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>linear</td>
      <td>NaN</td>
      <td>0.814919</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>linear</td>
      <td>NaN</td>
      <td>0.814917</td>
    </tr>
  </tbody>
</table>



## Random Forest (RF)


```python
rf = RandomForestClassifier()
rf_parameters = {'n_estimators': [500, 1000], 'max_features': [2, 4, 6], 'max_depth': [6, 9, 12]}

rf_cv = model_selection.GridSearchCV(rf, rf_parameters, scoring = 'roc_auc', cv = kfold, n_jobs = 4)

rf_cv.fit(X_train, y_train)
```



```python
pd.concat([pd.DataFrame(rf_cv.cv_results_['params']),
           pd.DataFrame(rf_cv.cv_results_['mean_test_score'],
                        columns = ['roc auc'])], axis = 1).sort_values('roc auc', ascending = False).head(5)

# The best RF model uses max_depth = 12, max_features = 6, and n_estimators = 1000
# and achieves a ROC AUC score of 88.8%
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max_depth</th>
      <th>max_features</th>
      <th>n_estimators</th>
      <th>roc auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>12</td>
      <td>6</td>
      <td>1000</td>
      <td>0.887904</td>
    </tr>
    <tr>
      <th>16</th>
      <td>12</td>
      <td>6</td>
      <td>500</td>
      <td>0.887888</td>
    </tr>
    <tr>
      <th>11</th>
      <td>9</td>
      <td>6</td>
      <td>1000</td>
      <td>0.887874</td>
    </tr>
    <tr>
      <th>10</th>
      <td>9</td>
      <td>6</td>
      <td>500</td>
      <td>0.887870</td>
    </tr>
    <tr>
      <th>15</th>
      <td>12</td>
      <td>4</td>
      <td>1000</td>
      <td>0.887769</td>
    </tr>
  </tbody>
</table>



## Extreme Gradient Boosting (XGBoost)


```python
xgb_clf = xgb.XGBClassifier()
xgb_parameters = {'learning_rate': [0.05, 0.1, 0.15, 0.3], 'max_depth': [5, 6, 7], 'colsample_bytree': [0.25, 0.5, 1]}

xgb_cv = model_selection.GridSearchCV(xgb_clf, xgb_parameters, scoring = 'roc_auc', cv = kfold, n_jobs = 4)

xgb_cv.fit(X_train, y_train)
```


```python
pd.concat([pd.DataFrame(xgb_cv.cv_results_['params']),
           pd.DataFrame(xgb_cv.cv_results_['mean_test_score'],
                        columns = ['roc auc'])], axis = 1).sort_values('roc auc', ascending = False).head(5)

# The best XGBoost model uses colsample_bytree = 0.5, learning_rate = 0.15, and max_depth = 5
# and achieves a ROC AUC score of about 89.0%
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>colsample_bytree</th>
      <th>learning_rate</th>
      <th>max_depth</th>
      <th>roc auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>0.5</td>
      <td>0.15</td>
      <td>5</td>
      <td>0.890029</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.5</td>
      <td>0.10</td>
      <td>5</td>
      <td>0.889848</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.5</td>
      <td>0.10</td>
      <td>6</td>
      <td>0.889840</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1.0</td>
      <td>0.15</td>
      <td>5</td>
      <td>0.889653</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1.0</td>
      <td>0.10</td>
      <td>5</td>
      <td>0.889615</td>
    </tr>
  </tbody>
</table>



# CatBoost (CB)


```python
cbc_clf = CatBoostClassifier(cat_features = [1, 2, 6, 7, 8], od_type = "Iter", od_wait = 20)
cbc_parameters = {'learning_rate': [0.1, 0.2, 0.3], 'depth': [4, 6, 8], 'iterations': [50, 100, 150]}

cbc_cv = model_selection.GridSearchCV(cbc_clf, cbc_parameters, scoring = 'roc_auc', cv = kfold, n_jobs = 4)

# CatBoost does not require one hot encoding of categorical variables, so we'll use the original training dataset here,
# changing some columns we'll treat as categories to "string" type to be processed properly by the model
catboost_train = train.drop(columns = ['id', 'CustomerId', 'Surname', 'Exited'])
catboost_train = catboost_train.astype({'NumOfProducts': 'string', 'HasCrCard': 'string', 'IsActiveMember': 'string'})


cbc_cv.fit(catboost_train, y_train)
```


```python
pd.concat([pd.DataFrame(cbc_cv.cv_results_['params']),
           pd.DataFrame(cbc_cv.cv_results_['mean_test_score'],
                        columns = ['roc auc'])], axis = 1).sort_values('roc auc', ascending = False).head(5)

# The best catboost model uses depth = 6, iterations = 150, and learning_rate = 0.2,
# and achieves a ROC AUC score of just under 89.0%
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>depth</th>
      <th>iterations</th>
      <th>learning_rate</th>
      <th>roc auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>6</td>
      <td>150</td>
      <td>0.2</td>
      <td>0.889753</td>
    </tr>
    <tr>
      <th>13</th>
      <td>6</td>
      <td>100</td>
      <td>0.2</td>
      <td>0.889599</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4</td>
      <td>150</td>
      <td>0.2</td>
      <td>0.889540</td>
    </tr>
    <tr>
      <th>24</th>
      <td>8</td>
      <td>150</td>
      <td>0.1</td>
      <td>0.889519</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4</td>
      <td>150</td>
      <td>0.3</td>
      <td>0.889510</td>
    </tr>
  </tbody>
</table>



## Model selection

The best performance of each method is given in the following table:

| Method    | ROC AUC |
| -------- | ------- |
| LR  | 81.8%   |
| KNN | 60.1%   |
| SVM    |   82.2%  |
| RF    |  88.8%   |
| XGBoost    | 89.0%    |
| CB   | 89.0%    |

The best methods, by far, are RF, XGBoost, and CB, which had similar performance. However, I will select XGBoost as the final model because it achieved a slightly higher CV ROC AUC score than RF and CB.

One nice feature of XGBoost is the ability to easily visualize the importance of each variable for predicting `Exited`. A feature important plot for the final XGBoost model is shown below. Note that

* The most important variables are `CreditScore`, `Age`, `EstimatedSalary`, and `Balance`.
* Feature importance decreases greatly after `Balance`.
* Being an active member (`IsActiveMember` = 1 or 0) is not as important as previously thought based on initial EDA.


```python
from xgboost import plot_importance
plot_importance(xgb_cv.best_estimator_)
plt.show()
```

<p align="center">
<img src="https://github.com/user-attachments/assets/809781db-c0a1-4bed-bd3f-e3b20a1759ba" width="600">
</p>

# Prediction for the testing data

Now that we've selected our best model based on cross-validation on the training data, we can use that model to obtain predictions for the testing set.


```python
# Add engineered features to testing data
test['EstimatedSalary2'] = test['EstimatedSalary']**2
test['CreditScore2'] = test['CreditScore']**2
test['Balance2'] = test['Balance']**2

# Next, we need to one hot encode the categorical variables in the testing data
X_test = test.drop(columns=['id', 'CustomerId','Surname'])

# Separate out the categorical columns for one hot encoding, apply one hot encoding
X_test_encoding = X_test[['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']]
X_test_encoded = encoder.transform(X_test_encoding)

# The encoded matrix is currently a numpy array. Change this to a data frame
X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder.get_feature_names_out())

# Combine the numerical columns with the categorical columns to create a single X_test data frame
X_test = pd.concat([X_test.drop(columns = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']).reset_index(drop = True), X_test_encoded], axis = 1)

# Make sure everything looks correct
X_test.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CreditScore</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>EstimatedSalary</th>
      <th>EstimatedSalary2</th>
      <th>CreditScore2</th>
      <th>Balance2</th>
      <th>Geography_France</th>
      <th>Geography_Germany</th>
      <th>Geography_Spain</th>
      <th>Gender_Female</th>
      <th>Gender_Male</th>
      <th>HasCrCard_0.0</th>
      <th>HasCrCard_1.0</th>
      <th>IsActiveMember_0.0</th>
      <th>IsActiveMember_1.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>586</td>
      <td>23.0</td>
      <td>2</td>
      <td>0.00</td>
      <td>2</td>
      <td>160976.75</td>
      <td>2.591351e+10</td>
      <td>343396</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>683</td>
      <td>46.0</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>72549.27</td>
      <td>5.263397e+09</td>
      <td>466489</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>656</td>
      <td>34.0</td>
      <td>7</td>
      <td>0.00</td>
      <td>2</td>
      <td>138882.09</td>
      <td>1.928823e+10</td>
      <td>430336</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>681</td>
      <td>36.0</td>
      <td>8</td>
      <td>0.00</td>
      <td>1</td>
      <td>113931.57</td>
      <td>1.298040e+10</td>
      <td>463761</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>752</td>
      <td>38.0</td>
      <td>10</td>
      <td>121263.62</td>
      <td>1</td>
      <td>139431.00</td>
      <td>1.944100e+10</td>
      <td>565504</td>
      <td>1.470487e+10</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>




```python
# Obtain predictions
final_predictions = pd.DataFrame(dict(id = test['id'],
                  Exited = xgb_cv.best_estimator_.predict_proba(X_test)[:, 1]))

# Look at the first few predictions
final_predictions.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>165034</td>
      <td>0.023930</td>
    </tr>
    <tr>
      <th>1</th>
      <td>165035</td>
      <td>0.835616</td>
    </tr>
    <tr>
      <th>2</th>
      <td>165036</td>
      <td>0.028150</td>
    </tr>
    <tr>
      <th>3</th>
      <td>165037</td>
      <td>0.232614</td>
    </tr>
    <tr>
      <th>4</th>
      <td>165038</td>
      <td>0.338327</td>
    </tr>
  </tbody>
</table>



We now have our final predictions stored in the `final_predictions` dataframe. Now, we can save those results to a .csv file and submit them to Kaggle to obtain our final score.


```python
# Save predictions to .csv
final_predictions.to_csv('churn_predictions.csv', index = False)
```

## Final Kaggle competition results

After submitting the predictions to Kaggle, a ROC AUC score on the testing data is returned.

According to Kaggle, the final ROC AUC score on the testing data of our model is 0.88864, or about 88.9%, which would have placed us in the top 38.7% of submissions during the competition (1406 place out of 3633 teams).

There are many additional things we could have done to improve model performance, such as:
* performing extensive feature engineering using domain expertise
* increasing grid search ranges for tuning model parameters
* making predictions with an ensemble model that combines output from our top models to produce a single prediction
* considering more complex models, such as neural networks (NN's)


[^1]: https://www.forbes.com/councils/forbesbusinesscouncil/2022/12/12/customer-retention-versus-customer-acquisition/
