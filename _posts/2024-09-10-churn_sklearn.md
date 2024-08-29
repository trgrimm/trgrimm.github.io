---
title: 'Analysis Walkthrough: Supervised Classification with Bank Churn Data'
date: 2024-09-10
permalink: /posts/2024/09/churn_sklearn/
tags:
  - machine learning
  - classification
  - statistics
---

This post provides a walkthrough demonstrating how to use the `sklearn` package in Python to tune and evaluate multiple supervised (machine learning) classification methods to predict whether bank customers will close their account. The dataset comes from a past [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s4e1/data) and contains several variables, including credit score, gender, and age.

Code to produce this blog post can be found in [this] GitHub repository.

<!-- Code to produce this blog post can be found in [this](https://github.com/trgrimm/abalone_analysis) GitHub repository}. -->

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











[^1]: https://www.forbes.com/councils/forbesbusinesscouncil/2022/12/12/customer-retention-versus-customer-acquisition/
