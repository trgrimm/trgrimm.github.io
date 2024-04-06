---
title: 'Analysis Walkthrough: Supervised Regression with Abalone Data'
date: 2024-04-05
permalink: /posts/2024/04/abalone_tidymodels/
tags:
  - machine learning
  - regression
  - statistics
---

This post provides a complete walkthrough of a simple analysis of [Abalone](https://en.wikipedia.org/wiki/Abalone) data from [Kaggle](https://www.kaggle.com/) in `R` using the `tidymodels` package. Various supervised statistical/machine learning regression methods are compared.

Code to produce this blog post can be found in [this](https://github.com/trgrimm/t2_mewma) GitHub repository [update repo link]}.


## Data description

Data for this analysis comes from a [Kaggle playground prediction competition](https://www.kaggle.com/competitions/playground-series-s4e4/overview) titled ``Regression with an Abalone Dataset". This Kaggle data is synthetically generated from a real dataset of various physical measurements contained [here](https://archive.ics.uci.edu/dataset/1/abalone) on the UC Irvine Machine Learning Repository.

Abalones are a group of marine gastropod mollusks found in various cold waters across the world. Typically, the age of an abalone is determined by cutting through its shell and counting the number of rings in a microsope. This process can be time-consuming. 

The abalone dataset contains 

* 8 predictor variables: sex, length, diameter, height, etc.
* 1 numeric response variable: `Rings`

**Analysis Goal:** Predict the number of `Rings` using the easily obtained physical measurements (predictor variables).

Train and test datasets are provided by Kaggle, and we want to minimize the root mean squared logarithmic error (RMSLE), which is defined as

$$
\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^n \left(\log(1 + \hat{y}_i) - \log(1 + y_i)\right)^2},
$$

where

* $n$ = number of observations in the test set
* $\hat{y}_i$ is the predicted value of `Rings` for observation $i$
* $y_i$ is the observed value of `Rings` for observation $i$
* $\log$ is the natural logarithm.
