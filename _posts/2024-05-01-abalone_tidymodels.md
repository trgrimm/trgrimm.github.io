---
title: 'Analysis Walkthrough: Supervised Regression with Abalone Data'
date: 2024-05-1
permalink: /posts/2024/05/abalone_tidymodels/
tags:
  - machine learning
  - regression
  - statistics
---

This post provides a complete walkthrough of analyzing [Abalone](https://en.wikipedia.org/wiki/Abalone) data from [Kaggle](https://www.kaggle.com/) and applying supervised machine learning (ML) regression methods in `R` using the `tidymodels` package. The best model is selected from a suite of candidate models, including random forests and extreme gradient boosting (XGBoost).

<!-- Code to produce this blog post can be found in [this](https://github.com/trgrimm/abalone_analysis) GitHub repository}. -->

# Data description

Data for this analysis comes from a [Kaggle playground prediction competition](https://www.kaggle.com/competitions/playground-series-s4e4/overview) titled "Regression with an Abalone Dataset". This Kaggle data is synthetically generated from a real dataset of various physical measurements contained [here](https://archive.ics.uci.edu/dataset/1/abalone) on the UC Irvine Machine Learning Repository.

Abalones are a group of marine gastropod mollusks found in various cold waters across the world. Typically, the age of an abalone is determined by cutting through its shell and counting the number of rings in a microsope. This process can be time-consuming. So, we want to use data-driven ML methods to predict the number of rings using other physical measurements that are more easily obtained.

Here's a picture of abalone for reference:

<p align="center">
    <img src="https://github.com/trgrimm/trgrimm.github.io/assets/70607091/24ec8cc6-c185-4136-bf12-6f5ce6a2a8ec" width="400">
</p>
<p align="center" class="caption">
Abalone
<a href="https://asc-aqua.org/learn-about-seafood-farming/farmed-abalone/">image source</a>.
</p>

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

For this analysis, we want to be able to visualize our final results to see how well we do on testing data. However, Kaggle does not release the true values of the response variable of the test set, even after the competition has ended.

First, I’ll set up a new train/test split using the `train` data provided by Kaggle. We’ll use this as our train/test data throughout the analysis below. After we’ve looked at those results, I’ll use the full original `train` set to obtain predictions for Kaggle’s `test` set so that we can enter those results into the competition.

# Load in and set up data

First, we load in the `tidyverse` and `tidymodels` packages, which will
be used throughout this analysis. Then, we load in the train and test
sets. These are stored in separate .csv files that I downloaded from the Kaggle competition.

``` r
library(tidyverse)
library(tidymodels)
theme_set(theme_light())
kaggle_train <- read_csv('abalone_data/train.csv', col_types = 'ifdddddddi')
kaggle_test <-read_csv('abalone_data/test.csv', col_types = 'ifddddddd')
```

First, let’s take a look at the original `kaggle_train` training set
provided by Kaggle. This will give us an idea of how the data is
structured.

``` r
glimpse(kaggle_train)
```

    Rows: 90,615
    Columns: 10
    $ id               <int> 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,…
    $ Sex              <fct> F, F, I, M, I, F, M, F, I, I, M, M, F, I, I, F, I, M,…
    $ Length           <dbl> 0.550, 0.630, 0.160, 0.595, 0.555, 0.610, 0.415, 0.61…
    $ Diameter         <dbl> 0.430, 0.490, 0.110, 0.475, 0.425, 0.480, 0.325, 0.49…
    $ Height           <dbl> 0.150, 0.145, 0.025, 0.150, 0.130, 0.170, 0.110, 0.15…
    $ `Whole weight`   <dbl> 0.7715, 1.1300, 0.0210, 0.9145, 0.7820, 1.2010, 0.331…
    $ `Whole weight.1` <dbl> 0.3285, 0.4580, 0.0055, 0.3755, 0.3695, 0.5335, 0.165…
    $ `Whole weight.2` <dbl> 0.1465, 0.2765, 0.0030, 0.2055, 0.1600, 0.3135, 0.071…
    $ `Shell weight`   <dbl> 0.2400, 0.3200, 0.0050, 0.2500, 0.1975, 0.3085, 0.130…
    $ Rings            <int> 11, 11, 6, 10, 9, 10, 9, 9, 4, 8, 8, 11, 15, 7, 10, 1…

We see that there are 90,615 rows and 10 total columns, one of which is
an ID column. We have 1 categorical variable, `Sex`, 7 numeric predictor
variables, and the numeric response variable of interest, `Rings`.

As discussed above, the original testing set provided by Kaggle does not
have the true values of `Rings`, meaning we are unable to evaluate
performance on that dataset here. So, we’ll set up our own train/test
splits from the original training set.

Let’s split the original training dataset from Kaggle into new `train`
and `test` sets using a random 80%/20% train/test split.

``` r
# Set a seed to ensure the same train/test split for reproducibility
set.seed(1234)
rand_split <- initial_split(kaggle_train,
                            prop = 0.80) #80/20 train/test split
train <- training(rand_split)
test <- testing(rand_split)
```

# Exploratory data analysis

Now that we’ve loaded in the data, let’s take a closer look at the
`train` data.

First, we’ll look at the variable we’re trying to predict: `Rings`. From
the output and plot below, we see that there are 28 different values of
`Rings` in the train set, ranging from 1 to 29. We also see that most
values are between 8 and 10.

``` r
length(unique(train$Rings))
```

    [1] 28

``` r
range(train$Rings)
```

    [1]  1 29

``` r
train |> 
  ggplot(aes(Rings)) +
  geom_histogram(bins = 28) +
  labs(y = 'Count', title = 'Histogram of Rings')
```

<p align="center">
    <img src="https://github.com/trgrimm/trgrimm.github.io/assets/70607091/1ed5c3f5-31a2-4917-8ebf-eba4196b3740" width="500">
</p>

Now, let’s split the histogram of `Rings` by `Sex`.

``` r
train |> 
  mutate(Sex = factor(Sex,
                      labels = c("Female", "Infant", "Male"))) |> 
  ggplot(aes(Rings)) +
  geom_histogram() +!

  facet_wrap(~Sex) +
  ggtitle("Histogram of Rings by Sex")
```

    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

<p align="center">
<img src="https://github.com/trgrimm/trgrimm.github.io/assets/70607091/e37fa9de-9f5c-4b9a-a3ab-288ff4447f1a" width="500">
</p>

From the histograms above, the distributions appear to have similar
shapes for each `Sex`, but the infants are centered around 7 while the
females and males are centered around 10. This makes sense because
`Rings` is a measure of age, so infants should have fewer `Rings` than
males and females.

Plots showing the relationship between `Rings` and all other numeric
variables are given below. There appears to be a positive relationship
between `Rings` and each numeric variable, but the relationships are not
clean linear relationships; there appears to be a lot of noise and some
nonlinearity, especially in the relationships between `Rings` and each
weight variable. In general, as the value of each predictor increases,
the spread of potential values of `Rings` also increases. We also see
that the infant abalones are smaller and have fewer rings, which is
expected. Furthermore, the male and female abalones, do not appear to
have clear separation in these plots and are generally similar.

``` r
train |> 
  pivot_longer(where(is.double) & !Rings,
               values_to = "Value") |> 
  ggplot(aes(Value, Rings, col = Sex)) +
  geom_point(alpha = .05) +
  facet_wrap(~name,
             scales = 'free') +
  # this lets us more easily see the color of the points on the legend
  guides(colour = guide_legend(override.aes = list(alpha = .5))) + 
  ggtitle("Relationship between Rings and Numeric Predictors")
```

<p align="center">
<img src="https://github.com/trgrimm/trgrimm.github.io/assets/70607091/94834cf3-e8e9-48c8-85ed-500a46f85e81" width="500">
</p>

To supplement the plots above, the plot below shows the pairwise
correlations between the numeric variables. All variables are positively
correlated with each other, and the variable that has the strongest
(linear) correlation with `Rings` is `Shell weight`.

``` r
ggcorrplot::ggcorrplot(train |>
                     select(where(is.numeric) & !id) |>
                     cor(),
                     title = 'Correlation Plot for Abalone Data',
                     type = 'lower',
                     lab = TRUE)
```

<p align="center">
<img src="https://github.com/trgrimm/trgrimm.github.io/assets/70607091/cb5d71e1-0c77-470d-8a18-dd0b3f27aeff" width="500">
</p>


# Building predictive models

Now that we’ve taken a look at the data and have a better understanding
of what it looks like, let’s start building some models.

## Defining models

We first need to define each model we want to compare in our workflow
later on. Here, we’ll use a penalized linear regression (elastic net)
model, $k$-nearest neighbors (KNN), random forest (RF), support vector
machine (SVM), extreme gradient boosting (XGBoost), and a neural network
(NN). When we define each model, we can also specify which tuning
parameters we want to tune later on to ensure optimal model fit.

``` r
# lasso/ridge/elastic net regression
lin_spec <- linear_reg(penalty = tune(),
                       mixture = tune()) |> 
  set_engine('glmnet')


# k nearest neighbors
knn_spec <- nearest_neighbor(neighbors = tune(),
                             weight_func = 'gaussian') |> 
  set_engine("kknn") |> 
  set_mode("regression")

# random forest
rf_spec <- rand_forest(mtry = tune(),
                       trees = tune(),
                       min_n = tune()) |>
  set_engine("randomForest") |>
  set_mode("regression")

# support vector machine (radial basis function)
svm_rad_spec <- svm_rbf(cost = tune(),
                        rbf_sigma = tune()) |>
  set_engine('kernlab') |>
  set_mode('regression')

# xgboost (extreme gradient boosting)
xgb_spec <- boost_tree(tree_depth = tune(),
                       learn_rate = tune(),
                       min_n = tune(),
                       sample_size = tune(),
                       trees = tune()) |>
  set_engine("xgboost") |>
  set_mode("regression")

# neural network with "brulee", which uses torch on the back-end
ann_spec <- mlp(hidden_units = tune(),
                penalty = tune(),
                epochs = tune(),
                learn_rate = tune()) |>
  set_engine("brulee") |>
  set_mode("regression")
```

## Setting up model workflow

We can now set up our full model workflow. To do this, we first specify
a “recipe” that defines what type of data preprocessing we want to do
prior to fitting the models. Some models require certain processing in
order to function properly. For example, linear and KNN models require
dummy encoding of categorical (factor) variables and normalization of
predictors. However, tree-based methods, such as RF and XGBoost, do not
require any transformation or normalization.

Here, for linear, KNN, SVM, and NN models, we’ll perform the Yeo-Johnson
transformation and normalize the numeric predictor variables, and create
dummy variables for `Sex`. We’ll also add a second-order polynomial term
for each weight variable to account for the nonlinearity we observed
during the EDA. For the tree-based methods, we will simply add a
second-order polynomial term for each weight variable.

``` r
# Define preprocessing recipes

# For linear, KNN, SVM, and NN models:
# - remove unnecessary variable (id)
# - apply Yeo-Johnson transformation to numeric variables
# - normalize numeric variables
# - create dummy variables (for Sex)
# - create 2nd order polynomial terms for each "weight" column
not_tree_preproc <- recipe(Rings ~., data = train) |> 
  step_rm(id) |> 
  step_YeoJohnson(all_numeric_predictors()) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_dummy(all_factor_predictors()) |> 
  step_poly(contains('weight'))

# For tree-based models (RF and XGBoost):
# - remove unnecessary variable (id)
# - create 2nd order polynomial terms for each "weight" column
tree_preproc <- recipe(Rings ~., data = train) |> 
  step_rm(id) |> 
  step_poly(contains('weight'))

# XGBoost requires us to create dummy variables, but RF does not:
xgb_preproc <- tree_preproc |> 
  step_dummy(all_factor_predictors())
```

Next, we put together a `workflow_set` that contains our preprocessing
recipe and the list of models we want to compare.

``` r
# Set up workflow with our preprocessing recipes and models we want to fit

# workflow for the methods that aren't tree-based
not_tree_wflow <- workflow_set(preproc = list(not_tree_preproc = not_tree_preproc),
                              models = list(lin = lin_spec,
                                            knn = knn_spec,
                                            svm_rad = svm_rad_spec,
                                            ann = ann_spec))

# workflows for tree-based methods
rf_wflow <- workflow_set(preproc = list(tree_preproc = tree_preproc),
                              models = list(rf = rf_spec))
xgb_wflow <- workflow_set(preproc = list(xgb_preproc = xgb_preproc),
                          models = list(xgb_spec))

# combine into a single workflow, rename workflow id's so they're all consistent
preproc_wflow <- bind_rows(not_tree_wflow, rf_wflow, xgb_wflow) |> 
  mutate(wflow_id = str_replace(wflow_id, '\\w+_preproc', 'preproc'))
```

## Tuning model parameters

Before we get final fits for each model, we need to tune the parameters
of each model to ensure optimal performance. To do this, we’ll perform
10-fold cross-validation (CV) with various sets of candidate tuning
parameter combinations.

We first set up the CV folds with `vfold_cv()`. Notice that we also
modify the `Rings` column so that
$\text{Rings} = \log(\text{Rings} + 1)$. This allows us to use the
built-in root mean square error (RMSE) metric (see code below) instead
of having to define a custom RMSLE metric, since RMSLE is not readily
available in `yardstick` (the package used by `tidymodels` that contains
metrics for model evaluation).

``` r
abalone_folds <- vfold_cv(train |> mutate(Rings = log(Rings + 1)), v = 10)
```

Now that we’ve defined our preprocessing recipe, models, and CV folds,
we can tune our models. Tuning is typically done by searching for the
best model performance over a grid of tuning parameters for each model.
However, this grid search approach can be very time-consuming.

To expedite model tuning, we’ll use racing\[^1\] with the `finetune`
package to tune the parameters of our models. We’ll also do this in
parallel across 3 cores with the `doParallel` package. Thankfully,
`tidymodels` makes this easy for us; all we have to do is set up the
parallel clusters, and `tidymodels` takes care of the parallelization
for us.

``` r
library(finetune)
race_ctrl <- control_race(save_pred = FALSE,
                          parallel_over = "everything",
                          save_workflow = TRUE)

# Set up parallel processing across 3 cores to decrease run time
library(doParallel)
cl <- makePSOCKcluster(3)
registerDoParallel(cl)

# Perform 10-fold CV tuning, obtain metrics for model fits
race_results <- preproc_wflow |>
  workflow_map("tune_race_anova", # use racing
               seed = 25,
               resamples = abalone_folds,
               grid = 10, # how many levels of each tuning parameter should we evaluate?
               control = race_ctrl,
               metrics = metric_set(rmse)) # optimization metric

# Shut down parallel processing cluster
stopCluster(cl)
```

Now that we’ve tuned our models, let’s look at the performance of the
best models:

``` r
# Table of best model results
race_results |>
  rank_results(select_best = 'TRUE') |>
  select(wflow_id, .config, rmse = mean, rank)
```

    # A tibble: 6 × 4
      wflow_id           .config                rmse  rank
      <chr>              <chr>                 <dbl> <int>
    1 preproc_boost_tree Preprocessor1_Model04 0.148     1
    2 preproc_rf         Preprocessor1_Model04 0.149     2
    3 preproc_svm_rad    Preprocessor1_Model01 0.154     3
    4 preproc_knn        Preprocessor1_Model8  0.156     4
    5 preproc_lin        Preprocessor1_Model10 0.159     5
    6 preproc_ann        Preprocessor1_Model04 0.177     6

``` r
# Visualize performance of best models
autoplot(race_results,
         rank_metric = 'rmse',
         metric = 'rmse',
         select_best = TRUE)
```

<p align="center">
<img src="https://github.com/trgrimm/trgrimm.github.io/assets/70607091/492a774a-1d77-4027-a4dd-363ab1a6d5f1" width="500">
</p>

The best model is XGBoost, which produces a 10-fold CV RMSLE of 0.148.
The performance of XGBoost is pretty close to RF, and the linear
(elastic net) and KNN models performed the worst by far.

## Evaluating test set performance

Now that we’ve found the best model with the best set of tuning
parameters, we can fit that model to the `test` set to predict values of
`Rings`.

First, we finalize our workflow with the best tuning parameters and fit
this to our testing data using the train/test split object `rand_split`
we created earlier.

``` r
# Get model info/tuning parameters for the best XGBoost model
best_results <- race_results |> 
  extract_workflow_set_result("preproc_boost_tree") |> 
  select_best(metric = 'rmse')

# Get final model with with the best model
xgb_test_results <- race_results |> 
  extract_workflow('preproc_boost_tree') |> 
  finalize_workflow(best_results) |> 
  last_fit(split = rand_split)


xgb_fit = race_results |> 
  extract_workflow('preproc_boost_tree') |> 
  finalize_workflow(best_results) |> 
  fit(train)

xgb_res <- augment(xgb_fit, test)
```

Let’s visualize the results by plotting the observed and predicted
values of `Rings`. We see that the observed and predicted values follow
a pretty linear trend, which means our model is providing predictions
that are similar to the observed values. However, there appears to be a
cloud of points on the far right that are being severely underpredicted
by our model.

``` r
xgb_test_results %>%
  collect_predictions() %>%
  ggplot(aes(x = Rings, y = .pred)) +
  geom_abline(color = "gray50", lty = 2) +
  geom_point(alpha = 0.5) +
  coord_obs_pred() +
  labs(x = "observed", y = "predicted")
```

<p align="center">
<img src="https://github.com/trgrimm/trgrimm.github.io/assets/70607091/8d9892aa-46e5-4353-bdde-5995ee267bf4" width="500">
</p>

# Conclusion

Overall, the process of exploring data, tuning and fitting models, and
obtaining predictions is very straightforward with `tidyverse` and
`tidymodels`. All of the code syntax is similar, and using the
`tidymodels` workflow to tune and compare many different models
simultaneously is almost effortless. There are also a lot of additional
things that could be considered in this analysis, such as additional
feature engineering and evaluating additional models (i.e., LightGBM,
deep NN’s). Furthermore, `tidymodels` has many additional cool features
that were not explored here that are useful. However, this post was a
nice way for me to demonstrate some simple things I’ve learned recently
with the `tidymodels` framework.

<br>
<br>

------------------------------------------------------------------------

# Bonus: Submitting predictions to Kaggle

Recall that the data came from Kaggle already split into train and test
sets, which I have called `kaggle_train` and `kaggle_test`. However, the
`kaggle_test` set does not contain the true values of `Rings`, making it
impossible to properly assess model performance for the purposes of the
analysis above.

Now that I’ve demonstrated an example of how we would fit and analyze
different models, we’re going to use the full `kaggle_train` and
`kaggle_test` datasets to obtain results that can be submitted to the
Kaggle competition.

Based on the poor performance of the NN, linear, and KNN models above,
I’m not going to use those models here so that I can save some
computation time. However, because the tree-based models (RF and
XGBoost) performed well above, I’m going to also tune and fit a light
gradient boosting (LightGBM) model for the full train and test sets.
Since LightGBM doesn’t require dummy variable encoding, we’ll apply the
same preprocessing steps as RF.

The code below performs all the model fitting, selection, and final
prediction steps that we did earlier. The only difference is that now
we’re using the full `kaggle_train` and `kaggle_test` datasets.

``` r
# Define lightgbm model
library(bonsai) # library needed for lightgbm
library(lightgbm)
lgbm_spec <- boost_tree(mtry = tune(), trees = tune(), tree_depth = tune(), 
  learn_rate = tune(), min_n = tune(), loss_reduction = tune()) |> 
  set_engine("lightgbm") |> 
  set_mode("regression")

# Set up model workflow
# workflow for SVM
not_tree_wflow <- workflow_set(preproc = list(not_tree_preproc = not_tree_preproc),
                              models = list(svm_rad = svm_rad_spec))

# workflow for tree-based methods
rf_wflow <- workflow_set(preproc = list(tree_preproc = tree_preproc),
                              models = list(rf = rf_spec,
                                            lgbm = lgbm_spec))
xgb_wflow <- workflow_set(preproc = list(xgb_preproc = xgb_preproc),
                          models = list(xgb_spec))

# combine into a single workflow, rename workflow id's so they're all consistent
preproc_wflow_final <- bind_rows(not_tree_wflow, rf_wflow, xgb_wflow) |> 
  mutate(wflow_id = str_replace(wflow_id, '\\w+_preproc', 'preproc'))

# define CV folds for tuning using the entire kaggle_train set
set.seed(6789)
abalone_folds_final <- vfold_cv(kaggle_train |> mutate(Rings = log(Rings + 1)), v = 10)


library(finetune)
race_ctrl <- control_race(save_pred = TRUE,
                          parallel_over = "everything",
                          save_workflow = TRUE)

# Set up parallel processing across 3 cores to decrease run time
library(doParallel)
cl <- makePSOCKcluster(3)
registerDoParallel(cl)

# Tune models using the CV folds defined above
race_results_final <- preproc_wflow_final |>
  workflow_map("tune_race_anova", # use racing
               seed = 25,
               resamples = abalone_folds_final,
               grid = 15,
               control = race_ctrl,
               metrics = metric_set(rmse))

stopCluster(cl)

# Table of best model results
race_results_final |>
  rank_results(select_best = 'TRUE') |>
  select(wflow_id, .config, rmse = mean, rank)
```

    # A tibble: 4 × 4
      wflow_id           .config                rmse  rank
      <chr>              <chr>                 <dbl> <int>
    1 preproc_boost_tree Preprocessor1_Model12 0.148     1
    2 preproc_lgbm       Preprocessor1_Model02 0.148     2
    3 preproc_rf         Preprocessor1_Model07 0.149     3
    4 preproc_svm_rad    Preprocessor1_Model11 0.152     4

Based on 10-fold CV, the XGBoost model is the best. Now, we can take the
best set of tuning parameters, finalize our model, and make predictions
on the full `kaggle_test` set.

``` r
# Get model info/tuning parameters for the best XGBoost model
best_results_final <- race_results_final |> 
  extract_workflow_set_result("preproc_boost_tree") |> 
  select_best(metric = 'rmse')

# Get final model with with the best model
xgb_fit_final = race_results_final |> 
  extract_workflow('preproc_boost_tree') |> 
  finalize_workflow(best_results_final) |> 
  fit(kaggle_train)

# Obtain predictions on kaggle_test data
xgb_res_final <- augment(xgb_fit_final, kaggle_test)
```

Now that we have our predictions, we can save them to a .csv file and
submit them to the Kaggle competition!

``` r
# store id and predictions in a tibble
test_preds <- xgb_res_final |>
  select(id, .pred) |> 
  rename(Rings = .pred)

# save the predictions to a .csv 
write.csv(test_preds, file = 'abalone_preds.csv', row.names = FALSE)
```


## Creating a stacked ensemble model

Above, we fit and evaluated several candidate models with different sets
of tuning parameters. Using model stacking, it is possible to weight the
predictions from multiple models to produce slightly improved
predictions. This can be done combining the results above with functions
from the `stacks` library.

### Creating the model stack

First, we create the model stack, adding a set of candidate models based
on the 10-fold CV model fits above.

``` r
library(stacks) # package needed to create a model ensemble via model stacking

# Create the model stack
abalone_stack <- stacks() |> 
  add_candidates(race_results_final)
```

Next, we “blend” these models in such a way that we improve our
predictions. Then, we fit the models with those weights to the full
training set. The final weights of each model can be shown in the plot
below.

``` r
# determine stacking coefficients (weight of each model for final predictions)
set.seed(1234)
blend <- blend_predictions(abalone_stack)

autoplot(blend, "weights") +
  geom_text(aes(x = weight + 0.01, label = model), hjust = 0) + 
  theme(legend.position = "none") +
  lims(x = c(-0.01, 0.8))
```

<p align="center">
<img src="https://github.com/trgrimm/trgrimm.github.io/assets/70607091/3993df18-21f3-4db6-ae83-ed4c3388f7dc" width="500">
</p>

``` r
# obtain final model fit to full training set
blend <- fit_members(blend)

blend
```

    ── A stacked ensemble model ─────────────────────────────────────


    Out of 6 possible candidate members, the ensemble retained 4.

    Penalty: 1e-06.

    Mixture: 1.


    The 4 highest weighted members are:

    # A tibble: 4 × 3
      member                  type        weight
      <chr>                   <chr>        <dbl>
    1 preproc_boost_tree_1_12 boost_tree  0.517 
    2 preproc_boost_tree_1_15 boost_tree  0.301 
    3 preproc_rf_1_08         rand_forest 0.137 
    4 preproc_lgbm_1_02       boost_tree  0.0488

We see that out of the 6 candidate models, 4 were kept in the stacked
ensemble. The two with the largest weights are XGBoost configurations.
The third is a RF, and the fourth is LightGBM.

We can now use this final ensemble model to obtain predictions and
submit these to Kaggle.

``` r
# make predictions on full testing set
blend_test_pred <- blend |> predict(kaggle_test)

# store id and predictions in a tibble
test_preds_ensemble <- blend_test_pred |>
  mutate(id = kaggle_test$id, .pred = exp(.pred) - 1) |>  # transform predictions back to original scale
  rename(Rings = .pred) |> 
  relocate(id, .before = Rings)

# save the predictions to a .csv 
write.csv(test_preds_ensemble, file = 'abalone_preds_ensemble.csv', row.names = FALSE)
```

## Final Kaggle competition results

After submitting the predictions, they are assessed by Kaggle once the competition ends. Our RMSLE
score for the final test set and leaderboard position are then reported. 

Based on predictions from our XGBoost and ensemble models above, we achieved a RMSLE score of:

* 0.14651 with just the XGBoost model, and
* 0.14579 with the ensemble model.

This places us in 379/2608 position in the final leaderboard standings from
the competition, meaning we are in the top 14.5% of participants, or the 85.5th percentile!

It’s possible to improve this RMSLE score by considering things such as
additional feature engineering, model preprocessing, and model tuning.
We could also consider using additional models such as deep NN’s.
However, everything here was kept pretty simple for the sake of
demonstrating how to perform a simple analysis with `tidymodels` to
produce pretty good results without a lot of additional effort.

\[^1\] Maron, O, and A Moore, (1994) “Hoeffding Races: Accelerating
Model Selection Search for Classification and Function Approximation”,
*Advances in Neural Information Processing Systems*, 59–66.
[link](https://proceedings.neurips.cc/paper/1993/file/02a32ad2669e6fe298e607fe7cc0e1a0-Paper.pdf)
