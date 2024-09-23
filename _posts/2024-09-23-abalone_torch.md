---
title: 'Analysis Walkthrough: Supervised Regression with Abalone Data'
date: 2024-09-23
permalink: /posts/2024/09/abalone_tidymodels/
tags:
  - machine learning
  - regression
  - statistics
---

This post illustrates a neural network using the `torch` package in R to
revisit the abalone [Kaggle
competition](https://www.kaggle.com/competitions/playground-series-s4e4/overview),
which is a supervised regression problem described and analyzed in a
[previous](https://trgrimm.github.io/posts/2024/05/abalone_tidymodels/)
blog post using `tidymodels`.

<!-- Code to produce this blog post can be found in [this](https://github.com/trgrimm/abalone_analysis) GitHub repository. -->

# Data description

The Abalone data for this analysis comes from a [Kaggle playground
prediction
competition](https://www.kaggle.com/competitions/playground-series-s4e4/overview)
titled “Regression with an Abalone Dataset”. 

The data was explored in
[this](https://trgrimm.github.io/posts/2024/05/abalone_tidymodels/)
previous blog post, so a brief description is given here. I will also refrain from performing any exploratory data analysis (EDA)
in this post because it was done previously.

As a reminder, the abalone dataset contains the following variables:

-   8 predictor variables: sex, length, diameter, height, etc.
-   1 numeric response variable: `Rings`

**Analysis Goal:** Predict the number of `Rings` using the easily
obtained physical measurements (predictor variables).

Train and test datasets are provided by Kaggle, and we want to minimize
the root mean squared logarithmic error (RMSLE), which is defined as

$$
\text{RMSLE} = \sqrt{\frac{1}{n} \sum\_{i=1}^n \left(\log(1 + \hat{y}\_i) - \log(1 + y_i)\right)^2},
$$

where

-   *n* = number of observations in the test set
-   *ŷ*<sub>*i*</sub> is the predicted value of `Rings` for observation
    *i*
-   *y*<sub>*i*</sub> is the observed value of `Rings` for observation
    *i*
-   log  is the natural logarithm.

For this analysis, we want to be able to evaluate our results to see how
well we do on testing data. However, Kaggle does not release the true
values of the response variable of the test set, even after the
competition has ended.

First, I’ll set up a new train/test split using the `train` data
provided by Kaggle. We’ll use this as our train/test data throughout the
analysis below. After we’ve looked at those results, I’ll use the full
original `train` set to obtain predictions for Kaggle’s `test` set so
that we can enter those predictions into the competition to see our
final score.

To compare the results of our custom neural network (NN) from torch,
I’ll also fit a neural network through the `tidymodels` framework with
the `mlp()` model using the `brulee` package, which actually uses
`torch` on the back-end. I’ll include this out of curiosity to compare
the performance of a custom NN versus a quick out-of-the-box
solution that is easily implemented through `tidymodels`.

# Load in and set up data

First, we load in the `tidyverse`, `torch`, and `tidymodels` packages,
which will be used throughout this analysis. Then, we load in the train
and test sets. These are stored in separate .csv files.

``` r
library(tidyverse)
library(torch)
library(tidymodels)

kaggle_train <- read_csv('abalone_data/train.csv', col_types = 'ifdddddddi')
kaggle_test <-read_csv('abalone_data/test.csv', col_types = 'ifddddddd')
```

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

Now, I will set up the NN in `torch`. This consists of using the
`nn_module()` function to define the architecture of our desired NN.
We’ll keep it pretty simple here by using an input layer, two hidden
layers, and an output layer, with a ReLU activation function for each hidden layer. Note that this is more complicated than the
`brulee` NN in `torch`, which only uses one hidden layer.

``` r
net <- nn_module(
  # Define the layers
  initialize = function(d_in, d_hidden1, d_hidden2, d_out) {
    self$net <- nn_sequential(
      # hidden layer 1
      nn_linear(d_in, d_hidden1),
      nn_relu(),
      # hidden layer 2
      nn_linear(d_hidden1, d_hidden2),
      nn_relu(),
      # output layer
      nn_linear(d_hidden2, d_out)
    )
  },
  # Execute the network. In this case, simply fit self$net() to the data (x) since
  # the order of the layers and activation functions is already defined in self$net with nn_sequential()
  forward = function(x) {
    self$net(x)$flatten(start_dim=1)
  }
)
```

The NN structure is now defined, so we just need to set up our data and
train the model.

To set up our data, let’s first define the same preprocessing steps
(recipe) that we used in the previous analysis with `tidymodels`. This
allows us to keep the dataset consistent for both our `torch` model and
the `tidymodels` version.

``` r
# Perform the following preprocessing steps:
# - remove unnecessary variable (id)
# - apply Yeo-Johnson transformation to numeric variables
# - normalize numeric variables
# - create dummy variables (for Sex)
# - create 2nd order polynomial terms for each "weight" column

preprocess <- recipe(Rings ~., data = train) |> 
  step_rm(id) |> 
  step_YeoJohnson(all_numeric_predictors()) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_dummy(all_factor_predictors()) |> 
  step_poly(contains('weight'))
```

Now, let’s apply the preprocessing steps to our train and test sets:

``` r
# Process training data
train_processed <- preprocess |> 
  prep() |>  # fit to training data
  bake(train) # apply preprocessing to training data

# Process testing data
# (use estimated parameters from fitting to the training data)
test_processed <- preprocess |> 
  prep() |>  # fit to training data
  bake(test) # apply preprocessing to testing data
```

Now that we’re done with preprocessing, we need to set up a `dataset`
function and `dataloader` for each dataset. This allows us to pass the
data easily in batches to the NN we defined earlier for computationally
efficient training purposes. Here, we’ll use batches of size 128.

``` r
make_dataset <- dataset(
  name = "make_dataset()",
  initialize = function(df) {
    self$x <- as.matrix(df |> select(!Rings)) %>% torch_tensor()
    self$y <- torch_tensor(df$Rings)$to(torch_float())
  },
  .getitem = function(i) {
    list(x = self$x[i, ], y = self$y[i])
  },
  .length = function() {
    dim(self$x)[1]
  }
)

# Do the same thing we did in the last analysis, where we use log(Rings + 1)
# so that we can use MSE loss instead of defining a custom RMSLE loss function
train_dataset_processed <- make_dataset(train_processed |> mutate(Rings = log(Rings + 1)))
test_dataset_processed <- make_dataset(test_processed |> mutate(Rings = log(Rings + 1)))

train_dl <- dataloader(train_dataset_processed, batch_size = 128, shuffle = TRUE)
test_dl <- dataloader(test_dataset_processed, batch_size = 128, shuffle = TRUE)
```

Preprocessing is now done, and the data is all ready to go! All we have
to do now is fit our model, which we do below. Model fitting and
training is simplified using the `luz` package, which allows us to use
the `setup()`, `set_hparams()`, and `fit()` functions below:

``` r
library(luz)

# Define NN node sizes
d_in <- ncol(train_processed) - 1  # Number of predictor columns (features)
d_hidden1 <- 32  # Size of the 1st hidden layer
d_hidden2 <- 16  # Size of the 2nd hidden layer
d_out <- 1   # Output size for regression (one output)


# Fit our NN:
fitted <- net %>%
# Set up out loss function, optimizer, and desired metric to print
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_rmse())
  ) %>%
# Pass NN model parameters
  set_hparams(
    d_in = d_in,
    d_hidden1 = d_hidden1, 
    d_hidden2 = d_hidden2,
    d_out = d_out
  ) %>%
# Fit the model to the training data,
# validate using the our testing data
  fit(train_dl,
      epochs = 200,
      valid_data = test_dl,
      callbacks = list(
        # implement early stopping if no improvement is seen for 20 epochs
        luz_callback_early_stopping(patience = 20))
  )

# Obtain performance metric for the fitted model on the test set
fitted %>% evaluate(test_dl)
```

    A `luz_module_evaluation`
    ── Results ─────────────────────────────────────────────────────────────────────
    loss: 0.0239
    rmse: 0.1548

This model achieves a RMSLE of 0.1544 on the test set.

Now, we’ll perform the same model training/fitting steps for the
`tidymodels` MLP with `brulee` as we did in the
[previous](https://trgrimm.github.io/posts/2024/05/abalone_tidymodels/)
analysis:

``` r
# Set up 10-fold cross-validation folds
set.seed(5678)
folds <- vfold_cv(train |> mutate(Rings = log(Rings + 1)), v = 10)

# Specify model and parameteres to tune
# - use multilayer perceptron (MLP) from the brulee package,
# which uses torch on the back-end
brulee_spec <- mlp(hidden_units = tune(),
                   penalty = tune(),
                   epochs = tune(),
                   learn_rate = tune()) |>
  set_engine("brulee") |>
  set_mode("regression")

# Set up model workflow
brulee_wflow <- workflow() |> 
  add_recipe(preprocess) |> 
  add_model(brulee_spec)

# Adjust tuning range for epochs
brulee_params <- brulee_wflow |> 
  extract_parameter_set_dials() |> 
  update(epochs = epochs(c(50, 200)))


# Set up parallel processing to expedite tuning
library(doParallel)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# Execute workflow, obtaining 10-fold CV RMSE values across all tuning parameter combinations
brulee_tune <- brulee_wflow |> 
  tune_grid(folds,
            grid = brulee_params |> grid_regular(levels = 3),
            metrics = metric_set(rmse))

stopCluster(cl)

brulee_tune |> show_best()
```

    # A tibble: 5 × 10
      hidden_units  penalty epochs learn_rate .metric .estimator  mean     n std_err
             <int>    <dbl>  <int>      <dbl> <chr>   <chr>      <dbl> <int>   <dbl>
    1           10    1e-10    125      0.316 rmse    standard   0.176    10 5.74e-4
    2           10    1e- 5    200      0.316 rmse    standard   0.176    10 8.33e-4
    3            5    1e-10    125      0.316 rmse    standard   0.176    10 1.11e-3
    4           10    1e- 5    125      0.316 rmse    standard   0.177    10 9.28e-4
    5            5    1e- 5    200      0.316 rmse    standard   0.177    10 1.09e-3
    # ℹ 1 more variable: .config <chr>

``` r
# The best tuned model has a hidden layer with 10 neurons, 
# a penalty of 1e-10, 125 epochs, and learn_rate = 0.316
brulee_tune |> select_best(metric = 'rmse')
```

    # A tibble: 1 × 5
      hidden_units      penalty epochs learn_rate .config              
             <int>        <dbl>  <int>      <dbl> <chr>                
    1           10 0.0000000001    125      0.316 Preprocessor1_Model66

The 10-fold CV RMSLE for the `brulee` MLP is 0.176.

Based on the results so far, we expect that the custom NN implemented
through `torch` with 2 hidden layers will perform better than the
`mlp()` model implemented through `brulee` and `tidymodels`.

Let’s see what happens when we train both models on the full training
set and use those to predict the values of `Rings` in the original
testing set provided by Kaggle.

First, we take the best `brulee` model, train it on the full
`kaggle_train` set, and obtain predictions on the full `kaggle_test`
set:

``` r
set.seed(24)
# Get model info/tuning parameters for the best torch nn model
best_results_final <- brulee_tune |> 
  select_best(metric = 'rmse')

# Get final model with with the best model
brulee_fit_final = brulee_wflow |> 
  finalize_workflow(best_results_final) |> 
  fit(kaggle_train)

# Obtain predictions on kaggle_test data
brulee_res_final <- augment(brulee_fit_final, kaggle_test)

# Store predictions in a tibble
brulee_test_preds <- brulee_res_final |>
  select(id, .pred) |> 
  rename(Rings = .pred)

# Save the predictions to a .csv 
write.csv(brulee_test_preds, file = 'abalone_preds_brulee.csv', row.names = FALSE)
```

Now, we do the same thing for the custom `torch` model:

``` r
# Train torch NN on entire kaggle_train set:

# Set up preprocessing
preprocess_final <- recipe(Rings ~., data = kaggle_train) |> 
  step_rm(id) |> 
  step_YeoJohnson(all_numeric_predictors()) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_dummy(all_factor_predictors()) |> 
  step_poly(contains('weight'))

# Preprocess training data
train_processed_final <- preprocess_final |> 
  prep() |>
  bake(train)

# Set up dataset() and dataloader
train_dataset_processed_final <- make_dataset(train_processed_final |> mutate(Rings = log(Rings + 1)))
train_dl_final <- dataloader(train_dataset_processed_final, batch_size = 128, shuffle = TRUE)

set.seed(25)
# Train NN
fitted_final <- net %>%
# Set up out loss function, optimizer, and desired metric to print
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_rmse())
  ) %>%
# Pass NN model parameters
  set_hparams(
    d_in = d_in,
    d_hidden1 = d_hidden1, 
    d_hidden2 = d_hidden2,
    d_out = d_out
  ) %>%
# Fit the model to the training data,
# validate using the our testing data
  fit(train_dl_final,
      epochs = 200,
      callbacks = list(
        # implement early stopping if no improvement is seen for 20 epochs
        luz_callback_early_stopping('train_loss', patience = 20))
  )

# format full test set
kaggle_test_processed <- preprocess_final |> 
  prep() |>  # fit to training data
  bake(kaggle_test) # obtain final test set for predictions


# Obtain predictions using the fitted torch NN
nn_preds <- fitted_final |>
  predict(as.matrix(kaggle_test_processed))

# store observation id and predictions in a tibble
torch_test_preds <- tibble(id = kaggle_test$id,
                           Rings = exp(as.double(nn_preds)) - 1)

# Save the predictions to a .csv 
write.csv(torch_test_preds, file = 'abalone_preds_torch.csv', row.names = FALSE)
```

## Final Kaggle test set results

After submitting the predictions, they are assessed by Kaggle, and the
final RMSLE score on the testing data is reported.

We achieved the final RMSLE scores:

-   `brulee`: 0.17537
-   `torch`: 0.15350

The `torch` score is clearly better (lower) than the `brulee` score,
meaning that our more complicated 2 hidden layer (versus 1 hidden layer)
architecture provided additional predictive power. However, in our
[previous](https://trgrimm.github.io/posts/2024/05/abalone_tidymodels/)
analysis, we achieved an even better final RMSLE of 0.14602 using an
ensemble approach.

It’s possible that our model would improve as the amount of training
data increased; neural networks are known to be very good when several
hundred thousand to millions of observations are available, but that is
not the case here. We could also investigate performance with different
NN architectures by varying things such as activation functions, number
of neurons in the hidden layers, number of hidden layers, etc.
