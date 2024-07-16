---
title: 'Basics of Hypothesis Testing'
date: 2024-06-01
permalink: /posts/2024/06/hypothesis-testing/
tags:
  - statistics
  - hypothesis testing
---

This post explains the basics of hypothesis testing and provides a simple hypothetical pharmaceutical example of testing whether a new drug is better than an existing drug.

<!-- , including a discussion type I and II errors and statistical power.-->

------------------------------------------------------------------------

# What is hypothesis testing?

Hypothesis testing is the statistical method used to determine whether a specific hypothesis is supported by available data. This procedure is important when it comes to making informed decisions using data and is used in a variety of fields and applications. There are two hypotheses involved in any hypothesis test: the *null* and *alternative* hypotheses. These hypotheses are typically formulated as follows:

* **Null hypothesis** (H<sub>0</sub>): States that there is no effect, or that any observed effect is simply due to random chance ("null").
* **Alternative hypothesis** (H<sub>1</sub> or H<sub>a</sub>): States that there is an effect not due to random chance.

Statistical tests are designed to work under the assumption that the null hypothesis is true. However, if the data provides sufficient evidence that the null hypothesis is false (the observed data is highly unlikely due to random chance alone), we say that we reject H<sub>0</sub> in favor of H<sub>1</sub>, meaning that we'll make decisions based on H<sub>1</sub> being true. Otherwise, we say that there is insufficient evidence to reject the null hypothesis, or that we "fail to reject" H<sub>0</sub>.

We usually define the hypotheses so that the alternative is the one we're trying to "prove" and the null is one we assume to be false, but this is not always the case. We usually do this so that when we reject H<sub>0</sub> in favor of H<sub>1</sub>, it is a decision that is sufficiently supported by the data. We cannot "accept" or "prove" H<sub>0</sub>; instead, when we fail to reject H<sub>0</sub>, it is because we have *insufficient* evidence to reject H<sub>0</sub>, not because we have sufficient evidence to "disprove" H<sub>1</sub>. This may seem like a simple matter of semantics, but it's important to understand so that we can correctly specify our null and alternative hypotheses for the problem of interest.

Applications of hypothesis testing arise in almost every field. A few examples include
* **Pharmaceutical development**: Testing the efficacy of a new drug relative to an existing treatment
* **Website design**: Determining whether one website design performs better than another
* **Quality control**: Deciding if a process is manufacturing items according to the desired specifications
* **Agriculture**: Testing whether a new fertilizer is better than existing fertilizers
* **Sociology**: Assessing if there are significant differences in crime rates across different socioeconomic groups

# Statistical significance

Whenever we want to test a hypothesis, we need to determine when we should reject H<sub>0</sub> and when we fail to reject H<sub>0</sub>. This rule for rejecting or failing to reject H<sub>0</sub> needs to be statistically sound and based on the data we've observed. We define this rule by determining whether our data is **statistically significant**. 

To understand statistical significance, we'll first define some important terms:
* $\boldsymbol\alpha$: the significance level, which is the probability of rejecting H<sub>0</sub> when H<sub>0</sub> is true (incorrectly rejecting H<sub>0</sub>)
* $\boldsymbol\beta$: the *power* of the test, which is the probability of rejecting H<sub>0</sub> when H<sub>1</sub> is true (correctly rejecting H<sub>0</sub>)
* ***p*-value**: the probability of obtaining results as extreme or more extreme than what we've observed
* **practical significance**: whether the observed effect is practically meaningful in real life

The value of $\alpha$ is also referred to as the probability of a Type I error. We reject the null hypothesis when the ***p*-value** is sufficiently small (i.e., *p*-value < $\alpha$). Alternatively, when *p*-value > $\alpha$, we fail to reject H<sub>0</sub>. Another important error to consider is the Type II error, which is equal to $(1 - \boldsymbol\beta)$ and is the probability of failing to reject H<sub>0</sub> when H<sub>1</sub> is true. Both of these errors relate to hypothesis tests that make an "incorrect" decision, which can have costly impacts. Therefore, it is important to properly understand the implications of each error whenever performing a hypothesis test. A more detailed discussion of Type I and Type II errors, along with power, will be given in a future blog post.

Sometimes, we may achieve statistical significance, but the results may not be *practically significant*. For example, we may determine that a new drug is (statistically) significantly better at helping patients lose weight over 12 months compared to an existing treatment. However, if the difference in weight loss between the new and existing drugs is only 0.5 lb, this may not be considered practically (or *clinically*) significant, meaning that the new drug is not *practically* better than the existing one.

# Simple example: Comparing the efficacy of two drugs

Let's consider a simple example where we are interested in determining whether a new drug is better than an existing drug based on data from patients in a clinical trial. As is true for any scenario, the null and alternative hypotheses should be specific statements that directly relate to a question we want to answer using the data that we have. For this example, we are interested in the following simple hypotheses:

H<sub>0</sub>: There is no difference in the effects of the new drug and the existing drug.

H<sub>1</sub>: The new drug is better than the existing drug.

## Defining specific hypotheses

To actually perform a hypothesis test, the idea of "no difference" and one drug being "better" than the other need to be mathematically defined. Let's say that the new drug must provide a certain amount of additional benefit to patients relative than the existing drug. For example, consider a new weight-loss drug. We'll say that the new drug is considered "better" than the existing drug if the average weight loss for patients on the new drug is at least 5 lb higher than for patients on the existing drug. Our specific hypotheses are now:

H<sub>0</sub>: The average weight loss for patients on the new drug is less than 5 lb greater than that of patients on the existing drug.

H<sub>1</sub>: The average weight loss for patients on the new drug is at least 5 lb greater than that of patients on the existing drug.

Let $\mu_\text{new}$ and $\mu_\text{existing}$ be the average weight loss for patients on the new and existing drugs, respectively. We now have the following mathematical hypotheses:

H<sub>0</sub>: $\mu_\text{new} - \mu_\text{existing} < 5$. (difference in weight loss is less than 5 lb between the drugs)

H<sub>1</sub>: $\mu_\text{new} - \mu_\text{existing} \ge 5$. (at least 5lb additional weight loss on the new drug)

## Testing our hypothesis

Now that we've defined our hypotheses, we can perform a test to determine whether our data provides sufficient evidence to reject H<sub>0</sub>. In other words, we can perform a test to decide whether our new drug is indeed better than the existing drug. Here, we'll use a significance level of $\alpha = 0.05$, which is common.

Suppose we observe the following:

* Group on the new treatment: 45 patients, average weight loss of 15 lb, weight loss standard deviation of 9 lb
* Group on the existing treatment: 42 patients, average weight loss of 8 lb, weight loss standard deviation of 10 lb

Clearly, the difference in average weight loss between groups is $15 - 8 = 7$, which is greater than 5. However, we need to use a statistical test to determine the probability of observing data such as this, at random, if the null hypothesis is true.

To compare the averages of weight loss between these two groups, we'll use a special version of a two-sample [*t*-test](https://en.wikipedia.org/wiki/Student%27s_t-test), called a [Welch's *t*-test](https://en.wikipedia.org/wiki/Welch%27s_t-test) to test our hypothesis. If we had individual-level data for all $45 + 42$ patients, we could use the `t.test()` function in R. However, because we only have summary statistics for this example, we'll use the `tsum.test()` function in the `BDSA`[^1] package in R.

The code and output are given below:

``` r
library(BSDA)

# Let "x" be the new treatment group, and "y" be the existing treatment group.
# mean.x, mean.y are the group means (average weight loss)
# s.x, s.y are the group standard deviations for weight loss
# n.x, n.y are the sample sizes for each group
# mu is the hypothesized difference between groups (5 lb)

tsum.test(mean.x = 15, s.x = 9, n.x = 45,
          mean.y = 8, s.y = 10, n.y = 42,
          alternative = 'greater', mu = 5)

```

	    Welch Modified Two-Sample t-Test

    data:  Summarized x and y
    t = 0.97812, df = 82.492, p-value = 0.1654
    alternative hypothesis: true difference in means is greater than 5
    95 percent confidence interval:
     3.598506       NA
    sample estimates:
    mean of x mean of y 
           15         8 

Above, the *p*-value is 0.1654, which is larger than $\alpha = 0.05$. In this case, we fail to reject H<sub>0</sub> and conclude that there is insufficient evidence to say that the average weight loss for patients who take the new drug is at least 5 lb greater than the average weight loss for patients who take the existing drug.

Now, let's pretend we actually have data for 300 total patients (150 on each drug) with the exact same summary statistics as above. That is, we have

* Group on the new treatment: 150 patients, average weight loss of 15 lb, weight loss standard deviation of 9 lb
* Group on the existing treatment: 150 patients, average weight loss of 8 lb, weight loss standard deviation of 10 lb

Here, we can run the same test with slight modifications for the sample sizes:

``` r
tsum.test(mean.x = 15, s.x = 9, n.x = 45,
          mean.y = 8, s.y = 10, n.y = 42,
          alternative = 'greater', mu = 5)
```
    	Welch Modified Two-Sample t-Test
    
    data:  Summarized x and y
    t = 1.8207, df = 294.75, p-value = 0.03483
    alternative hypothesis: true difference in means is greater than 5
    95 percent confidence interval:
     5.187458       NA
    sample estimates:
    mean of x mean of y 
           15         8 

When we had smaller sample sizes, our *p*-value was greater than $0.05$. Now, the *p*-value is 0.035, which would cause us to reject H<sub>0</sub> and conclude that there the average weight loss for patients who take the new drug is indeed at least 5 lb greater than the average weight loss for patients who take the existing drug.

Based on the results from these two scenarios, we see that the results of a hypothesis test can be affected by many factors, including the sample size, the value of $\alpha$, and the observed difference between the treatment groups. There are also other things that affect the results of a hypothesis test, but they are beyond the scope of this blog post.

# Conclusion

Hypothesis testing is important in many fields and is essential to data-driven decision making. There are a lot of things to consider whenever performing a hypothesis test, and in practice, many things are often not considered or are done incorrectly. This post was intended to provide a very basic introduction to hypothesis testing with a simple example to demonstrate how to interpret results of a hypothesis test. However, many important things have not been discussed here, such as selecting which statistical test to use, assessing whether a test is appropriate, and selecting the value of $\alpha$, among many other considerations.


[^1]: Arnholt A, Evans B (2023). "BSDA: Basic Statistics and Data Analysis", R package version 1.2.2, [link](https://CRAN.R-project.org/package=BSDA).
