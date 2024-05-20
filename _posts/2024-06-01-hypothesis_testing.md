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

The value of $\alpha$ is also referred to as the probability of a Type 1 error. We reject the null hypothesis when the ***p*-value** is sufficiently small (i.e., *p*-value < $\alpha$). Alternatively, when *p*-value > $\alpha$, we fail to reject H<sub>0</sub>. Another important error to consider is the Type 2 error, which is equal to $(1 - \boldsymbol\beta)$ and is the probability of failing to reject H<sub>0</sub> when H<sub>1</sub> is true. Both of these errors relate to hypothesis tests that make an "incorrect" decision, both of which can have costly impacts. Therefore, it is important to properly understand the implications of each error whenever performing a hypothesis test. A more detailed discussion of Type 1 and Type 2 errors will be given in a future blog post.

Sometimes, we may achieve statistical significance, but the results may not be *practically significant*. For example, we may determine that a new drug is (statistically) significantly better at helping patients lose weight over 12 months compared to an existing treatment. However, if the difference in weight loss between the new and existing drugs is only 0.5 lb, this is not practically (or *clinically*) significant, and the new drug is not *practically* better than the existing one.

# Simple example: Comparing the efficacy of two drugs

Let's consider a simple example where we are interested in determining whether a new drug is better than an existing drug based on data from patients in a clinical trial. As is true for any scenario, the null and alternative hypotheses should be specific statements that directly relate to a question we want to answer using the data that we have. For this example, we are interested in the following simple hypotheses:

H<sub>0</sub>: There is no difference in the effects of the new drug and the existing drug.

H<sub>1</sub>: The new drug is better than the existing drug.

## Defining specific hypotheses

To actually perform a hypothesis test, the idea of "no difference" and one drug being "better" than the other need to be mathematically defined. Let's say that the new drug must provide a certain amount of additional benefit to patients relative than the existing drug. For example, consider a new weight-loss drug. We'll say that the new drug is considered "better" than the existing drug if the average weight loss for patients on the new drug is at least 5 lb higher than for patients on the existing drug. Our specific hypotheses are now:

H<sub>0</sub>: The average weight loss for patients on the new drug is less than 5 lb greater than that of patients on the existing drug.

H<sub>1</sub>: The average weight loss for patients on the new drug is at least 5 lb greater than that of patients on the existing drug.

Let $\mu_\text{new}$ and $\mu_\text{existing}$ be the average weight loss for patients on the new and existing drugs, respectively. We now have the following mathematical hypotheses:

H<sub>0</sub>: $\mu_\text{new} = \mu_\text{existing}$, or $\mu_\text{new} - \mu_\text{existing} = 0$. (no difference)

H<sub>1</sub>: $\mu_\text{new} - \mu_\text{existing} \ge 5$. (at least 5lb additional weight loss on the new drug)

## Testing our hypothesis

Now that we've defined our hypotheses, we can perform a test to determine whether our data provides sufficient evidence to reject H<sub>0</sub>. In other words, we can perform a test to decide whether our new drug is indeed better than the existing drug.

Suppose we observe the following:


Because we're comparing averages of weight loss between two groups, we'll use a two-sample [*t*-test](https://en.wikipedia.org/wiki/Student%27s_t-test) to test our hypothesis.




