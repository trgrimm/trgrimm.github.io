---
title: 'Basics of Hypothesis Testing'
date: 2024-06-01
permalink: /posts/2024/06/hypothesis-testing/
tags:
  - statistics
  - hypothesis testing
---

This post explains the basics of hypothesis testing, including a discussion type I and II errors and statistical power.

# What is hypothesis testing?

Hypothesis testing is the statistical method used to determine whether a specific hypothesis is supported by available data. This procedure is important when it comes to making informed decisions using data and is used in a variety of fields and applications. There are two hypotheses involved in any hypothesis test: the *null* and *alternative* hypotheses. These hypotheses are typically formulated as follows:

* **Null hypothesis** (H<sub>0</sub>): States that there is no effect, or that any observed effect is simply due to random chance ("null").
* **Alternative hypothesis** (H<sub>1</sub> or H<sub>a</sub>): States that there is an effect not due to random chance.

Statistical tests are designed to work under the assumption that the null hypothesis is true. However, if the data provides sufficient evidence that the null hypothesis is false, we say that we reject H<sub>0</sub> in favor of H<sub>1</sub> and decide that H<sub>1</sub> is true. Otherwise, we say that there is insufficient evidence to reject the null hypothesis, or that we "fail to reject" H<sub>0</sub>

We usually define the hypotheses so that the alternative is the one we're trying to "prove" and the null is one we assume to be false. However, this is not always the case. We usually do this so that when we reject H<sub>0</sub> in favor of H<sub>1</sub>, it is a decision that is sufficiently supported by the data. If we instead wanted to "prove" H<sub>0</sub>, this would only be done by having "insufficient evidence" to reject H<sub>1</sub>.

Applications of hypothesis testing arise in almost every field. A few examples include
* **Pharmaceutical development**: Testing the efficacy of a new drug relative to an existing treatment
* **Website design**: Determine whether one website design performs better than another
* **Quality control**: Decide if a process is manufacturing items according to the desired specifications
* **Agriculture**: Test whether a new fertilizer is better than existing fertilizers
* **Sociology**: Assess if there are significant differences in crime rates across different socio-economic groups

## Simple example: Comparing the efficacy of two drugs

The null and alternative hypotheses should be specific statements that directly relate to a question we want to answer using the data that we have. For example, suppose we have data describing the effects of a new drug and an existing drug on patients in a clinical trial. We are interested in the following simple hypotheses:

H<sub>0</sub>: There is no difference in the effects of the new drug and the existing drug.

H<sub>1</sub>: The new drug is better than the existing drug.

### Defining specific hypotheses

To actually perform a hypothesis test, the idea of "no difference" and one drug being "better" than the other would need to be mathematically described in order to test this hypothesis. Let's say that the new drug must provide a certain amount of additional benefit to patients relative than the existing drug. For example, consider a new weight-loss drug. We'll say that the new drug is considered "better" than the existing drug if the average weight loss for patients on the new drug is at least 5lb higher than for patients on the existing drug.

Our specific hypotheses are now

H<sub>0</sub>: The average weight loss for patients on the new drug is less than 5 pounds greater than that of patients on the existing drug.

H<sub>1</sub>: The average weight loss for patients on the new drug is at least 5 pounds greater than that of patients on the existing drug.

Let $\mu_\text{new}$ and $\mu_\text{existing}$ be the average weight loss for patients on the new and existing drugs, respectively. We now have the following mathematical hypotheses:

H<sub>0</sub>: $\mu_\text{new} = \mu_\text{existing}$, or $\mu_\text{new} - \mu_\text{existing} = 0$. (no difference)

H<sub>1</sub>: $\mu_\text{new} - \mu_\text{existing} \ge 5$. (at least 5lb additional weight loss on the new drug)

### Testing our hypothesis

Now that we've defined our hypotheses, we can perform a test to determine whether our data provides



