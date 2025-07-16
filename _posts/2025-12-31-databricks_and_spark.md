---
title: 'Modern Data Science Tools: Getting Started with Databricks and Spark'
date: 2025-12-31
permalink: /posts/2025/12/databricks_spark_intro/
tags:
  - machine learning
  - classification
  - big data
---

This post gives a simple introduction to popular modern data science technologies. In this post, we will explore features of the [Databricks](https://www.databricks.com/) platform and distributed computing with [Spark](https://spark.apache.org/) using census income data from the popular [Adult](https://archive.ics.uci.edu/dataset/2/adult) dataset, which is openly available on the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/).

Some features covered in this post include:

* running interactive code in Notebooks using the [Databricks Free Edition](https://www.databricks.com/learn/free-edition) platform
* querying the [Unity Catalog](https://www.databricks.com/product/unity-catalog) data lakehouse with PySpark and SQL
* using [PySpark](https://spark.apache.org/docs/latest/api/python/index.html) to efficiently access and manipulate data

<!--A future post will demonstrate the basics of using [MLflow](https://mlflow.org/) to log and monitor machine learning model runs and setting up automated training workflows.-->

<!-- Code to produce this blog post can be found in [this](https://github.com/trgrimm/) GitHub repository. -->

------------------------------------------------------------------------

# What is Databricks?

Databricks is a popular data science platform that facilitates everything from efficient exploratory data analysis to model building and deployment. Databricks can be hosted by popular cloud service providers, including Amazon (AWS), Microsoft (Azure), and Google (GCP), and allows for seamless connectivity to serverless and server computing, cloud storage, and a variety of IDEs (e.g., PyCharm, RStudio, VS Code).

Databricks also has a powerful Notebook capability that supports Python, R, Scala, and SQL languages. Databricks notebooks offer similar interactivity and interface as Jupyter notebooks, but are hosted on the Databricks platform and therefore have seamless access to connected cloud services. This makes real-time analysis of data and development of complex models extremely fast and easy.

One of the primary benefits of Databricks is that it is built on *Spark*.

# What is Spark?


