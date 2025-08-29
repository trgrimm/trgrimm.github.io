---
title: 'Modern Data Science Tools: Getting Started with Databricks and Spark'
date: 2025-08-29
permalink: /posts/2025/08/databricks_spark_intro/
tags:
  - machine learning
  - classification
  - big data
---

This post gives a simple introduction to popular modern data science technologies. In this post, we will explore features of the [Databricks](https://www.databricks.com/) platform and distributed computing with [Spark](https://spark.apache.org/) using data that is freely available in the Databricks Free Edition.

Some features covered in this post include:

* running interactive code in Notebooks using the [Databricks Free Edition](https://www.databricks.com/learn/free-edition) platform
* querying the [Unity Catalog](https://www.databricks.com/product/unity-catalog) data lakehouse with PySpark and SQL
* using [PySpark](https://spark.apache.org/docs/latest/api/python/index.html) to efficiently access and manipulate data

<!--A future post will demonstrate the basics of using [MLflow](https://mlflow.org/) to log and monitor machine learning model runs and setting up automated training workflows.-->

<!-- Code to produce this blog post can be found in [this](https://github.com/trgrimm/) GitHub repository. -->

------------------------------------------------------------------------

# What is Databricks? 

<p align="center">
    <img src="https://github.com/user-attachments/assets/cefae77a-79eb-4beb-a0dd-978c12d11466" width="300">
</p>

Databricks is a popular data science platform that facilitates everything from efficient exploratory data analysis to model building and deployment. Databricks can be hosted by popular cloud service providers, including Amazon (AWS), Microsoft (Azure), and Google (GCP), and allows for seamless connectivity to serverless and server computing, cloud storage, and a variety of IDEs (e.g., PyCharm, RStudio, VS Code). The [Databricks Free Edition](https://www.databricks.com/learn/free-edition) of Databricks can be accessed simply by creating an account with an email, and it provides free access to many of the features available in Databricks.

Databricks also has a powerful Notebook capability that supports Python, R, Scala, and SQL languages. Databricks notebooks offer similar interactivity and interface as Jupyter notebooks, but are hosted on the Databricks platform and therefore have seamless access to connected cloud services. This makes real-time analysis of data and development of complex models extremely fast and easy.

One of the primary benefits of Databricks is that it is built on *Spark*.

# What is Spark?

<p align="center">
    <img src="https://github.com/user-attachments/assets/c0e88422-a9e3-4dd4-811a-40701ec69789" width="200">
</p>

[Spark](https://spark.apache.org/docs/latest/index.html) is a popular framework that facilitates distributed processing for working with big data. Spark is most commonly used within Scala or Python (PySpark) and allows for lazy data manipulation and efficient processing. One great benefit of Spark is that it allows for easy processing of big datasets that do not fit in memory.

------------------------------------------------------------------------

# Getting Started with Databricks Notebooks

Databricks notebooks are the native solution for working with data in Databricks. Inside a notebook, there are cells that can run R, Python, Scala, SQL, or Markdown code. 

<img width="1217" height="303" alt="Screenshot 2025-08-28 at 8 55 14 PM" src="https://github.com/user-attachments/assets/26210883-8af3-415f-8fce-06f6726b9e4b" />

Running code is as simple as typing some code in a cell and clicking the Run Cell button (or ⌘+Enter/Cntrl+Enter).

<img width="1038" height="124" alt="Screenshot 2025-08-28 at 8 52 52 PM" src="https://github.com/user-attachments/assets/0d83a722-4f72-4e10-9bcc-529fdf3a7217" />

# Accessing the Unity Catalog

The Unity Catalog (UC) is a data lakehouse where structured data is stored in Delta Tables and can easily be accessed directly in Notebooks by SQL or Spark queries.

Start by navigating to the Unity Catalog:

<img width="540" height="250" alt="Screenshot 2025-08-28 at 9 06 21 PM" src="https://github.com/user-attachments/assets/c42cd266-474d-4ea5-9ca8-809210913ccc" />

In the Databricks Free Edition, there are some free tables that can be accessed right away. In UC, tables are organized in the following structure: `database`.`schema`.`table`. For example, `samples`.`bakehouse`.`sales_customers`. Alternatively, data can easily be uploaded to a database and schema of choice.

<img width="1910" height="998" alt="image" src="https://github.com/user-attachments/assets/776cb136-c0a2-4d87-90c6-832c166b25ff" />

UC tables can be directly queries within a notebook using SQL or (Py)Spark queries:

<img width="2084" height="1204" alt="image" src="https://github.com/user-attachments/assets/f8c108b1-bb8d-4bb5-9db7-9bbfb7a0a3ec" />

# Basics of PySpark

PySpark is the Python API for Spark. Tables from UC can be accessed by referencing the table and storing it as a Spark DataFrame:

<img width="2082" height="742" alt="image" src="https://github.com/user-attachments/assets/d037aefa-43b0-4d46-96f8-71f8bdfa94ec" />

Once a Spark DataFrame has been created, various PySpark functions can be used to execute SQL code, compute statistics, fit models, etc. For example, we can count the number of customers in each continent using SQL code, executed by PySpark, using `spark.sql()`

<img width="2088" height="698" alt="image" src="https://github.com/user-attachments/assets/2e813b38-5f33-4a0c-a7bd-4e477247422e" />

or using the PySpark API directly

<img width="2080" height="486" alt="image" src="https://github.com/user-attachments/assets/943ebb2d-2430-493b-942d-fe6d0b68d0cc" />

## Common PySpark Functions

Some common PySpark functions I use on a daily basis to manipulate Spark DataFrames (`df`) include:
* `df.count()`: returns the number of rows in `df`
* `df.show()`: prints `df`
* `df.display()`: displays `df` in an interactive table
* `df.filter()`: filter rows of `df` using SQL or PySpark syntax.
  * `df.filter("quantity >= 15 and paymentMethod is not null")`
  * `df.filter((col("quantity") >= 15) & (col("paymentMethod").isNotNull()))`
  * although the syntax is different, the results are equivalent
* `df.select()`: select columns to keep in `df`
* `df.drop()`: choose columns to drop from `df`
* `df.join()`: join `df` to another `df` on one or more columns. Allows for various types of joins (e.g., left, right, inner, outer, anti)
* `df.orderBy()`: order values by one or more columns
* `df.dropDuplicates()`: remove all duplicate rows (or rows with duplicate values of specified columns)
* `df.dropna()`: remove all rows with missing values
* `df.withColumnsRenamed()`: rename columns in `df` using a dictionary following `{'old_name': 'new_name'}` syntax. A version for working with one column at a time `df.withColumnRenamed()` also exists.
* `df.withColumns()`: Create/modify columns of `df` using a dictionary with specified columns and operations. A version for working with one column at a time `df.withColumn()` also exists.
  * `df.withColumns{'price_squared': col("totalPrice")**2, 'price_sqroot': sqrt(col("totalPrice")}`
* `df.groupBy()`: groups `df` by one or more columns to perform aggregation (i.e., compute min, max, or other statistics)
  * `df.groupBy("customerID").agg({"totalPrice": "sum"}).withColumnRenamed('sum(totalPrice)', 'overall_totalPrice')`
 
## Demonstrating PySpark on `Bakehouse` Data

In the Databricks Free Edition, there are free `bakehouse` tables available in UC that contain customer, transaction, and franchise data. At least in my verion of those tables, there are errors in the ID columns, and those errors are resolved by downloading the free "Cookies Dataset DAIS 2024" dataset that can be found by searching in the "Marketplace" in Databricks.

In the code below, I access the tables, do some filtering and joins, and perform some aggregation to determine the number of transactions, average quantity per transaction, and total amount spent by a subset of customers.

<img width="2206" height="1228" alt="image" src="https://github.com/user-attachments/assets/5a32c39f-70af-4900-98b1-400ffbc6deb2" />

<img width="2204" height="1270" alt="image" src="https://github.com/user-attachments/assets/39f7180c-3338-405d-a2e3-52d04fbf3356" />

<img width="2144" height="884" alt="image" src="https://github.com/user-attachments/assets/c6b36194-e005-46ce-9315-932b240c75de" />

## Conclusion

As you can see, working in Databricks is straightforward, and using PySpark to work with UC is seamless. The benefits are even larger for big datasets that do not fit in memory. Databricks has even more great built-in features that will be explored in a future blog post.
