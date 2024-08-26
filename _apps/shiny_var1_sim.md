---
layout: archive
title: "Control Charts for VAR(1) Simulated Data"
permalink: /shiny_apps/var1_sim
collection: apps
classes: wide
---

This shiny app displays simulated data and fitted classical and robust control charts for VAR(1) data, illustrating the effects of contamination during Phase I on control chart performance. It can be viewed in full-screen at <a href="https://taylor-grimm.shinyapps.io/var_shiny/">this link</a>.

Within the app, users have the ability to change the:
* number of Phase I observations (Sample Size)
* number of variables (Dimension)
* percentage of contamination in the Phase I data (Contamination %). Contamination shows up as blue points in the plots.
* size of the shift in Phase II (Shift Size), which is the same shift size applied to contaminate the Phase I data.

Here, the shift in Phase II (black lines) begins immediately after Phase I (gray lines) ends.

The performance of the classical $T^2$ and multivariate exponentially weighted moving average (MEWMA) charts is shown, along with robust $T^2$ and MEWMA charts that use the reweighted minimum covariance determinant (RMCD) estimators to handle contamination.

In general, the methods perform similarly when no contamination is present, but the robust methods perform better when contamination is present. Also, MEWMA charts are better at detecting small shifts than $T^2$ charts.

More details on the $T^2$ and MEWMA control charts and how they work can be found in <a href="https://trgrimm.github.io/posts/2024/03/t2_mewma/">this</a> blog post.

<embed src="https://taylor-grimm.shinyapps.io/var_shiny/" style="width:100%; height: 50vw;">
