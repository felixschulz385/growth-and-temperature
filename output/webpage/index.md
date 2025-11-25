---
layout: default
title: Home
---

# GNT Analysis Results

This page displays results from the Growth and Temperature analysis pipeline.

## Regression Analysis Results

{% include table_main.html %}

## About

This analysis uses the DuckReg package for efficient fixed effects regression with large datasets. All models employ Mundlak fixed effects estimation with bootstrapped standard errors.

---

*Last updated: {{ site.time | date: "%B %d, %Y" }}*
