# Enhancing, Extending, and Analyzing ReFED's Capital Flows Tracker

## Overview 
[ReFED](https://www.refed.org/) is a U.S. non-profit "working to catalyze the food system to take evidence-based action to stop wasting food." One of the services they provide is their Capital Tracker, which lets users quantify, analyze, and visualize major investments that have been made in the food waste reduction space, ranging from venture capital funding to mergers/acquistions to philanthropic gifts to food waste nonprofits.

This project seeks to enhance ReFED's existing capital flows dataset by doing the following: 
1. Scraping the ReFED Capital Tracker from ReFED's website
2. Developing and tuning a NLP model to predict the "Solution" category for investments where this category is missing
3. Plotting investment size against the "Solution" category (the original one where available, or the predicted where none was originally listed)
4. Creating a semantic similarity engine that identifies similar Capital Flows, by integrating with various news APIs

## Results
The supervised NLP model generates predictions of Solution category for ~1,800 investments that were previously missing this field (classifier accuracy: 92.4%; macro-averaged per-class F1 score: 0.79). The enhanced dataset containing predicted solution classes is [here](https://github.com/jakejsmith/ReFED-capital/blob/main/refed_full.csv). The visualization resulting from that data can be found [here](https://jakejsmith.github.io/refed_capital_tracker.html). 

The ranked list of news stories identified by the semantic similarity engine can be found [here](https://github.com/jakejsmith/ReFED-capital/blob/main/scores.csv). While this engine is very much in the "proof of concept" stage, it did successfully identify several articles that may be eligible for inclusion in the Capital Flows Tracker:
- [Papa Johns Canada to Present $73,411 Donation to Second Harvest](https://financialpost.com/globe-newswire/media-advisory-papa-johns-canada-to-present-73411-donation-to-second-harvest)
- [Safe and Green Development Corporation Executes Agreement for Strategic Shift via Acquisition of Resource Group a Next-Generation Engineered Soils and Composting Company](https://www.prnewswire.com/news-releases/mazda-foundation-usa-inc-awards-grants-to-focus-on-hunger-relief-stem-and-workforce-development-in-underserved-communities-across-the-us-302385679.html)

## Methodology
The code used to generate this project can be found [here](https://github.com/jakejsmith/ReFED-capital/blob/main/refed_final.py) (Python script) or in [here](https://github.com/jakejsmith/ReFED-capital/blob/main/ReFED%20Final.ipynb) (Jupyter notebook).
