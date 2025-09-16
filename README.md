# Return on Investment of cohort: case study

This is a case project with the objective to build a model capable of predicting the Return on Investment (ROI) of a cohort. To do this, I have access to a database with three different tables, cointaning information on the cohort creation, loans and repayments

### Basic structure of this repository

This repository is structured as such:

  - The EDA of the data can be accessed in noteooks/EDA.ipynb
  - Test of the models can be seen in notebooks/scenario1_modeling.ipynb and notebooks/scenario2_modeling.ipynb. 
  - A few figures are saved inside the folder notebooks/Figures.
  - The folder src contaings the different functions used throughout this project
  - model_implementation.py is the main python file in this repository

Running model_implementation.py will return to you which is the best model tested here, and the predictions of ROI, with 95% CI, for each cohort.
