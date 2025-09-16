# Final Report

This case study had the objective of building a model that could predict the return on investment ROI of a cohort (batch) from the information of its individual loans.

To do this, I was given a database with three distinct tables: allowlist, loans and repayments:
    - allowlist: table containing information on the creation of the cohorts
        - Relates the cohort, the user_id and the cohort creation/allowlisted date
    - loans: table containing information of a quantity of different loans
        - Relates loan_id, user_id, loan_amount, loan creation date, date of status update...
    - repayments: table with loan repayment and billing information
        - Relates loan_id, repayment and billing dates, repayment and billing amounts...

I started to tackle this case with the EDA. I defined functions to merge all of the relevant information in these tables and avaluated the data through plots of the ROI evolution and correlation between parameters. Another fundamental role of the EDA was the definition of the decision time t and horizon H.
---
### EDA

From the EDA I concluded that:
    - The strongest correlation in the date can be found between inflow and outflow amounts of both loans and cohorts
    - There is also smaller correlation between the inflow and outlfow features with the length and age of the loans
    - From the EDA it was also possible to make an estimate for the decision time and horizon H
        - These quantities also have a dependency on the business model and goals.
        - Only through a data driven viewpoint, I chose t = 200 days and H = 730 days.
        - From the data, t = 200 seemed to be a good decision time, and it is still early enough in most batches lifecycle that measures can still be taken
        - H = 730 days (two years) also seemed reasonable, since we need to get enough time for the cohorts to become profitable. They are not a short term (less than a year) profitable business. They are a long term business deal.
        - All batches have some outlier very large loans. The influence of these loans on the overal cohort behaviour and evolution
            - The removal of the 90% upper percentile of loan amount did not change the correlation behaviour of our model features
            - However, the removal of these outlier loans led to the early turnover to positive cohort ROI
    
You can check a few plots from the EDA on the EDA.ipynb notebook

### Modeling

I tested four different models, separated in two different scenarios. However, all four models are XBGRegressor based.

Both scenarios aim to predict ROI(H) of our cohorts. Scenario 1 only consider loans up to the decision time t when computing the true ROI(H) of the cohorts. On the other hand, Scenario 2 considers all loans up to the horizon H in the computation of the true ROI(H) of the cohorts. This leads to a fundamental change in interpretation of the models.

I personally think that Scenario 1 is the one that makes the most amount of sense. Of course, it only takes into account loans up to the decision time t, and information of new loans and payments between t and H is "lost". However, the trend of the ROI up to t can be extrapolated to H, given that t is chosen with a significative value.

You can see more details on each Scenario and the Models in the notebooks scenario1_modeling.ipynb and scenario2_modeling.ipynb

### Conclusion

Analyzing the data in both scenarions proposed here, with our four different models, we can come to the conclusion that the best approach to predict a cohort ROI(H) is to use the Model 2 from Scenario 1, and to remove the loan amount outliers from the model input data.

This model led to predictions of the cohort ROI(H = 730) with RMSE \~ 0.007.

