# ROI Prediction Case Study

This repository contains the implementation and analysis for a case study focused on predicting the **Return on Investment (ROI)** of cohorts (batches) of loans, using information from their individual loans.

---

## 📂 Project Structure

```
IgorReis_Case/
├── database.db                             # Provided database with allowlist, loans and repayments
├── final_report.md                         # Full case study report
├── model_implementation.py                 # Core model implementation
├── Best_result_scenario1_model2.png        # Best result plot of batch ROI prediction vs batch ROI true
├── src                                     # Folder to store the definition of different functions used in the analysis
│   ├── eda_plots.py                        # Exploratory data analysis helper functions
│   ├── frame_creation.py                   # Functions related to the creation of dataframes
│   ├── read_data.py                        # Function on how to read the database tables
│   ├── run_models.py                       # Functions designed to run the models and return their results
│   ├── scenario1_modeling_functions.py     # Functions that define the Scenario 1 models
│   ├── scenario2_modeling_functions.py     # Functions that define the Scenario 2 models
├── notebooks/                              # Exploratory analysis and modeling notebooks
│   ├── EDA.ipynb                           # Exploratory data analysis
│   ├── scenario1_modeling.ipynb            # Detailed training of scenario 1
│   ├── scenario2_modeling.ipynb            # Detailed training of scenario 2
│   └── Figures/                            # Plots and figures generated
```

---

## 🚀 Usage

1. **Exploratory Data Analysis**  
   Open `notebooks/EDA.ipynb` to reproduce plots and initial data exploration.

2. **Model Training**  
   - Use `notebooks/scenario1_modeling.ipynb` and `notebooks/scenario2_modeling.ipynb` for detailed training.  
   - Alternatively, run the script:
     ```bash
     python model_implementation.py
     ```

3. **Database**  
   The `database.db` file includes three tables:
   - `allowlist`
   - `loans`
   - `repayments`

---

## 📊 Results

- **Best Model**: XGBoost -- Scenario 1, Model 2 with filtered dataset (90% upper percentile of loan amount removed)  
- **Main Metric**: RMSE = 0.007775  
- Example visualization:  

![Best Result](Best_result_scenario1_model2.png)

---

## 📝 Report

For methodology, findings, and conclusions, see [final_report.md](final_report.md).
