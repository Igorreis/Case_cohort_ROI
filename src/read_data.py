import sqlite3
import pandas as pd

"""
This file is dedicated to host the function loan_tables.
As the name sugests, this function is dedicated to load the .db database and return the datasets.
"""

def load_tables(db_path):

    """
    Function dedicated to loan the database tables and return them as pandas DataFrames with the exact same columns 
    
    *Input:
        - db_path: path of the database.db file

    -> It is important to know that we modify a little bit of the tables to make sure we corrent eventual erros in the data

    -> LOANS: In loans we only keep the loans created after the batch was allowed (loans[loans['created_at'] >= loans['allowlisted_date']])
        - This choice could be changed, and instead of removing the loans, we could set their creation date to the 
            same date of when they were allowed.

    - REPAYMENTS: In repayments we first make sure that all of the loan ids in the dataframe are also in the loans dataframe
        - This ensures that we will consider a valid loan id in our analysis
        - We also check if each individual loan payment date happens after or before the loan creation
            - If it happens before, we move the payment to happen in the same day as the loan was created

    *Output:
        - allowlist: dataframe with columns 'user_id' 'batch' and 'allowlisted_date'
        - loans: dataframe with columns 'loan_id', 'user_id', 'created_at', 'updated_at', 'annual_interest', 'loan_amount', 'status'
        - repayments: dataframe with columns 'date', 'loan_id', 'repayment_amount', 'billings_amount'

    """

    con = sqlite3.connect(db_path)
    allowlist = pd.read_sql_query("SELECT * FROM allowlist", con, parse_dates=['allowlisted_date'])
    loans = pd.read_sql_query("SELECT * FROM loans", con, parse_dates=['created_at','updated_at'])
    repayments = pd.read_sql_query("SELECT * FROM loans_repayments", con, parse_dates=['date'])
    con.close()

    loans['allowlisted_date'] = loans['user_id'].map(allowlist.set_index('user_id')['allowlisted_date'])
    loans = loans[loans['created_at'] >= loans['allowlisted_date']]
    loans = loans.drop('allowlisted_date', axis = 1)

    valid_ids = pd.Index(loans['loan_id'].dropna().unique())
    mask = repayments['loan_id'].isin(valid_ids)

    repayments = repayments.loc[mask].copy()
    repayments['created_at'] = repayments['loan_id'].map(loans.drop_duplicates('loan_id').set_index('loan_id')['created_at'])
    repayments['days'] = repayments['date'] - repayments['created_at']
    repayments.loc[repayments['days'] == repayments['days'].unique().min(), 'date'] = repayments['created_at']
    repayments = repayments.drop(['created_at', 'days'], axis = 1)
    
    return allowlist, loans, repayments