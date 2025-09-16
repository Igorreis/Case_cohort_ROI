import pandas as pd
import numpy as np

"""
This file is dedicated to host functions required to create dataframes used in the EDA and in the model..
"""


def loan_cum_frame(allowlist, loans, repayments):

    """
    This function is responsible to merge the important information from the database tables into a single DataFrame.
    It stores all the information required and create new features from the data.

    *Inputs:
        - allowlist: allowlist table from the database
        - loans: loans table from the database
        - repayments: repayments table from the database

    *Outputs:
        - CF: this function outputs a dataframe with all the important information the database tables + new engineered features
    """

    #-------------- Here we start to merge the information from the three database tables
    L = loans.merge(allowlist[['user_id','batch']], on='user_id', how='left')
    L = L.drop(['status', 'updated_at'], axis = 1).drop_duplicates(keep = 'first')
    L['date'] = L['created_at']
    outflows = L.groupby(['batch', 'loan_id', 'date'], as_index=False)['loan_amount'].sum().rename(columns={'loan_amount':'outflow', 'created_at':'created_at'})

    R = repayments.merge(
    L[['loan_id','batch','date']],
    on='loan_id',
    how='outer',
    suffixes=('_rep', '_loan')
)

    # New date column = date from repayments, otherwise date from L
    R['date'] = R['date_rep'].combine_first(R['date_loan'])

    # Filling absent values (if loan only exists in L -> no repayment made yet)
    for col in ['repayment_amount', 'billings_amount']:
        if col in R.columns:
            R[col] = R[col].fillna(0.0)

    # Discarding auxiliary columns
    R = (R
        .drop(columns=['date_rep','date_loan'])
        .sort_values(['loan_id','date'])
        .reset_index(drop=True))

    #--------------- Inflows dataframe created per batch, date and loan_id as the sum of the repayment and billing amounts
    #--------------- This basically makes it so that we sum all payments made to a given loan in a day
    inflows = R.groupby(['batch','date', 'loan_id'], as_index=False)[['repayment_amount','billings_amount']].sum().rename(
        columns={'repayment_amount':'inflow','billings_amount':'billing'})
    
    #--------------- Here we create CF and start filling its new columns
    CF = outflows.merge(inflows, on=['batch', 'loan_id', 'date'], how='outer').fillna(0)
    CF = CF.sort_values(['loan_id', 'date'])
    CF['annual_interest'] = CF['loan_id'].map(L.drop_duplicates('loan_id').set_index('loan_id')['annual_interest'])

    CF['cum_in']   = CF.groupby('loan_id')['inflow'].cumsum() # This is the cumulative sum (per day) of the repayments of a given loan
    CF['cum_bill'] = CF.groupby('loan_id')['billing'].cumsum() # This is the cumulative sum (per day) of the billings of a given loan

    loan_amount_map = L.drop_duplicates('loan_id').set_index('loan_id')['loan_amount']
    CF['out'] = CF['loan_id'].map(loan_amount_map)

    CF['days_step'] = CF.groupby(['loan_id'])['date'].diff().dt.days.fillna(0) # This counts how many days have gone by from one payment to the next (per loan)

    #----------------- The next step requires some explaining:
    #----------------- In the dataset we have the annual interest rate (%) or each loan. Since I don't have information on the business rules
    #----------------- I decided to apply daily interests to the loan based on the initial loan_amount. Here'carry' is the amount of interest
    #----------------- that is computed at each line of the dataframe.
    #----------------- In a real case scenario, the interest would have a pre-definite rule on how it is applied. Once the rule is defined, it can
    #----------------- be applied bellow.
    CF['carry'] = CF['out'] * (1 + CF['annual_interest']/100/365)**CF['days_step'] - CF['out']
    CF['carry_cum'] = (CF['carry'].groupby(CF['loan_id']).cumsum())

    CF['outflow'] = CF['outflow'] + CF['carry'] # outflow represents how much money the user owes per line (date) in the dataframe
    CF['out'] = CF['out'] + CF['carry_cum'] # This column gives the total amount of money the user owes (loan amount + interest)

    CF['roi_cum'] = (CF['cum_in'] + CF['cum_bill'] - CF['out'])/CF['out'] # Here we compute the daily ROI of each loan
    CF['roi_cum'] = CF['roi_cum'].fillna(-1)

    CF['net_in'] = CF['cum_in'] + CF['cum_bill'] # Net inflow (repayments + billings)
    CF['net_cash'] = CF['cum_in'] + CF['cum_bill'] - CF['out'] # Net cash (Inflow - out) -> how much was paid - how much was lent

    CF['allowlisted_date'] = CF['batch'].map(allowlist.drop_duplicates('batch').set_index('batch')['allowlisted_date']) # Including allowlisted date
    CF['days_since_allowed'] = (CF['date'] - CF['allowlisted_date']).dt.days # Computing how long has it been since the batch was allowed
    days_per_loan = CF.groupby('loan_id')['date'].agg(lambda x: (x.max() - x.min()).days + 1) # Computing how long each loan was

    CF['days_per_loan'] = CF['loan_id'].map(days_per_loan) # Creating the days_per_loan column

    #----------------- Now we wish to compute the ROI per batch per day
    #----------------- The column will be eventually called roi_cum_weighted, since it is computed as a weighted average
    #----------------- of the individual loan ROIs, where the weight is the value of 'out'
    tmp = (CF[['batch','loan_id','date','roi_cum','out']].copy().sort_values(['batch','loan_id','date'])) # Temp working dataframe

    #----------------- Here we compute the ROI variation per loan
    tmp['roi_prev']  = tmp.groupby(['batch','loan_id'])['roi_cum'].shift()
    tmp['delta_roi'] = tmp['roi_cum'] - tmp['roi_prev'].fillna(0.0)

    #----------------- We take the first date of each loan
    first_dt = tmp.groupby(['batch','loan_id'], as_index=False)['date'].min().rename(columns={'date':'first_date'})
    tmp = tmp.merge(first_dt, on=['batch','loan_id'], how='left')

    #----------------- Computing daily contributions. Note the use of the weight tmp['out']
    tmp['num_delta'] = tmp['delta_roi'] * tmp['out'] # Variation ROI*out
    tmp['den_delta'] = tmp['out']

    agg_w = (tmp.groupby(['batch','date'], as_index=False)
                .agg(num_delta=('num_delta','sum'),
                    den_delta=('den_delta','sum'))
                .sort_values(['batch','date']))

    #----------------- Cumulative sum
    agg_w['num_cum'] = agg_w.groupby('batch')['num_delta'].cumsum()
    agg_w['den_cum'] = agg_w.groupby('batch')['den_delta'].cumsum()

    #----------------- Final step
    agg_w['roi_cum_weighted'] = agg_w['num_cum'] / agg_w['den_cum'].replace(0, np.nan)
    # opcional: sentinel -1 antes de qualquer out (para comparar com CF)
    agg_w['roi_cum_weighted'] = agg_w['roi_cum_weighted'].fillna(-1)

    CF = CF.merge(
        agg_w[['batch','date','roi_cum_weighted']],
        on=['batch','date'],
        how='left'
    )
    CF = CF.rename(columns={'roi_cum_weighted' :'roi_cum_batch_weighted'})

    return CF

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

def rebuild_CF_from_loanCF(loan_CF):

    """
    Function dedicated to construct a batch level dataframe from the loan level dataframe

    *Inputs: 
        - loan_CF: loan level dataframe

    *Outputs:
        - tmp: batch level dataframe
    """

    df = loan_CF.copy()

    #----------------- Grouping by batch and suming the necessary quantities
    tmp = (df.groupby(['batch','date'], as_index=False)
             .agg(inflow=('inflow','sum'),
                  outflow=('outflow','sum'),
                  billing=('billing','sum')))
    tmp = tmp.sort_values(['batch','date'], kind='mergesort')

    #----------------- Computing cumulative quantities of the batch
    tmp['cum_in']   = tmp.groupby('batch', sort=False)['inflow'].cumsum()
    tmp['cum_out']  = tmp.groupby('batch', sort=False)['outflow'].cumsum()
    tmp['cum_bill'] = tmp.groupby('batch', sort=False)['billing'].cumsum()
    tmp['net_cash'] = tmp['cum_in'] + tmp['cum_bill'] - tmp['cum_out']

    #----------------- Computing ROI
    den = tmp['cum_out'].replace(0, np.nan)
    tmp['roi_cum'] = (tmp['cum_in'] + tmp['cum_bill'] - tmp['cum_out']) / den
    tmp['roi_cum'] = tmp['roi_cum'].fillna(-1.0)

    allow_map = (df.drop_duplicates('batch')[['batch','allowlisted_date']]
                   .set_index('batch')['allowlisted_date'])
    tmp['allowlisted_date'] = tmp['batch'].map(allow_map)

    # dias desde allow por linha (batch, date)
    tmp['days_since_allowlisted'] = (tmp['date'] - tmp['allowlisted_date']).dt.days.astype('Int64')

    return tmp

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------


def day_when_loans_positive(percent, loan_CF_batch,time_col):

    """
    Function dedicated to return the first day (from allowlisted data of each batch) for which a percentage p
    of loans are paid (net cash >0)

    *Inputs:
        - percent: percentage of loans with positive net cash
        - loan_CF_batch: loan level dataframe for a specific batch
        - time_col: name of the column with the allowlisted dates (should be set as 'allowlisted_date')

    *Outputs:
        - dictionay with a some quantities
            -> half_day
            -> half_day: number of days since the batch was allowed for which 'percent' of loans were paid
            -> half_date: date corresponding to half day
            -> prop_by_day: Series with the cumulative proportion of paid loans
            -> n_loans: total number of loans 
            -> n_pos_ever: number of loans that were ever repaid
    """
    df = loan_CF_batch.copy()

    #----------------- Check if the correct allowed date column name was inputed
    if time_col != 'days_since_allowed':
        raise ValueError("Provide time_col = 'days_since_allowed' and make sure that it is also present in the DF.")


    df['net_cash'] = pd.to_numeric(df['net_cash'], errors='coerce').fillna(0.0)
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce').astype('Int64')

    #----------------- Finding first day with net cash >0
    pos = df.loc[df['net_cash'] > 0, ['loan_id', time_col]]
    first_pos = pos.groupby('loan_id', as_index=False)[time_col].min() \
                   .rename(columns={time_col: 'first_positive_day'})


    loans = df['loan_id'].drop_duplicates()
    n_loans = len(loans)

    #----------------- Marking loans that never got net cash >0
    first_pos_full = loans.to_frame().merge(first_pos, on='loan_id', how='left')
    # first_positive_day = NaN => nunca ficou positivo

    #----------------- Cumulative proportion
    counts = (first_pos_full
              .dropna(subset=['first_positive_day'])
              .groupby('first_positive_day').size()
              .sort_index())

    if counts.empty:
        # ninguém ficou positivo
        return {
            'half_day': None,
            'half_date': None,
            'prop_by_day': pd.Series(dtype=float),
            'n_loans': n_loans,
            'n_pos_ever': 0
        }

    cum_pos = counts.cumsum()
    prop_by_day = (cum_pos / n_loans).rename('prop_positive')

    #----------------- Comparing the proportion of loand that were paid 
    reached = prop_by_day[prop_by_day >= percent]
    if reached.empty:
        half_day = None
        half_date = None
    else:
        half_day = int(reached.index[0])
        # tentar inferir a data do batch (allowlisted_date + half_day)
        if 'allowlisted_date' in df.columns and df['allowlisted_date'].notna().any():
            # para batch único, a allowlisted_date costuma ser igual dentro do batch (ou o mínimo do batch)
            allow_date = pd.to_datetime(df['allowlisted_date']).min()
            half_date = (allow_date + pd.to_timedelta(half_day, unit='D')).date()
        else:
            half_date = None

    return {
        'half_day': half_day,
        'half_date': half_date,
        'prop_by_day': prop_by_day,   # Series index=day, values=proporção acumulada
        'n_loans': n_loans,
        'n_pos_ever': int(first_pos_full['first_positive_day'].notna().sum())
    }


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------


def remove_top_loans_per_batch(df: pd.DataFrame,
                               metric_col: str = 'out',
                               p: float = 0.90,
                               min_loans_per_batch: int = 10,
                               strict: bool = True):
    """
    This function is dedicated to remove loans with outlier values for each batch.
    That is, we choose a percentile p, and remove all loans with loan amount larger than this percentile.
    
    *Inputs:
      - metric_col: column that represents the amount of the loan
      - p: percentile (0.9 = 90%)
      - min_loans_per_batch: minimum number of loans required 
      - strict: if True, we remove loans '>' (above percentile p); if False we remove '>=' (equal of abote percentile p)
      
    *Outputs:
      - df_filtered: new dataframe with loans above p removed
      - summary: dataframe with information of how many loans were removed for each batch
      - thresholds: dataframe with the p threshold of loan amount
    """
    df = df.copy()


    #----------------- Taking loan size (amount)
    loan_size = (df.groupby(['batch', 'loan_id'], as_index=False)[metric_col]
                   .min()
                   .rename(columns={metric_col: 'loan_size'}))


    #----------------- Counting per batch and applying percentile p
    counts = loan_size.groupby('batch')['loan_id'].size().rename('n_loans').reset_index()
    thresholds = (loan_size.merge(counts, on='batch', how='left')
                            .query('n_loans >= @min_loans_per_batch')
                            .groupby('batch')['loan_size']
                            .quantile(p)
                            .rename('thr')
                            .reset_index())

    #----------------- Marking loans to be removed
    loan_size = loan_size.merge(thresholds, on='batch', how='left')
    if strict:
        mask_top = loan_size['loan_size'] > loan_size['thr']
    else:
        mask_top = loan_size['loan_size'] >= loan_size['thr']
    mask_top = mask_top.fillna(False)

    bad_ids = set(loan_size.loc[mask_top, 'loan_id'])

    #----------------- Removing loans
    df_filtered = df[~df['loan_id'].isin(bad_ids)].copy()

    #----------------- Creating summary per batch
    removed = (loan_size.assign(removed=mask_top.astype(int))
                         .groupby('batch')['removed']
                         .sum()
                         .rename('n_removed')
                         .reset_index())
    summary = counts.merge(removed, on='batch', how='left').fillna({'n_removed': 0})
    summary['n_removed'] = summary['n_removed'].astype(int)
    summary['pct_removed'] = np.where(summary['n_loans'] > 0,
                                      summary['n_removed'] / summary['n_loans'],
                                      0.0)

    return df_filtered, summary.sort_values('batch'), thresholds.sort_values('batch')