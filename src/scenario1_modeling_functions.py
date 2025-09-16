import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.read_data import *
from src.frame_creation import *
from src.eda_plots import *
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, balanced_accuracy_score
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

"""
This file is dedicated to host the functions used in the modeling scenario 1.
This scenario has two different models:
    1: the first model is constructed such as it tries to predict each individual loan ROI(H), using its data at
        a decision time t
        - Later, the individual loans ROI(H) predicted by the model are agregated to compute ROI(H) of their
        respective batches
        - The true batch ROI(H) is computed using aonly loans that were created up to allowlisted_date + t!!!
    2: the second model is constructed such as it tries to predict each individual loan net cash (H), using its
        data at decision time t
        - Lates we aggregate the net cash predicted for all loans to compute the ROI(H) of their respective batches
        - Again, the  true batch ROI(H) is computed using only loans that were created up to allowlisted_date + t!!!
"""

def sort_final(eval_, batches):
    eval_.batch = eval_.batch.astype('category')

    eval_.batch = eval_.batch.cat.set_categories(batches)

    return eval_.sort_values('batch')


def wbootstrap_mean_ci(roi, w, B=2000, alpha=0.05, seed=123):
    """
    Function to compute the batch ROI from the weighted average of the loans ROIs.
    The weight used here is the value of the loan 'out' column in our dataframe.
    We also compute CI using a bootstrap method.
    
    *Inputs:
        - roi: loan roi data
        - w: weights
        - B: bootstrap parameter
        - alpha: determines the CI
        - seed: random seed

    *Outputs:
        - mu: mean value
        - lo: lower CI bound
        - hi: upper CI bound
    """

    roi = np.asarray(roi, dtype=float)
    w   = np.asarray(w,   dtype=float)
    m = np.isfinite(roi) & np.isfinite(w) & (w > 0)

    roi = roi[m]; w = w[m]

    if roi.size == 0:
        return np.nan, np.nan, np.nan
    
    p = w / w.sum()
    mu = float(np.sum(p * roi))

    if roi.size == 1:
        return mu, mu, mu
    
    rng = np.random.default_rng(seed)
    n = roi.size
    boots = np.empty(B, dtype=float)

    for b in range(B):
        idx = rng.choice(n, size=n, replace=True, p=p)
        boots[b] = roi[idx].mean()   # média simples do reamostrado; p embute o peso

    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])

    return mu, float(lo), float(hi)

def wavg(y, w):

    y = np.asarray(y); w = np.asarray(w)
    m = np.isfinite(y) & np.isfinite(w) & (w > 0)
    return np.average(y[m], weights=w[m]) if m.any() else np.nan


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
############################################# MODEL 1 ################################################################
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------


def scenario1_snap_roi_func(df_, H, t):
    
    """
    This function is dedicated to produce a snapshot at time t for all loans.
    We also compute new columns for each loan based on its values 7 days before t and 30 days before t

    *Inputs:
        - df_: out data dataframe
        - H: horizon. This quantity is given in days. We wish to predict the batch return H days after it was allowed
        - t: decision time. This quantity is given in days. Repersents how many days of data (after the batch was allowed)
            our model will use

    *Outputs:
        - snap: snapshot dataframe assuming decision time t and horizon H
    """

    df = df_.copy()
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df['allowlisted_date'] = pd.to_datetime(df['allowlisted_date']).dt.tz_localize(None)

    #------------------- Daily events per loan

    events = (df.groupby(['loan_id','batch','date'], as_index=False)
                .agg(inflow=('inflow','sum'),
                    billing=('billing','sum'),
                    outflow=('outflow','sum'),
                    cum_in=('cum_in','max'),
                    cum_bill=('cum_bill','max'),
                    roi_cum=('roi_cum','max'),
                    out=('out','max')))
    events = events.sort_values(['loan_id','date'], kind='mergesort').reset_index(drop=True)
    events['cum_outflow'] = events.groupby('loan_id', sort=False)['outflow'].cumsum()

    #------------------- Preparing new columns to compute features at days t-7 and t-30

    base = (df.drop_duplicates('loan_id')[['loan_id','batch','allowlisted_date','out']].copy())
    base['t0']  = base['allowlisted_date'] + pd.to_timedelta(t,  unit='D')
    base['t7']  = base['t0']               - pd.to_timedelta(7,  unit='D')
    base['t30'] = base['t0']               - pd.to_timedelta(30, unit='D')
    base['tH']  = base['allowlisted_date'] + pd.to_timedelta(H, unit='D')

    first_date = (events.groupby('loan_id', as_index=False)['date']
                        .min().rename(columns={'date':'first_date'}))
    base = base.merge(first_date, on='loan_id', how='left')
    base = base[base['first_date'] <= base['t0']].reset_index(drop=True)

    #------------------- Selecting all the relevant rows and adding the new time columns

    EV_COLS = ['inflow','billing','outflow','cum_in','cum_bill','cum_outflow','roi_cum','out']

    def ts_to_i8(ts):
        return np.int64(pd.Timestamp(ts).value)

    rows = []
    base_idx = base.set_index('loan_id')
    for lid, g in events.groupby('loan_id', sort=False):
        if lid not in base_idx.index:
            continue
        b = base_idx.loc[lid]

        dates_i8 = g['date'].astype('datetime64[ns]').astype('int64').to_numpy()
        t0_i, t7_i, t30_i, tH_i = map(ts_to_i8, [b['t0'], b['t7'], b['t30'], b['tH']])

        def pos_leq(tv_i8):  return np.searchsorted(dates_i8, tv_i8, side='right') - 1
        p0, p7, p30, pH = pos_leq(t0_i), pos_leq(t7_i), pos_leq(t30_i), pos_leq(tH_i)

        rec = {
            'loan_id': lid, 'batch': b['batch'],
            'allowlisted_date': b['allowlisted_date'], 'out': b['out'],
            't0': b['t0'], 't7': b['t7'], 't30': b['t30'], 'tH': b['tH'],
            'date_t0':  pd.to_datetime(dates_i8[p0],  unit='ns') if p0  >= 0 else pd.NaT,
            'date_t7':  pd.to_datetime(dates_i8[p7],  unit='ns') if p7  >= 0 else pd.NaT,
            'date_t30': pd.to_datetime(dates_i8[p30], unit='ns') if p30 >= 0 else pd.NaT,
            'date_tH':  pd.to_datetime(dates_i8[pH],  unit='ns') if pH  >= 0 else pd.NaT,
        }
        for c in EV_COLS:
            arr = g[c].to_numpy()
            rec[f'{c}_t0']  = arr[p0]  if p0  >= 0 else np.nan
            rec[f'{c}_t7']  = arr[p7]  if p7  >= 0 else np.nan
            rec[f'{c}_t30'] = arr[p30] if p30 >= 0 else np.nan
            rec[f'{c}_tH']  = arr[pH]  if pH  >= 0 else np.nan

        rows.append(rec)

    snap = pd.DataFrame(rows)

    def diff0(a, b): 
        return snap[a].fillna(0.0) - snap[b].fillna(0.0)

    #------------------- Windows with differences between cumulative quantities

    snap['inflow_7']   = diff0('cum_in_t0',      'cum_in_t7')
    snap['billing_7']  = diff0('cum_bill_t0',    'cum_bill_t7')
    snap['outflow_7']  = diff0('cum_outflow_t0', 'cum_outflow_t7')
    snap['inflow_30']  = diff0('cum_in_t0',      'cum_in_t30')
    snap['billing_30'] = diff0('cum_bill_t0',    'cum_bill_t30')
    snap['outflow_30'] = diff0('cum_outflow_t0', 'cum_outflow_t30')

    snap['cum_in_t']   = snap['cum_in_t0'].fillna(0.0)
    snap['cum_bill_t'] = snap['cum_bill_t0'].fillna(0.0)
    snap['out_fix']    = snap['out'].fillna(0.0)
    snap['net_cash_t'] = snap['cum_in_t'] + snap['cum_bill_t'] - snap['out_fix']

    roi_t0  = snap['roi_cum_t0'].where(snap['roi_cum_t0'].notna(), -1.0)
    roi_t7  = snap['roi_cum_t7'].where(snap['roi_cum_t7'].notna(), -1.0)
    roi_t30 = snap['roi_cum_t30'].where(snap['roi_cum_t30'].notna(), -1.0)
    snap['roi_t']   = roi_t0
    snap['roi_d7']  = roi_t0 - roi_t7
    snap['roi_d30'] = roi_t0 - roi_t30

    snap['frac_paid_t']   = np.where(snap['out_fix']>0, snap['cum_in_t']/snap['out_fix'], 0.0)
    snap['frac_paid_d30'] = np.where(snap['out_fix']>0, diff0('cum_in_t0','cum_in_t30')/snap['out_fix'], 0.0)
    snap['pace_in_7']     = snap['inflow_7']  / 7.0
    snap['pace_in_30']    = snap['inflow_30'] / 30.0

    #------------------- Target

    snap['y_roiH'] = snap['roi_cum_tH'].where(snap['roi_cum_tH'].notna(), -1.0)

    return snap


def scenario1_data_split_model_train_roi(snap, RNG = 42):

    """
    This function is dedicated to train the scenario 1 model 1 (model that directly predicts the loan ROI)
    Here we are using a XBGRegression model.

    *Inputs:
        - snap: snapshot prepared in the previous function
        - RNG: seed parameter

    *Outputs:
        - snap: snapshot with model predicitons
        - reg: regression model trained
    """

    feat_cols = [
    'roi_t','cum_in_t','cum_bill_t','net_cash_t','frac_paid_t',
    'inflow_7','billing_7','outflow_7',
    'inflow_30','billing_30','outflow_30',
    'pace_in_7','pace_in_30','roi_d7','roi_d30','frac_paid_d30','out_fix'
    ]

    #------------------- Separating the data into X (model input) and y (model output/answer)

    X_all = snap[feat_cols].fillna(0.0).astype('float32')
    y_all = snap['y_roiH'].astype('float32')
    groups = snap['batch']

    #------------------- Out of fold per batch (Kfold group)

    K = min(5, groups.nunique())
    gkf = GroupKFold(n_splits=K)

    snap['y_pred_oof'] = np.nan
    all_zero_mask = (X_all.abs().sum(axis=1) == 0)

    #------------------- Training the model

    for fold, (tr, val) in enumerate(gkf.split(X_all, y_all, groups), 1):

        tr_use = np.array([i for i in tr if not all_zero_mask.iloc[i]])
        Xtr, ytr = X_all.iloc[tr_use].copy(), y_all.iloc[tr_use].copy()
        Xval     = X_all.iloc[val].copy()

        nun  = Xtr.nunique(dropna=False)
        keep = nun[nun > 1].index.tolist()
        if len(keep) == 0:
            continue

        reg = XGBRegressor(
            n_estimators=800, learning_rate=0.03, max_depth=6,
            subsample=0.9, colsample_bytree=0.8, min_child_weight=1.0,
            reg_lambda=1.0, objective='reg:squarederror',
            tree_method='hist', n_jobs=-1, random_state=RNG,
            eval_metric='rmse', verbosity=0
        )
        reg.fit(Xtr[keep], ytr)
        snap.loc[X_all.iloc[val].index, 'y_pred_oof'] = reg.predict(Xval[keep])
        
    return snap, reg


def scenario1_compute_metrics_roi(snap, RNG = 42, eps = 0.006):

    """
    This function is dedicated to save metrics of avaluation between the predicted ROI and true ROI
    Since the ROI can be very small, we also compute a metric to check if the model can correctly predict
    if the ROI is positive or negative based on a tolerance eps

    *Inputs:
        - snap: snapshot
        - RNG: random seed
        - eps: tolerance for ROI. if pred ROI > eps we consider it as positive. if ROI < eps we consider it negative
            -> In practice, a ROI < eps would mean a negligible return from the batch

    *Outputs:
        - mask_predcohort: mask representing the loans in the test dataset
        - eval_df_cohort: dataframe with true and predicted ROIs
        - metrics: metrics to compare true and predicted ROIs
    """

    #------------------- Creating data index variables
    train_idx = []
    rng = np.random.RandomState(RNG)
    for b, d in snap.groupby('batch', sort=False):
        n_tr = int(np.floor(0.8 * len(d)))
        train_idx += d.sample(n=n_tr, random_state=RNG).index.tolist()
    train_idx = np.array(sorted(set(train_idx)))
    test_idx  = np.array(sorted(set(snap.index) - set(train_idx)))

    #------------------- Computing loan-level metrics
    mask_pred = snap['y_pred_oof'].notna()
    rmse_loan = mean_squared_error(snap.loc[mask_pred,'y_roiH'],
                                snap.loc[mask_pred,'y_pred_oof'], squared=False)
    mae_loan  = mean_absolute_error(snap.loc[mask_pred,'y_roiH'],
                                    snap.loc[mask_pred,'y_pred_oof'])
    sign_true_loan = (snap.loc[mask_pred,'y_roiH'] > eps).astype(int)
    sign_pred_loan = (snap.loc[mask_pred,'y_pred_oof'] > eps).astype(int)
    acc_sign_loan  = (sign_true_loan == sign_pred_loan).mean()
    bacc_loan      = balanced_accuracy_score(sign_true_loan, sign_pred_loan) if sign_true_loan.nunique() > 1 else np.nan



    #------------------- Taking the test loans
    cohort = snap.loc[test_idx, ['batch','out_fix','y_roiH','y_pred_oof']].copy()

    batch_true_cohort = (cohort.groupby('batch')
                        .apply(lambda d: wavg(d['y_roiH'], d['out_fix']))
                        .rename('y_true_batch'))
    batch_pred_cohort = (cohort.groupby('batch')
                        .apply(lambda d: wavg(d['y_pred_oof'], d['out_fix']))
                        .rename('y_pred_batch'))

    eval_df_cohort = (pd.concat([batch_true_cohort, batch_pred_cohort], axis=1)
                    .dropna()
                    .reset_index())

    #------------------- Computing batch level metrics

    rmse_b = mean_squared_error(eval_df_cohort['y_true_batch'], eval_df_cohort['y_pred_batch'], squared=False)
    mae_b  = mean_absolute_error(eval_df_cohort['y_true_batch'],  eval_df_cohort['y_pred_batch'])
    sign_true_b = (eval_df_cohort['y_true_batch'] > eps).astype(int)
    sign_pred_b = (eval_df_cohort['y_pred_batch'] > eps).astype(int)
    acc_sign_b  = (sign_true_b == sign_pred_b).mean()
    bacc_b      = balanced_accuracy_score(sign_true_b, sign_pred_b) if sign_true_b.nunique() > 1 else np.nan

    #------------------- Here we print the metrics for loan and batch levels
    print(f"[Loan-level | ROI(H)] RMSE={rmse_loan:.6f} | MAE={mae_loan:.6f} | "
        f"SignACC={acc_sign_loan:.3f} | BACC={bacc_loan if bacc_loan==bacc_loan else float('nan'):.3f}")
    print('-'*50)
    print(f"[Batch-level | ROI(H)] RMSE={rmse_b:.6f} | MAE={mae_b:.6f} | "
        f"SignACC={acc_sign_b:.3f} | BACC={bacc_b if bacc_b==bacc_b else float('nan'):.3f}")

    metrics = {
        'rmse_loan': rmse_loan,
        'mae_loan': mae_loan,
        'acc_sign_loan': acc_sign_loan,
        'bacc_loan': bacc_loan,
        'rmse_b': rmse_b,
        'mae_b': mae_b,
        'acc_sign_b': acc_sign_b,
        'bacc_b': bacc_b,
        'comp_metric': rmse_b
        }

    return mask_pred, cohort, eval_df_cohort, metrics


def scenario1_plots_roi(scenario1_snap_roi, scenario1_mask_pred_roi, scenario1_cohort_roi, scenario1_eval_df_cohort_roi, scenario1_metrics_roi, savefig = True, filtered = '', RNG = 42):


    """
    This function is dedicated to plot our model results

    *Inputs:
        - scenario1_snap_roi: snapshot
        - scenario1_mask_pred_roi: mask with prediction locations in the dataframe
        - scenario1_cohort_roi: batch level dataframe
        - scenario1_eval_df_cohort_roi: dataframe with batch evaluation data
        - scenario1_metrics_roi: metrics dictionary
        - savefig: if True, the plots are saved to disk
        - filtered: in case you use a filtered dataframe, add a string here to modify the figure name
    """

    loan_eval = (
        scenario1_snap_roi.loc[
            scenario1_mask_pred_roi, ['loan_id','batch','y_roiH','y_pred_oof']
            ].rename(columns={'y_roiH':'y_true_loan','y_pred_oof':'y_pred_loan'})
            )
    
    #------------------- First plot (loan-level ROI)
    plt.figure(figsize=(8,6))
    i = 0
    for b, g in loan_eval.groupby('batch'):
        i += 1
        plt.scatter(g['y_true_loan']*100, g['y_pred_loan']*100, s=22, color=list(mcolors.TABLEAU_COLORS.keys())[i-1],
                    label=f'Batch {i}')
    plt.axvline(0, color='k', ls=':', lw=1); plt.axhline(0, color='k', ls=':', lw=1)
    plt.plot([-1000,1000], [-1000,1000], 'k--', lw=1)
    plt.xlim((-150,150)); plt.ylim((-150,150))
    plt.xlabel('ROI(H) true (\%)', fontsize = 12)
    plt.ylabel('ROI(H) pred (\%)', fontsize = 12)
    plt.title(f'Loan-level ROI -- RMSE = {scenario1_metrics_roi["rmse_loan"]:.6f}')
    plt.legend()
    plt.tight_layout()

    if savefig:
        plt.savefig('Figures/' + filtered+'scenario1_loan_roi.png', dpi = 300, format = 'png')

    plt.show()


    #------------------- Compute batch level CI
    rows = []
    for r in scenario1_eval_df_cohort_roi.itertuples(index=False):
        b = r.batch
        g =  scenario1_cohort_roi.loc[ scenario1_cohort_roi['batch'] == b, ['out_fix','y_roiH','y_pred_oof']].copy()

        mu_t, lo_t, hi_t = wbootstrap_mean_ci(g['y_roiH'],     g['out_fix'], B=2000, seed=RNG+123)
        mu_p, lo_p, hi_p = wbootstrap_mean_ci(g['y_pred_oof'], g['out_fix'], B=2000, seed=RNG+456)

        rows.append({
            'batch': b,
            'true_mu': mu_t, 'true_lo': lo_t, 'true_hi': hi_t,
            'pred_mu': mu_p, 'pred_lo': lo_p, 'pred_hi': hi_p
        })

    eval_roi_boot = pd.DataFrame(rows)

    #------------------- Second plot (batch-level ROI + error)
    plt.figure(figsize=(8,6))
    palette_keys = list(mcolors.TABLEAU_COLORS.keys())

    for i, r in enumerate(eval_roi_boot.itertuples(index=False), start=1):
        c = palette_keys[(i-1) % len(palette_keys)]
        x  = r.true_mu * 100.0
        y  = r.pred_mu * 100.0
        
        yerr = np.array([[ (r.pred_mu - r.pred_lo)*100.0 ],
                        [ (r.pred_hi - r.pred_mu)*100.0 ]])

        plt.errorbar(x=x, y=y, yerr=yerr,
                    fmt='o', ms=6, color=c, elinewidth=1.2, capsize=3, alpha=0.9,
                    label=f'Batch {i}')

    plt.axvline(0, color='k', ls=':', lw=1)
    plt.axhline(0, color='k', ls=':', lw=1)
    plt.plot([-1000,1000], [-1000,1000], 'k--', lw=1)

    plt.xlim((-22, 22))
    plt.ylim((-22, 22))

    plt.xlabel('ROI(H) true (\%)', fontsize = 12)
    plt.ylabel('ROI(H) predicted (\%)', fontsize = 12)
    plt.title(f'Batch-level ROI -- RMSE = {scenario1_metrics_roi["rmse_b"]:.6f}')

    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig('Figures/'+filtered+'scenario1_batch_roi_werror.png', dpi = 300, format = 'png')
    plt.show()



def scenario1_pre_err_roi(scenario1_cohort_roi, scenario1_eval_df_cohort_roi, RNG = 42):
    rows = []
    for r in scenario1_eval_df_cohort_roi.itertuples(index=False):
        b = r.batch
        g =  scenario1_cohort_roi.loc[ scenario1_cohort_roi['batch'] == b, ['out_fix','y_roiH','y_pred_oof']].copy()

        mu_t, lo_t, hi_t = wbootstrap_mean_ci(g['y_roiH'],     g['out_fix'], B=2000, seed=RNG+123)
        mu_p, lo_p, hi_p = wbootstrap_mean_ci(g['y_pred_oof'], g['out_fix'], B=2000, seed=RNG+456)

        rows.append({
            'batch': b,
            'true_mu': mu_t, 'true_lo': lo_t, 'true_hi': hi_t,
            'pred_mu': mu_p, 'pred_lo': lo_p, 'pred_hi': hi_p
        })

    eval_roi_boot = pd.DataFrame(rows)

    return eval_roi_boot

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
############################################# MODEL 2 ################################################################
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------



def scenario1_snap_cash_func(df_, H = 730, t = 200):

    """
    This function is dedicated to produce a snapshot at time t for all loans.
    We also compute new columns for each loan based on its values 7 days before t and 30 days before t

    *Inputs:
        - df_: out data dataframe
        - H: horizon. This quantity is given in days. We wish to predict the batch return H days after it was allowed
        - t: decision time. This quantity is given in days. Repersents how many days of data (after the batch was allowed)
            our model will use

    *Outputs:
        - snap: snapshot dataframe assuming decision time t and horizon H
    """

    df = df_.copy()
    df['date']              = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
    df['allowlisted_date']  = pd.to_datetime(df['allowlisted_date']).dt.tz_localize(None).dt.normalize()

    #------------------- Daily events per loan

    events = (df.groupby(['loan_id','batch','date'], as_index=False)
                .agg(inflow=('inflow','sum'),
                    billing=('billing','sum'),
                    outflow=('outflow','sum'),
                    cum_in=('cum_in','max'),
                    cum_bill=('cum_bill','max'),
                    roi_cum=('roi_cum','max'),
                    out=('out','max')))
    events['date'] = pd.to_datetime(events['date']).dt.tz_localize(None).astype('datetime64[ns]')
    events = events.sort_values(['loan_id','date'], kind='mergesort').reset_index(drop=True)
    events['cum_outflow'] = events.groupby('loan_id', sort=False)['outflow'].cumsum()

    #------------------- Preparing new columns to compute features at days t-7 and t-30 

    base = (df.drop_duplicates('loan_id')[['loan_id','batch','allowlisted_date','out']].copy())
    base['t0']  = base['allowlisted_date'] + pd.to_timedelta(t,  unit='D')
    base['t7']  = base['t0'] - pd.to_timedelta(7,  unit='D')
    base['t30'] = base['t0'] - pd.to_timedelta(30, unit='D')
    base['tH']  = base['allowlisted_date'] + pd.to_timedelta(H, unit='D')

    for c in ['t0','t7','t30','tH','allowlisted_date']:
        base[c] = pd.to_datetime(base[c]).dt.tz_localize(None).astype('datetime64[ns]')

    first_date = (events.groupby('loan_id', as_index=False)['date']
                        .min().rename(columns={'date':'first_date'}))
    base = base.merge(first_date, on='loan_id', how='left')
    base = base[base['first_date'] <= base['t0']].reset_index(drop=True)

    #------------------- Selecting all the relevant rows and adding the new time columns

    EVENT_COLS = ['date','inflow','billing','outflow',
                'cum_in','cum_bill','cum_outflow','roi_cum','out']

    def loan_snapshots_onepass(events: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
        grp    = events.groupby('loan_id', sort=False)
        base_i = base.set_index('loan_id')
        rows   = []

        def ts_to_i8(ts):
            if pd.isna(ts):
                return np.int64(-2**63)
            return np.int64(pd.Timestamp(ts).value)

        for lid, g in grp:
            if lid not in base_i.index:
                continue
            b = base_i.loc[lid]

            dates_i8 = g['date'].astype('datetime64[ns]').astype('int64').to_numpy()
            t0_i  = ts_to_i8(b['t0']);  t7_i  = ts_to_i8(b['t7'])
            t30_i = ts_to_i8(b['t30']); tH_i  = ts_to_i8(b['tH'])

            def pos_leq(tv_i8):  return np.searchsorted(dates_i8, tv_i8, side='right') - 1
            p0, p7, p30, pH = pos_leq(t0_i), pos_leq(t7_i), pos_leq(t30_i), pos_leq(tH_i)

            rec = {
                'loan_id'         : lid,
                'batch'           : b['batch'],
                'allowlisted_date': b['allowlisted_date'],
                'out'             : b['out'],
                't0'              : b['t0'],  't7': b['t7'],  't30': b['t30'],  'tH': b['tH'],
                'date_t0'         : pd.to_datetime(dates_i8[p0],  unit='ns') if p0  >= 0 else pd.NaT,
                'date_t7'         : pd.to_datetime(dates_i8[p7],  unit='ns') if p7  >= 0 else pd.NaT,
                'date_t30'        : pd.to_datetime(dates_i8[p30], unit='ns') if p30 >= 0 else pd.NaT,
                'date_tH'         : pd.to_datetime(dates_i8[pH],  unit='ns') if pH  >= 0 else pd.NaT,
            }
            for c in EVENT_COLS:
                arr = g[c].to_numpy()
                rec[f'{c}_t0']  = arr[p0]  if p0  >= 0 else np.nan
                rec[f'{c}_t7']  = arr[p7]  if p7  >= 0 else np.nan
                rec[f'{c}_t30'] = arr[p30] if p30 >= 0 else np.nan
                rec[f'{c}_tH']  = arr[pH]  if pH  >= 0 else np.nan

            rows.append(rec)

        return pd.DataFrame(rows)

    snap = loan_snapshots_onepass(events, base)

    #------------------- Windows with differences between cumulative quantities

    def diff0(a, b):
        return snap[a].fillna(0.0) - snap[b].fillna(0.0)

    snap['inflow_7']    = diff0('cum_in_t0',      'cum_in_t7')
    snap['billing_7']   = diff0('cum_bill_t0',    'cum_bill_t7')
    snap['outflow_7']   = diff0('cum_outflow_t0', 'cum_outflow_t7')
    snap['inflow_30']   = diff0('cum_in_t0',      'cum_in_t30')
    snap['billing_30']  = diff0('cum_bill_t0',    'cum_bill_t30')
    snap['outflow_30']  = diff0('cum_outflow_t0', 'cum_outflow_t30')

    snap['cum_in_t']    = snap['cum_in_t0'].fillna(0.0)
    snap['cum_bill_t']  = snap['cum_bill_t0'].fillna(0.0)
    snap['out_fix']     = snap['out'].fillna(0.0)
    snap['net_cash_t']  = snap['cum_in_t'] + snap['cum_bill_t'] - snap['out_fix']

    roi_t0  = snap['roi_cum_t0'].where(snap['roi_cum_t0'].notna(), -1.0)
    roi_t7  = snap['roi_cum_t7'].where(snap['roi_cum_t7'].notna(), -1.0)
    roi_t30 = snap['roi_cum_t30'].where(snap['roi_cum_t30'].notna(), -1.0)
    snap['roi_t']   = roi_t0
    snap['roi_d7']  = roi_t0 - roi_t7
    snap['roi_d30'] = roi_t0 - roi_t30

    snap['frac_paid_t']   = np.where(snap['out_fix']>0, snap['cum_in_t']/snap['out_fix'], 0.0)
    snap['frac_paid_d30'] = np.where(snap['out_fix']>0, diff0('cum_in_t0','cum_in_t30')/snap['out_fix'], 0.0)
    snap['pace_in_7']     = snap['inflow_7']  / 7.0
    snap['pace_in_30']    = snap['inflow_30'] / 30.0

    snap['y_netH']  = (snap['cum_in_tH'].fillna(0.0) + snap['cum_bill_tH'].fillna(0.0) - snap['out_fix'])
    snap['y_roiH']  = np.where(snap['out_fix']>0, snap['y_netH'] / snap['out_fix'], 0.0)

    return snap


def scenario1_data_split_model_train_cash(snap, RNG = 42):

    """
    This function is dedicated to train the scenario 1 model 2 (model that directly predicts the loan net cash)
    Here we are using a XBGRegression model.

    *Inputs:
        - snap: snapshot prepared in the previous function
        - RNG: seed parameter

    *Outputs:
        - snap: snapshot with model predicitons
        - reg: regression model trained
    """

    feat_cols = [
    'roi_t','cum_in_t','cum_bill_t','net_cash_t','frac_paid_t',
    'inflow_7','billing_7','outflow_7',
    'inflow_30','billing_30','outflow_30',
    'pace_in_7','pace_in_30','roi_d7','roi_d30','frac_paid_d30',
    'out_fix'
    ]

    #------------------- Separating the data into X (model input) and y (model output/answer)

    X_loan = snap[feat_cols].fillna(0.0).astype('float32')
    y_net  = snap['y_netH'].astype('float32')

    #------------------- Train test split

    train_idx = []
    for b, d in snap.groupby('batch', sort=False):
        n_tr = int(np.floor(0.8 * len(d)))
        train_idx += d.sample(n=n_tr, random_state=42).index.tolist()
    train_idx = np.array(sorted(set(train_idx)))
    test_idx  = np.array(sorted(set(snap.index) - set(train_idx)))

    Xtr, Xte = X_loan.loc[train_idx], X_loan.loc[test_idx]
    ytr, yte = y_net.loc[train_idx], y_net.loc[test_idx]

    #------------------- Model train

    reg = XGBRegressor(
        n_estimators=1200, learning_rate=0.02,
        max_depth=8, subsample=0.9, colsample_bytree=0.9,
        min_child_weight=1.0, reg_lambda=1.0, reg_alpha=0.0,
        objective='reg:squarederror', random_state=42, n_jobs=-1, verbosity=0,
        tree_method='hist', eval_metric='rmse'
    )
    reg.fit(Xtr, ytr)
    snap.loc[test_idx, 'y_pred_netH'] = reg.predict(Xte)

    return snap, reg


def scenario1_compute_metrics_cash(snap, RNG = 42, eps = 0.006):

    """
    This function is dedicated to save metrics of avaluation between the predicted and true quantities

    *Inputs:
        - snap: snapshot
        - RNG: random seed
        - eps: tolerance for ROI. if pred ROI > eps we consider it as positive. if ROI < eps we consider it negative
            -> In practice, a ROI < eps would mean a negligible return from the batch

    *Outputs:
        - mask_predcohort: mask representing the loans in the test dataset
        - eval_df_cohort: dataframe with true and predicted ROIs
        - metrics: metrics to compare true and predicted quantities
    """
    #------------------- Train indexes

    train_idx = []
    for b, d in snap.groupby('batch', sort=False):
        n_tr = int(np.floor(0.8 * len(d)))
        train_idx += d.sample(n=n_tr, random_state=42).index.tolist()
    train_idx = np.array(sorted(set(train_idx)))
    test_idx  = np.array(sorted(set(snap.index) - set(train_idx)))

    #------------------- Separating test loans

    test_loans = (snap.loc[test_idx, ['batch','out_fix','y_netH','y_pred_netH']].copy())
    test_loans = test_loans.assign(
        roi_true = np.where(test_loans['out_fix'] > 0, test_loans['y_netH']      / test_loans['out_fix'], np.nan),
        roi_pred = np.where(test_loans['out_fix'] > 0, test_loans['y_pred_netH'] / test_loans['out_fix'], np.nan)
    )

    #------------------- Getting true and pred roi per batch

    batch_true_roi = (test_loans.groupby('batch').apply(lambda d: wavg(d['roi_true'], d['out_fix'])).rename('true_roiH'))
    batch_pred_roi = (test_loans.groupby('batch').apply(lambda d: wavg(d['roi_pred'], d['out_fix'])).rename('pred_roiH'))
    eval_roi_cohort = (pd.concat([batch_true_roi, batch_pred_roi], axis=1).dropna().reset_index())

    #-------------------Getting true and pred net cash per batch
    batch_true_net = (test_loans.groupby('batch')['y_netH'].sum().rename('true_netH'))
    batch_pred_net = (test_loans.groupby('batch')['y_pred_netH'].sum().rename('pred_netH'))
    eval_net_cohort = (pd.concat([batch_true_net, batch_pred_net], axis=1).dropna().reset_index())

    #------------------- Computing batch level ROI metrics

    rmse_roi = mean_squared_error(eval_roi_cohort['true_roiH'], eval_roi_cohort['pred_roiH'], squared=False)
    mae_roi  = mean_absolute_error(eval_roi_cohort['true_roiH'],  eval_roi_cohort['pred_roiH'])
    sign_true = (eval_roi_cohort['true_roiH'] > eps).astype(int)
    sign_pred = (eval_roi_cohort['pred_roiH'] > eps).astype(int)
    acc_sign  = (sign_true == sign_pred).mean()
    bacc      = balanced_accuracy_score(sign_true, sign_pred) if sign_true.nunique() > 1 else np.nan

    rmse_net = mean_squared_error(eval_net_cohort['true_netH'], eval_net_cohort['pred_netH'], squared=False)
    mae_net  = mean_absolute_error(eval_net_cohort['true_netH'],  eval_net_cohort['pred_netH'])

    #------------------- Computing loan level net cash metrics
    loan_te = snap.loc[test_idx, ['loan_id','batch','out_fix','y_netH','y_pred_netH']].copy()
    mask_nc = loan_te['y_pred_netH'].notna() & np.isfinite(loan_te['y_netH'])

    rmse_nc_loan = mean_squared_error(loan_te.loc[mask_nc,'y_netH'],
                                    loan_te.loc[mask_nc,'y_pred_netH'], squared=False)
    mae_nc_loan  = mean_absolute_error(loan_te.loc[mask_nc,'y_netH'],
                                    loan_te.loc[mask_nc,'y_pred_netH'])

    #------------------- Computing loan level net cash

    loan_te['roi_true'] = np.where(loan_te['out_fix'] > 0,
                                loan_te['y_netH'] / loan_te['out_fix'], np.nan)
    loan_te['roi_pred'] = np.where(loan_te['out_fix'] > 0,
                                loan_te['y_pred_netH'] / loan_te['out_fix'], np.nan)

    mask_roi = loan_te['roi_true'].notna() & loan_te['roi_pred'].notna()

    rmse_roi_loan = mean_squared_error(loan_te.loc[mask_roi,'roi_true'],
                                    loan_te.loc[mask_roi,'roi_pred'], squared=False)
    mae_roi_loan  = mean_absolute_error(loan_te.loc[mask_roi,'roi_true'],
                                        loan_te.loc[mask_roi,'roi_pred'])

    #------------------- Batch signal metrics 
    sign_true_loan = (loan_te.loc[mask_roi,'roi_true'] > eps).astype(int)
    sign_pred_loan = (loan_te.loc[mask_roi,'roi_pred'] > eps).astype(int)
    acc_sign_loan  = (sign_true_loan == sign_pred_loan).mean()
    bacc_loan      = balanced_accuracy_score(sign_true_loan, sign_pred_loan) \
                    if sign_true_loan.nunique() > 1 else np.nan

    #------------------- Printing metrics
    print(f"[Loan-level | NET_CASH(H)] RMSE={rmse_nc_loan:.6f} | MAE={mae_nc_loan:.6f}")
    print(f"[Loan-level | ROI(H)] RMSE={rmse_roi_loan:.6f} | MAE={mae_roi_loan:.6f} | "
        f"SignACC={acc_sign_loan:.3f} | BACC={bacc_loan if bacc_loan==bacc_loan else float('nan'):.3f}")
    print('-'*50)
    print(f"[Batch-level | NET_CASH(H)] RMSE={rmse_net:.6f} | MAE={mae_net:.6f}")
    print(f"[Batch-level | ROI(H)]  RMSE={rmse_roi:.6f} | MAE={mae_roi:.6f} | SignACC={acc_sign:.3f} | BACC={bacc if bacc==bacc else float('nan'):.3f}")

    metrics = {
        'rmse_nc_loan': rmse_nc_loan,
        'mae_nc_loan': mae_nc_loan,
        'rmse_roi_loan': rmse_roi_loan,
        'mae_roi_loan': mae_roi_loan,
        'acc_sign_loan': acc_sign_loan,
        'bacc_loan': bacc_loan,
        'rmse_roi_b': rmse_roi, 
        'mae_roi_b': mae_roi,
        'acc_sign_b': acc_sign,
        'bacc_b': bacc,
        'rmse_net_b': rmse_net,
        'mae_net_b': mae_net,
        'comp_metric': rmse_roi
        }
    
    return eval_net_cohort, eval_roi_cohort, metrics

def scenario1_plots_cash(scenario1_snap_cash, scenario1_eval_b_net, scenario1_eval_b_roi, scenario1_metrics_cash, savefig = True, filtered = '', RNG = 42):

    """
    This function is dedicated to plot our model results

    *Inputs:
        - scenario1_snap_cash: snapshot
        - scenario1_eval_b_net: evaluation net cash dataframe
        - scenario1_eval_b_roi: evaluation roi dataframe
        - scenario1_metrics_cashi: metric dictionary
        - savefig: if True, the plots are saved to disk
        - filtered: in case you use a filtered dataframe, add a string here to modify the figure name
    """

    #------------------- Computing indexes

    train_idx = []
    for b, d in scenario1_snap_cash.groupby('batch', sort=False):
        n_tr = int(np.floor(0.8 * len(d)))
        train_idx += d.sample(n=n_tr, random_state=42).index.tolist()
    train_idx = np.array(sorted(set(train_idx)))
    test_idx  = np.array(sorted(set(scenario1_snap_cash.index) - set(train_idx)))
    # ================= Helpers =================
    def compute_lims(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        lo = np.nanmin(np.concatenate([y_true, y_pred]))
        hi = np.nanmax(np.concatenate([y_true, y_pred]))
        if not np.isfinite(hi - lo) or (hi - lo) == 0:
            lo -= 1.0; hi += 1.0
        pad = 0.05 * (hi - lo)
        return (lo - pad, hi + pad)

    #------------------- Arranging data

    loan_plot = (scenario1_snap_cash.loc[scenario1_snap_cash['y_pred_netH'].notna(),
                        ['loan_id','batch','out_fix','y_netH','y_pred_netH','y_roiH']]
                    .copy())

    loan_plot['y_pred_roiH'] = np.where(loan_plot['out_fix'] > 0,
                                        loan_plot['y_pred_netH'] / loan_plot['out_fix'],
                                        np.nan)
    loan_plot_roi = loan_plot[np.isfinite(loan_plot['y_pred_roiH']) & (loan_plot['out_fix'] > 0)].copy()

    if 'eval_net_cohort' in globals() and 'eval_roi_cohort' in globals():
        eval_net_plot = scenario1_eval_b_net.dropna(subset=['pred_netH']).copy()
        eval_roi_plot = scenario1_eval_b_roi.dropna(subset=['pred_roiH']).copy()
    else:
        test_loans = (scenario1_snap_cash.loc[test_idx, ['batch','out_fix','y_netH','y_pred_netH']].copy())
        test_loans = test_loans.assign(
            roi_true = np.where(test_loans['out_fix'] > 0, test_loans['y_netH']      / test_loans['out_fix'], np.nan),
            roi_pred = np.where(test_loans['out_fix'] > 0, test_loans['y_pred_netH'] / test_loans['out_fix'], np.nan)
        )
        batch_true_net = (test_loans.groupby('batch')['y_netH'].sum().rename('true_netH'))
        batch_pred_net = (test_loans.groupby('batch')['y_pred_netH'].sum().rename('pred_netH'))
        eval_net_plot  = (pd.concat([batch_true_net, batch_pred_net], axis=1).dropna().reset_index())

        batch_true_roi = (test_loans.groupby('batch').apply(lambda d: wavg(d['roi_true'], d['out_fix'])).rename('true_roiH'))
        batch_pred_roi = (test_loans.groupby('batch').apply(lambda d: wavg(d['roi_pred'], d['out_fix'])).rename('pred_roiH'))
        eval_roi_plot  = (pd.concat([batch_true_roi, batch_pred_roi], axis=1).dropna().reset_index())

    #------------------- First plot (loan level net cash)
    plt.figure(figsize=(8,6))
    lims_nc_loan = compute_lims(loan_plot['y_netH'], loan_plot['y_pred_netH'])
    i = 0
    for b, g in loan_plot.groupby('batch'):
        i+=1
        plt.scatter(g['y_netH'], g['y_pred_netH'], s=22, color=list(mcolors.TABLEAU_COLORS.keys())[i-1], label='Batch ' + str(i))
    plt.plot(lims_nc_loan, lims_nc_loan, 'k--', lw=1)
    plt.axvline(0, color='k', ls=':', lw=1); plt.axhline(0, color='k', ls=':', lw=1)
    plt.xlim(lims_nc_loan); plt.ylim(lims_nc_loan)
    plt.xlabel('Net cash(H) true (\%)', fontsize = 12)
    plt.ylabel('Net cash pred (\%)', fontsize = 12)
    plt.title(f'Loan-level: net cash (H) -- RMSE = {scenario1_metrics_cash["rmse_nc_loan"]:.0f}')
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig('Figures/'+filtered+'scenario1cash_loan_cash.png', dpi = 300, format = 'png')
    plt.show()

    #------------------- Second plot (loan level ROI)
    plt.figure(figsize=(8,6))
    x_roi = loan_plot_roi['y_roiH'] * 100.0
    y_roi = loan_plot_roi['y_pred_roiH'] * 100.0
    lims_roi_loan = compute_lims(x_roi, y_roi)
    i = 0
    for b, g in loan_plot_roi.groupby('batch'):
        i+=1
        plt.scatter(g['y_roiH']*100, g['y_pred_roiH']*100, s=22, color=list(mcolors.TABLEAU_COLORS.keys())[i-1], label = 'Batch ' + str(i))
    plt.plot([-1000000, 1000000], [-1000000, 1000000], 'k--', lw=1)
    plt.axvline(0, color='k', ls=':', lw=1); plt.axhline(0, color='k', ls=':', lw=1)
    plt.xlim((-200, 200))
    plt.ylim((-200, 200))
    plt.xlabel('ROI(H) true (\%)', fontsize = 12)
    plt.ylabel('ROI(H) pred (\%)', fontsize = 12)
    plt.title(f'Loan-level: ROI(H) -- RMSE = {scenario1_metrics_cash["rmse_roi_loan"]:.6f}')
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig('Figures/'+filtered+'scenario1cash_loan_roi.png', dpi = 300, format = 'png')
    plt.show()

    #------------------- Third plot (batch level net cash)
    plt.figure(figsize=(8,6))
    lims_nc_batch = compute_lims(eval_net_plot['true_netH'], eval_net_plot['pred_netH'])
    i = 0
    for _, r in eval_net_plot.iterrows():
        i+=1
        b = r['batch']
        plt.scatter(r['true_netH'], r['pred_netH'], s=70, color=list(mcolors.TABLEAU_COLORS.keys())[i-1], label='Batch ' + str(i))
    plt.plot(lims_nc_batch, lims_nc_batch, 'k--', lw=1)
    plt.axvline(0, color='k', ls=':', lw=1); plt.axhline(0, color='k', ls=':', lw=1)
    plt.xlim(lims_nc_batch); plt.ylim(lims_nc_batch)
    plt.xlabel('Net cash(H) true', fontsize = 12)
    plt.ylabel('Net cash(H) pred', fontsize = 12)
    plt.title(f'Batch-level: net cash (H) -- RMSE = {scenario1_metrics_cash["rmse_net_b"]:.0f}')
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig('Figures/'+filtered+'scenario1cash_batch_cash.png', dpi = 300, format = 'png')
    plt.show()

    #------------------- Organizing batch data and computing CI
    cohort = scenario1_snap_cash.loc[test_idx, ['batch','out_fix','y_netH','y_pred_netH']].copy()
    cohort['roi_true'] = np.where(cohort['out_fix'] > 0, cohort['y_netH']      / cohort['out_fix'], np.nan)
    cohort['roi_pred'] = np.where(cohort['out_fix'] > 0, cohort['y_pred_netH'] / cohort['out_fix'], np.nan)

    batch_order = scenario1_eval_b_net['batch'].tolist()

    rows = []
    for b in batch_order:
        g = cohort[cohort['batch'] == b]
        if g.empty:
            continue
        mu_t, lo_t, hi_t = wbootstrap_mean_ci(g['roi_true'], g['out_fix'], B=2000, seed=101)
        mu_p, lo_p, hi_p = wbootstrap_mean_ci(g['roi_pred'], g['out_fix'], B=2000, seed=202)
        rows.append({
            'batch': b,
            'true_mu': mu_t, 'true_lo': lo_t, 'true_hi': hi_t,
            'pred_mu': mu_p, 'pred_lo': lo_p, 'pred_hi': hi_p
        })
    eval_roi_boot = pd.DataFrame(rows).dropna(subset=['true_mu','pred_mu'])

    #------------------- Fourth plot (batch level roi + CI)

    plt.figure(figsize=(8,6))

    for i, b in enumerate(batch_order, start=1):
        r = eval_roi_boot.loc[eval_roi_boot['batch'] == b]
        if r.empty:
            continue
        r = r.iloc[0]
        x = r['true_mu']*100.0
        y = r['pred_mu']*100.0
        yerr = np.array([[y - r['pred_lo']*100.0], [r['pred_hi']*100.0 - y]])
        plt.errorbar(
            x=x, y=y, yerr=yerr,
            fmt='o', ms=7, capsize=3, elinewidth=1.2,
            color=list(mcolors.TABLEAU_COLORS.keys())[i-1],
            label=f'Batch {i}'
        )


    plt.axvline(0, color='k', ls=':', lw=1)
    plt.axhline(0, color='k', ls=':', lw=1)
    plt.plot([-100,100], [-100,100], 'k--', lw=1)
    plt.xlim((-18,18)); plt.ylim((-18,18))
    plt.xlabel('ROI(H) true (\%)', fontsize = 12)
    plt.ylabel('ROI(H) pred (\%)', fontsize = 12)
    plt.title(f'Batch-level: ROI(H) -- RMSE = {scenario1_metrics_cash["rmse_roi_b"]:.6f}')
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig('Figures/'+filtered+'scenario1cash_batch_roi.png', dpi = 300, format = 'png')
    plt.show()

def scenario1_pre_err_cash(scenario1_snap_cash, scenario1_eval_b_net, RNG = 42):
    train_idx = []
    for b, d in scenario1_snap_cash.groupby('batch', sort=False):
        n_tr = int(np.floor(0.8 * len(d)))
        train_idx += d.sample(n=n_tr, random_state=42).index.tolist()
    train_idx = np.array(sorted(set(train_idx)))
    test_idx  = np.array(sorted(set(scenario1_snap_cash.index) - set(train_idx)))
    cohort = scenario1_snap_cash.loc[test_idx, ['batch','out_fix','y_netH','y_pred_netH']].copy()
    cohort['roi_true'] = np.where(cohort['out_fix'] > 0, cohort['y_netH']      / cohort['out_fix'], np.nan)
    cohort['roi_pred'] = np.where(cohort['out_fix'] > 0, cohort['y_pred_netH'] / cohort['out_fix'], np.nan)

    batch_order = scenario1_eval_b_net['batch'].tolist()

    rows = []
    for b in batch_order:
        g = cohort[cohort['batch'] == b]
        if g.empty:
            continue
        mu_t, lo_t, hi_t = wbootstrap_mean_ci(g['roi_true'], g['out_fix'], B=2000, seed=101)
        mu_p, lo_p, hi_p = wbootstrap_mean_ci(g['roi_pred'], g['out_fix'], B=2000, seed=202)
        rows.append({
            'batch': b,
            'true_mu': mu_t, 'true_lo': lo_t, 'true_hi': hi_t,
            'pred_mu': mu_p, 'pred_lo': lo_p, 'pred_hi': hi_p
        })
    eval_roi_boot = pd.DataFrame(rows).dropna(subset=['true_mu','pred_mu'])

    return eval_roi_boot