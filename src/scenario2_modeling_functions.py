import sys
sys.path.append('..')  # Adjust the path as needed
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tqdm import tqdm

from src.read_data import *
from src.frame_creation import *


from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, balanced_accuracy_score
from xgboost import XGBRegressor
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


"""
This file is dedicated to host the functions used in the modeling scenario 2.
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

    eval_ = eval_.sort_values('batch')

    
    return eval_.sort_values('batch')


def canonical_truth_from_CF(CF: pd.DataFrame, allow_df: pd.DataFrame, H: int, mode: str = 'asof') -> pd.DataFrame:
    CF2 = CF.copy()
    CF2['date'] = pd.to_datetime(CF2['date']).dt.tz_localize(None).dt.normalize()

    allow = (allow_df.drop_duplicates('batch').set_index('batch')['allowlisted_date'])
    allow = pd.to_datetime(allow).dt.tz_localize(None).dt.normalize()

    rows = []
    for b, g in CF2.groupby('batch', sort=False):
        if b not in allow.index: 
            continue
        tgt = allow.loc[b] + pd.Timedelta(days=H)
        g   = g.sort_values('date')

        sel = g[g['date'] <= tgt].tail(1)
        if sel.empty:
            sel = g.head(1)

        row = sel.iloc[0]
        rows.append({
            'batch': b, 'target_date': tgt,
            'true_netH': float(row['cum_in'] + row['cum_bill'] - row['cum_out']),
            'true_roiH': float(row['roi_cum'])
        })
    return pd.DataFrame(rows)

def wavg(y, w):
    y = np.asarray(y); w = np.asarray(w)
    m = np.isfinite(y) & np.isfinite(w) & (w > 0)
    return np.average(y[m], weights=w[m]) if m.any() else np.nan

def compute_lims(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    lo = np.nanmin(np.concatenate([x, y]))
    hi = np.nanmax(np.concatenate([x, y]))
    if not np.isfinite(hi - lo) or (hi - lo) == 0:
        lo -= 1.0; hi += 1.0
    pad = 0.05 * (hi - lo)
    return (lo - pad, hi + pad)

def wmean_bootstrap(x, w, B=2000, seed=0):
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[m]; w = w[m]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    p = w / w.sum()
    mu = (p * x).sum()
    n = len(x)
    if n == 1:
        return mu, mu, mu
    boots = np.empty(B, float)
    for b in range(B):
        idx = rng.choice(n, size=n, replace=True, p=p)
        boots[b] = x[idx].mean()  # média simples; p embute os pesos
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mu, lo, hi



#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
############################################# MODEL 1 ################################################################
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------


def scenario2_snap_roi_func(df_, H, t):

        
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
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)  # sem normalize aqui, não é obrigatório
    df['allowlisted_date'] = pd.to_datetime(df['allowlisted_date']).dt.tz_localize(None).dt.normalize()

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

        def pos_leq(tv_i8):
            return np.searchsorted(dates_i8, tv_i8, side='right') - 1
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

    # ============ 4) Features no snapshot t ============
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


def scenario2_data_split_model_train_roi(snap, RNG = 42):

    """
    This function is dedicated to train the scenario 2 model 1 (model that directly predicts the loan ROI)
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

    K = min(5, groups.nunique())  # até 5 dobras
    gkf = GroupKFold(n_splits=K)

    snap['y_pred_oof'] = np.nan
    all_zero_mask = (X_all.abs().sum(axis=1) == 0)

    #------------------- Training the model

    for fold, (tr, val) in enumerate(gkf.split(X_all, y_all, groups), 1):
        tr_use = np.array([i for i in tr if not all_zero_mask.iloc[i]])
        Xtr, ytr = X_all.iloc[tr_use].copy(), y_all.iloc[tr_use].copy()
        Xval     = X_all.iloc[val].copy()

        nun = Xtr.nunique(dropna=False)
        keep = nun[nun > 1].index.tolist()
        if len(keep) == 0:
            continue

        reg = XGBRegressor(
            n_estimators=800, learning_rate=0.03, max_depth=6,
            subsample=0.9, colsample_bytree=0.8, min_child_weight=1.0,
            reg_lambda=1.0, objective='reg:squarederror',
            tree_method='hist', n_jobs=-1, random_state=42, eval_metric='rmse', verbosity=0
        )
        reg.fit(Xtr[keep], ytr)
        snap.loc[X_all.iloc[val].index, 'y_pred_oof'] = reg.predict(Xval[keep])

    return snap, reg


def scenario2_compute_metrics_roi(snap, df_, H, eps = 0.006):

    """
    This function is dedicated to save metrics of avaluation between the predicted ROI and true ROI
    Since the ROI can be very small, we also compute a metric to check if the model can correctly predict
    if the ROI is positive or negative based on a tolerance eps

    *Inputs:
        - snap: snapshot
        - df_: dataframe of all loans (loan_CF)
        - RNG: random seed
        - eps: tolerance for ROI. if pred ROI > eps we consider it as positive. if ROI < eps we consider it negative
            -> In practice, a ROI < eps would mean a negligible return from the batch

    *Outputs:
        - mask_predt: mask representing the loans in the test dataset
        - truth_can: value of ROI(H) at H, considering all of the loans (from loan_CF)
            -> Unlike scenario 1, here we take the batch ROI also considering the loans taken between t and H
        - metrics: metrics to compare true and predicted ROIs
    """

    #------------------- Creating dataframe of allowed loans and batches

    allow_df = (df_.drop_duplicates('batch')[['batch','allowlisted_date']]
              .assign(allowlisted_date=lambda d: pd.to_datetime(d['allowlisted_date'])
                                                    .dt.tz_localize(None).dt.normalize()))
    
    #------------------- Constructing mask of predictions and computing metrics

    mask_pred = snap['y_pred_oof'].notna()
    rmse_loan = mean_squared_error(snap.loc[mask_pred,'y_roiH'],
                                snap.loc[mask_pred,'y_pred_oof'], squared=False)
    mae_loan  = mean_absolute_error(snap.loc[mask_pred,'y_roiH'],
                                    snap.loc[mask_pred,'y_pred_oof'])
    sign_true_loan = (snap.loc[mask_pred,'y_roiH'] > eps).astype(int)
    sign_pred_loan = (snap.loc[mask_pred,'y_pred_oof'] > eps).astype(int)
    acc_sign_loan  = (sign_true_loan == sign_pred_loan).mean()
    bacc_loan      = balanced_accuracy_score(sign_true_loan, sign_pred_loan) \
                    if sign_true_loan.nunique() > 1 else np.nan

    #------------------- Building a batch level dataframe

    CF_use = rebuild_CF_from_loanCF(df_)

    #------------------- Getting what I call canonical truth: the ROI value at H assuming all loans

    truth_can = canonical_truth_from_CF(CF_use, allow_df, H, mode='asof')

    #------------------- Agregating predictions

    oof = snap.loc[mask_pred, ['batch','out_fix','y_pred_oof']]
    batch_pred_roi = (oof.groupby('batch')
                        .apply(lambda d: wavg(d['y_pred_oof'], d['out_fix']))
                        .rename('pred_roiH')
                        .reset_index())

    #------------------- More metrics

    eval_roi_can = truth_can[['batch','true_roiH']].merge(batch_pred_roi, on='batch', how='left')

    rmse_roi_can = mean_squared_error(eval_roi_can['true_roiH'], eval_roi_can['pred_roiH'], squared=False)
    mae_roi_can  = mean_absolute_error(eval_roi_can['true_roiH'],  eval_roi_can['pred_roiH'])
    sign_true    = (eval_roi_can['true_roiH'] > eps).astype(int)
    sign_pred    = (eval_roi_can['pred_roiH'] > eps).astype(int)
    acc_sign     = (sign_true == sign_pred).mean()
    bacc         = balanced_accuracy_score(sign_true, sign_pred) if sign_true.nunique() > 1 else np.nan
    
    
    #------------------- Printing metrics

    print(f"[Loan-level | ROI(H)] RMSE={rmse_loan:.6f} | MAE={mae_loan:.6f} | "
        f"SignACC={acc_sign_loan:.3f} | BACC={bacc_loan:.3f}")
    print('-'*50)
    print(f"[Batch-level | ROI(H)] RMSE={rmse_roi_can:.6f} | MAE={mae_roi_can:.6f} | "
        f"SignACC={acc_sign:.3f} | BACC={bacc:.3f}")
    
    metrics = {
    'rmse_loan': rmse_loan,
    'mae_loan': mae_loan,
    'acc_sign_loan': acc_sign_loan,
    'bacc_loan': bacc_loan,
    'rmse_b': rmse_roi_can,
    'mae_b': mae_roi_can ,
    'acc_sign_b': acc_sign,
    'bacc_b': bacc,
    'comp_metric': rmse_roi_can
    }

    return mask_pred, truth_can, metrics

def scenario2_plots_roi(snap, truth_can, scenario2_metrics_roi, savefig = True, filtered = ''):
    """
    This function is dedicated to plot our model results

    *Inputs:
        - snap: snapshot
        - truth_can: canonical value of ROI(H)
        - scenario2_metrics_roi: metrics dictionary
        - savefig: if True, the plots are saved to disk
        - filtered: in case you use a filtered dataframe, add a string here to modify the figure name
    """

    B = 2000
    eps = 0.006
    seed_true: int = 42
    seed_pred: int = 42


    # ---------- Loan-level (OOF) ----------
    loan_eval = (snap.loc[snap['y_pred_oof'].notna(), ['loan_id','batch','out_fix','y_roiH','y_pred_oof']]
                    .rename(columns={'y_roiH':'y_true_loan','y_pred_oof':'y_pred_loan'})
                    .copy())

    loan_eval = loan_eval[loan_eval['y_true_loan'] > -1].copy()

    #------------------- First plot (loan-level ROI)

    plt.figure(figsize=(8,6))
    x_loan = loan_eval['y_true_loan'] * 100.0
    y_loan = loan_eval['y_pred_loan'] * 100.0
    lims_loan = compute_lims(x_loan, y_loan)
    i = 0
    for b, g in loan_eval.groupby('batch'):
        i+=1
        plt.scatter(g['y_true_loan']*100.0, g['y_pred_loan']*100.0,
                    s=18, color=list(mcolors.TABLEAU_COLORS.keys())[i-1], label='Batch ' + str(i))

    plt.plot(lims_loan, lims_loan, 'k--', lw=1)
    plt.axvline(0, color='k', ls=':', lw=1)
    plt.axhline(0, color='k', ls=':', lw=1)
    plt.xlim(lims_loan); plt.ylim(lims_loan)
    plt.xlabel('ROI(H) true (\%)', fontsize = 12)
    plt.ylabel('ROI(H) pred (\%)', fontsize = 12)
    plt.title(f'Loan-level ROI -- RMSE = {scenario2_metrics_roi["rmse_loan"]:.6f}')
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig('Figures/' + filtered+'scenario2_loan_roi.png', dpi = 300, format = 'png')
    plt.show()


    #------------------- Preparing loan level data

    lp = snap.loc[:, ['batch','out_fix','y_roiH','y_pred_oof']].copy()
    lp = lp[np.isfinite(lp['y_roiH']) & (lp['out_fix'] > 0)].copy()

    #------------------- Compute batch level CI
    rows = []
    for b, g in lp.groupby('batch'):
        # CI do verdadeiro usando ROIs por-loan (centrado em truth_can no gráfico)
        true_mu, true_lo, true_hi = wmean_bootstrap(g['y_roiH'], g['out_fix'], B=B, seed=seed_true)
        pred_mu, pred_lo, pred_hi = wmean_bootstrap(g['y_pred_oof'], g['out_fix'], B=B, seed=seed_pred)

        rows.append({
            'batch': b,
            'true_mu_loans': true_mu, 'true_lo': true_lo, 'true_hi': true_hi,
            'pred_mu': pred_mu,       'pred_lo': pred_lo, 'pred_hi': pred_hi
        })
    ci_df = pd.DataFrame(rows)


    eval_plot = truth_can[['batch','true_roiH']].merge(ci_df, on='batch', how='inner')
    eval_plot = eval_plot.dropna(subset=['pred_mu'])  # precisa ter predição no batch

    #------------------- Second plot (batch-level ROI + error)

    x = eval_plot['true_roiH'].to_numpy()
    y = eval_plot['pred_mu'].to_numpy()

    yerr = np.vstack([y - eval_plot['pred_lo'].to_numpy(),
                      eval_plot['pred_hi'].to_numpy() - y])

    x, y, yerr = x*100.0, y*100.0, yerr*100.0

    plt.figure(figsize=(8,6))
    c = 0
    for i, r in eval_plot.iterrows():
        c+=1
        b = r['batch']
        xi = x[i]; yi = y[i]
        yi_err = np.c_[yerr[0, i], yerr[1, i]].T
        plt.errorbar(x=xi, y=yi, yerr=yi_err.reshape(2,1),
                     fmt='o', ms=6, color=list(mcolors.TABLEAU_COLORS.keys())[c-1],
                     elinewidth=1.2, capsize=3, alpha=0.9, label='Batch ' + str(c))

    plt.plot([-1000000, 1000000], [-1000000, 1000000], 'k--', lw=1)
    plt.axvline(0, color='k', ls=':', lw=1)
    plt.axhline(0, color='k', ls=':', lw=1)
    plt.xlim((-19,19))
    plt.ylim((-19,19))
    plt.xlabel('ROI(H) real do batch (\%)', fontsize = 12)
    plt.ylabel('ROI(H) predito do batch (\%)', fontsize = 12)
    plt.title(f'Batch-level ROI -- RMSE = {scenario2_metrics_roi["rmse_b"]:.6f}')
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig('Figures/'+filtered+'scenario2_batch_roi_werror.png', dpi = 300, format = 'png')
    plt.show()


def scenario2_pre_err_roi(snap, truth_can, RNG = 42, B = 2000):
    lp = snap.loc[:, ['batch','out_fix','y_roiH','y_pred_oof']].copy()
    lp = lp[np.isfinite(lp['y_roiH']) & (lp['out_fix'] > 0)].copy()

    #------------------- Compute batch level CI
    rows = []
    for b, g in lp.groupby('batch'):
        # CI do verdadeiro usando ROIs por-loan (centrado em truth_can no gráfico)
        true_mu, true_lo, true_hi = wmean_bootstrap(g['y_roiH'], g['out_fix'], B=B, seed=RNG)
        pred_mu, pred_lo, pred_hi = wmean_bootstrap(g['y_pred_oof'], g['out_fix'], B=B, seed=RNG)

        rows.append({
            'batch': b,
            'true_mu_loans': true_mu, 'true_lo': true_lo, 'true_hi': true_hi,
            'pred_mu': pred_mu,       'pred_lo': pred_lo, 'pred_hi': pred_hi
        })
    ci_df = pd.DataFrame(rows)
    eval_plot = truth_can[['batch','true_roiH']].merge(ci_df, on='batch', how='inner')
    eval_plot = eval_plot.dropna(subset=['pred_mu'])  # precisa ter predição no batch

    return eval_plot


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
############################################# MODEL 2 ################################################################
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------


def scenario2_snap_cash_func(df_, H, t):
    
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


    allow_df = (df.drop_duplicates('batch')[['batch','allowlisted_date']]
                .assign(allowlisted_date=lambda d: pd.to_datetime(d['allowlisted_date'])
                                                        .dt.tz_localize(None).dt.normalize()))

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

    # garantir dtype datetime64[ns] (sem timezone)
    for c in ['t0','t7','t30','tH','allowlisted_date']:
        base[c] = pd.to_datetime(base[c]).dt.tz_localize(None).astype('datetime64[ns]')

    # manter apenas loans que "existem" até t0
    first_date = (events.groupby('loan_id', as_index=False)['date']
                        .min().rename(columns={'date':'first_date'}))
    base = base.merge(first_date, on='loan_id', how='left')
    base = base[base['first_date'] <= base['t0']].reset_index(drop=True)

    #------------------- Selecting all the relevant rows and adding the new time columns

    EVENT_COLS = ['date','inflow','billing','outflow',
                'cum_in','cum_bill','cum_outflow','roi_cum','out']

    def loan_snapshots_onepass(events: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
        """
        Procura, para cada loan, o último evento <= t0, t7, t30, tH.
        Converte datas para int64(ns) para evitar erros de comparação entre tipos.
        """
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
            t0_i  = ts_to_i8(b['t0'])
            t7_i  = ts_to_i8(b['t7'])
            t30_i = ts_to_i8(b['t30'])
            tH_i  = ts_to_i8(b['tH'])

            def pos_leq(tv_i8):
                return np.searchsorted(dates_i8, tv_i8, side='right') - 1

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

    # janelas via diferença de acumulados
    snap['inflow_7']    = diff0('cum_in_t0',      'cum_in_t7')
    snap['billing_7']   = diff0('cum_bill_t0',    'cum_bill_t7')
    snap['outflow_7']   = diff0('cum_outflow_t0', 'cum_outflow_t7')
    snap['inflow_30']   = diff0('cum_in_t0',      'cum_in_t30')
    snap['billing_30']  = diff0('cum_bill_t0',    'cum_bill_t30')
    snap['outflow_30']  = diff0('cum_outflow_t0', 'cum_outflow_t30')

    # estados em t
    snap['cum_in_t']    = snap['cum_in_t0'].fillna(0.0)
    snap['cum_bill_t']  = snap['cum_bill_t0'].fillna(0.0)
    snap['out_fix']     = snap['out'].fillna(0.0)
    snap['net_cash_t']  = snap['cum_in_t'] + snap['cum_bill_t'] - snap['out_fix']

    # deltas de ROI (opcional, ajudam como sinal)
    roi_t0  = snap['roi_cum_t0'].where(snap['roi_cum_t0'].notna(), -1.0)
    roi_t7  = snap['roi_cum_t7'].where(snap['roi_cum_t7'].notna(), -1.0)
    roi_t30 = snap['roi_cum_t30'].where(snap['roi_cum_t30'].notna(), -1.0)
    snap['roi_t']   = roi_t0
    snap['roi_d7']  = roi_t0 - roi_t7
    snap['roi_d30'] = roi_t0 - roi_t30

    # frações e ritmos
    snap['frac_paid_t']   = np.where(snap['out_fix']>0, diff0('cum_in_t0','cum_in_t0')/snap['out_fix'] + snap['cum_in_t']/snap['out_fix'], 0.0)  # = cum_in_t / out_fix
    snap['frac_paid_t']   = np.where(snap['out_fix']>0, snap['cum_in_t']/snap['out_fix'], 0.0)  # (linha acima só para destacar a ideia)
    snap['frac_paid_d30'] = np.where(snap['out_fix']>0, diff0('cum_in_t0','cum_in_t30')/snap['out_fix'], 0.0)
    snap['pace_in_7']     = snap['inflow_7']  / 7.0
    snap['pace_in_30']    = snap['inflow_30'] / 30.0

    # ALVOS em H
    snap['y_netH']  = (snap['cum_in_tH'].fillna(0.0) + snap['cum_bill_tH'].fillna(0.0) - snap['out_fix'])
    snap['y_roiH']  = np.where(snap['out_fix']>0, snap['y_netH'] / snap['out_fix'], 0.0)

    return snap


def scenario2_data_split_model_train_cash(snap, RNG = 42):

    """
    This function is dedicated to train the scenario 2 model 2 (model that directly predicts the loan net cash)
    Here we are using a XBGRegression model.

    *Inputs:
        - snap: snapshot prepared in the previous function
        - RNG: seed parameter

    *Outputs:
        - snap: snapshot with model predicitons
        - reg: regression model trained
    """

    feat_cols = [
        'roi_t','cum_in_t', 'cum_bill_t', 'net_cash_t','frac_paid_t',
        'inflow_7','billing_7', 'outflow_7', 'inflow_30','billing_30',
        'outflow_30', 'pace_in_7','pace_in_30', 'roi_d7','roi_d30',
        'frac_paid_d30', 'out_fix'
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
        tree_method='hist',
        eval_metric='rmse'
    )
    reg.fit(Xtr, ytr)
    snap.loc[test_idx, 'y_pred_netH'] = reg.predict(Xte)

    return snap, reg, test_idx


def scenario2_compute_metrics_cash(df, scenario2_snap_cash, test_idx, H, eps = 0.006):

    """
    This function is dedicated to save metrics of avaluation between the predicted and true quantities

    *Inputs:
        - snap: snapshot
        - RNG: random seed
        - eps: tolerance for ROI. if pred ROI > eps we consider it as positive. if ROI < eps we consider it negative
            -> In practice, a ROI < eps would mean a negligible return from the batch

    *Outputs:
        - eval_net_can: canonical evaluation dataframe for net cash
        - eval_roi_can: canonical evaluation dataframe for ROI
        - metrics: metrics to compare true and predicted quantities
    """
    #------------------- Train indexes

    snap = scenario2_snap_cash
    df = df.copy()

    allow_df = (df.drop_duplicates('batch')[['batch','allowlisted_date']]
                .assign(allowlisted_date=lambda d: pd.to_datetime(d['allowlisted_date'])
                                                        .dt.tz_localize(None).dt.normalize()))

    CF_use = rebuild_CF_from_loanCF(df)

    truth = canonical_truth_from_CF(CF_use, allow_df, H, mode='asof')

    def wavg_num_denom(num, den):
        num = np.asarray(num); den = np.asarray(den)
        m = np.isfinite(num) & np.isfinite(den) & (den > 0)
        return (num[m].sum(), den[m].sum())
    
    #------------------- Separating test loans

    test_loans = snap.loc[test_idx, ['loan_id','batch','out_fix','y_netH','y_pred_netH']].copy()

    #------------------- Net cash from test data

    batch_pred_net = (test_loans.groupby('batch')['y_pred_netH'].sum()
                      .rename('pred_netH')).reset_index()

    #------------------- ROI predicted per batch

    pred_num_den = (test_loans.groupby('batch')
                    .apply(lambda d: wavg_num_denom(d['y_pred_netH'], d['out_fix'])))
    pred_num = np.array([v[0] for v in pred_num_den.values])
    pred_den = np.array([v[1] for v in pred_num_den.values])
    batch_pred_roi = pd.DataFrame({
        'batch': pred_num_den.index.values,
        'pred_roiH': np.where(pred_den > 0, pred_num / pred_den, np.nan)
    })

    #------------------- Batch evaluation vs truth
    eval_net_can = truth[['batch','true_netH']].merge(batch_pred_net, on='batch', how='left')
    eval_roi_can = truth[['batch','true_roiH']].merge(batch_pred_roi, on='batch', how='left')

    rmse_net_can = mean_squared_error(eval_net_can['true_netH'], eval_net_can['pred_netH'], squared=False)
    mae_net_can  = mean_absolute_error(eval_net_can['true_netH'],  eval_net_can['pred_netH'])
    rmse_roi_can = mean_squared_error(eval_roi_can['true_roiH'], eval_roi_can['pred_roiH'], squared=False)
    mae_roi_can  = mean_absolute_error(eval_roi_can['true_roiH'],  eval_roi_can['pred_roiH'])

    sign_true = (eval_roi_can['true_roiH'] > eps).astype(int)
    sign_pred = (eval_roi_can['pred_roiH'] > eps).astype(int)
    acc_sign  = (sign_true == sign_pred).mean()
    bacc      = balanced_accuracy_score(sign_true, sign_pred) if sign_true.nunique() > 1 else np.nan

    #------------------- loan level avaluation
    #------------------- Metric computation

    mask_nc = test_loans['y_pred_netH'].notna() & np.isfinite(test_loans['y_netH'])
    rmse_nc_loan = mean_squared_error(test_loans.loc[mask_nc,'y_netH'],
                                      test_loans.loc[mask_nc,'y_pred_netH'], squared=False)
    mae_nc_loan  = mean_absolute_error(test_loans.loc[mask_nc,'y_netH'],
                                       test_loans.loc[mask_nc,'y_pred_netH'])

    test_loans['roi_true'] = np.where(test_loans['out_fix'] > 0,
                                      test_loans['y_netH'] / test_loans['out_fix'], np.nan)
    test_loans['roi_pred'] = np.where(test_loans['out_fix'] > 0,
                                      test_loans['y_pred_netH'] / test_loans['out_fix'], np.nan)

    mask_roi = test_loans['roi_true'].notna() & test_loans['roi_pred'].notna()
    rmse_roi_loan = mean_squared_error(test_loans.loc[mask_roi,'roi_true'],
                                       test_loans.loc[mask_roi,'roi_pred'], squared=False)
    mae_roi_loan  = mean_absolute_error(test_loans.loc[mask_roi,'roi_true'],
                                        test_loans.loc[mask_roi,'roi_pred'])

    sign_true_loan = (test_loans.loc[mask_roi,'roi_true'] > eps).astype(int)
    sign_pred_loan = (test_loans.loc[mask_roi,'roi_pred'] > eps).astype(int)
    acc_sign_loan  = (sign_true_loan == sign_pred_loan).mean()
    bacc_loan      = balanced_accuracy_score(sign_true_loan, sign_pred_loan) \
                     if sign_true_loan.nunique() > 1 else np.nan

    #------------------- Printing metrics

    print(f"[Loan-level | NET_CASH(H)] RMSE={rmse_nc_loan:.6f} | MAE={mae_nc_loan:.6f}")
    print(f"[Loan-level | ROI(H)] RMSE={rmse_roi_loan:.6f} | MAE={mae_roi_loan:.6f} | "
          f"SignACC={acc_sign_loan:.3f} | BACC={bacc_loan if bacc_loan==bacc_loan else float('nan'):.3f}")
    print('-'*50)
    # Batch-level (canônica)
    print(f"[Batch-level | NET_CASH(H)] RMSE={rmse_net_can:.6f} | MAE={mae_net_can:.6f}")
    print(f"[Batch-level | ROI(H)]  RMSE={rmse_roi_can:.6f} | MAE={mae_roi_can:.6f} | "
          f"SignACC={acc_sign:.3f} | BACC={bacc if bacc==bacc else float('nan'):.3f}")

    metrics = {
        'rmse_nc_loan': rmse_nc_loan,
        'mae_nc_loan': mae_nc_loan,
        'rmse_roi_loan': rmse_roi_loan,
        'mae_roi_loan': mae_roi_loan,
        'acc_sign_loan': acc_sign_loan,
        'bacc_loan': bacc_loan,
        'rmse_net_b': rmse_net_can,
        'mae_net_b': mae_net_can,
        'rmse_roi_b': rmse_roi_can,
        'mae_roi_b': mae_roi_can,
        'acc_sign_b': acc_sign,
        'bacc_b': bacc,
        'comp_metric': rmse_roi_can
    }
    

    return eval_net_can, eval_roi_can, metrics

def scenario2_plots_cash(snap, eval_net_can, eval_roi_can, scenario2_metrics_cash, test_idx, savefig = True, RNG = 42, filtered = ''):
   
    """
    This function is dedicated to plot our model results

    *Inputs:
        - sanp: snapshot
        - eval_net_can: evaluation net cash dataframe
        - eval_roi_can: evaluation ROIh dataframe
        - scenario2_metrics_cashi: metric dictionary
        - test_idx: index of test loans 
        - savefig: if True, the plots are saved to disk
        - filtered: in case you use a filtered dataframe, add a string here to modify the figure name
    """

    #------------------- Loan level data

    loan_plot = (snap.loc[snap['y_pred_netH'].notna(),
                        ['loan_id','batch','out_fix','y_netH','y_pred_netH','y_roiH']]
                    .copy())

    #------------------- Predicted ROI

    loan_plot['y_pred_roiH'] = np.where(loan_plot['out_fix'] > 0,
                                        loan_plot['y_pred_netH'] / loan_plot['out_fix'],
                                        np.nan)

    #------------------- Batch level data

    eval_net_plot = eval_net_can.dropna(subset=['pred_netH']).copy()
    eval_roi_plot = eval_roi_can.dropna(subset=['pred_roiH']).copy()


    #------------------- First plot (loan level net cash)
    
    plt.figure(figsize=(8,6))
    i = 0
    lims_nc_loan = compute_lims(loan_plot['y_netH'], loan_plot['y_pred_netH'])
    for b, g in loan_plot.groupby('batch'):
        i+=1
        plt.scatter(g['y_netH'], g['y_pred_netH'], s=20, color=list(mcolors.TABLEAU_COLORS.keys())[i-1], label='Batch ' + str(i))
    plt.plot(lims_nc_loan, lims_nc_loan, 'k--', lw=1)
    plt.axvline(0, color='k', ls=':', lw=1); plt.axhline(0, color='k', ls=':', lw=1)
    plt.xlim(lims_nc_loan); plt.ylim(lims_nc_loan)
    plt.xlabel('Net cash(H) true (\%)', fontsize = 12)
    plt.ylabel('Net cash(H) pred (\%)', fontsize = 12)
    plt.title(f'Loan-level: net cash (H) -- RMSE = {scenario2_metrics_cash["rmse_nc_loan"]:.0f}')
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig('Figures/'+filtered+'scenario2cash_loan_cash.png', dpi = 300, format = 'png')
    plt.show()

    #------------------- Second plot (loan level ROI)

    loan_plot_roi = loan_plot[np.isfinite(loan_plot['y_pred_roiH']) & (loan_plot['out_fix'] > 0)].copy()
    plt.figure(figsize=(8,6))
    
    x_roi = loan_plot_roi['y_roiH'] * 100.0
    y_roi = loan_plot_roi['y_pred_roiH'] * 100.0
    lims_roi_loan = compute_lims(x_roi, y_roi)
    i = 0
    for b, g in loan_plot_roi.groupby('batch'):
        i+=1
        plt.scatter(g['y_roiH']*100, g['y_pred_roiH']*100, s=20, color=list(mcolors.TABLEAU_COLORS.keys())[i-1], label='Batch ' + str(i))
    plt.plot([-10000, 10000], [-10000, 10000], 'k--', lw=1)
    plt.axvline(0, color='k', ls=':', lw=1)
    plt.axhline(0, color='k', ls=':', lw=1)
    plt.xlim(-120, 120)
    plt.ylim(-120, 120)
    plt.xlabel('ROI(H) true (\%)', fontsize = 12)
    plt.ylabel('ROI(H) pred (\%)', fontsize = 12)
    plt.title(f'Loan-level: ROI(H) -- RMSE = {scenario2_metrics_cash["rmse_roi_loan"]:.6f}')
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig('Figures/'+filtered+'scenario2cash_loan_roi.png', dpi = 300, format = 'png')
    plt.show()

    #------------------- Third plot (batch level net cash)

    plt.figure(figsize=(8,6))
    lims_nc_batch = compute_lims(eval_net_plot['true_netH'], eval_net_plot['pred_netH'])
    i = 0
    for _, r in eval_net_plot.iterrows():
        i+=1
        b = r['batch']
        plt.scatter(r['true_netH'], r['pred_netH'], s=70, color=list(mcolors.TABLEAU_COLORS.keys())[i-1], label='Batch ' + str(i))
        # rótulo opcional
        # plt.annotate(str(b), (r['true_netH'], r['pred_netH']), textcoords="offset points", xytext=(5,3), fontsize=8)
    plt.plot(lims_nc_batch, lims_nc_batch, 'k--', lw=1)
    plt.axvline(0, color='k', ls=':', lw=1); plt.axhline(0, color='k', ls=':', lw=1)
    plt.xlim(lims_nc_batch); plt.ylim(lims_nc_batch)
    plt.xlabel('Net cash(H) true', fontsize = 12)
    plt.ylabel('Net cash(H)  pred', fontsize = 12)
    plt.title(f'Batch-level: net cash (H) -- RMSE = {scenario2_metrics_cash["rmse_net_b"]:.0f}')
    plt.tight_layout()
    if savefig:
        plt.savefig('Figures/'+filtered+'scenario2cash_batch_cash.png', dpi = 300, format = 'png')
    plt.show()

    #------------------- Organizing batch data and computing CI

    loan_pred = (snap.loc[test_idx, ['loan_id','batch','out_fix','y_pred_netH']]
                    .dropna(subset=['y_pred_netH'])
                    .copy())
    loan_pred = loan_pred[loan_pred['out_fix'] > 0]
    loan_pred['roi_pred_loan'] = loan_pred['y_pred_netH'] / loan_pred['out_fix']

    rows = []
    for b, g in loan_pred.groupby('batch'):
        mu, lo, hi = wmean_bootstrap(g['roi_pred_loan'], g['out_fix'], B=2000, seed=123)
        rows.append({'batch': b, 'pred_roiH': mu, 'pred_lo': lo, 'pred_hi': hi})
    pred_boot = pd.DataFrame(rows).dropna()


    plot_df = (eval_roi_can[['batch','true_roiH']]
            .merge(pred_boot, on='batch', how='inner')
            .dropna(subset=['true_roiH','pred_roiH']))

    #------------------- Fourth plot (batch level roi + CI)

    plt.figure(figsize=(8,6))
    i = 0
    for i, r in plot_df.iterrows():
        i+=1
        b = r['batch']
        x = r['true_roiH'] * 100.0
        y = r['pred_roiH'] * 100.0
        yerr = np.array([[y - r['pred_lo']*100.0], [r['pred_hi']*100.0 - y]])  # vertical only
        plt.errorbar(x=x, y=y, yerr=yerr, fmt='o', ms=6,
                    color=list(mcolors.TABLEAU_COLORS.keys())[i-1], elinewidth=1.2, capsize=3, alpha=0.95, label='Batch ' + str (i))

    # identidade e eixos
    plt.plot([-10000, 10000], [-10000, 10000], 'k--', lw=1)
    plt.axvline(0, color='k', ls=':', lw=1)
    plt.axhline(0, color='k', ls=':', lw=1)
    plt.xlim(-17, 17)
    plt.ylim(-17, 17)
    plt.xlabel('ROI(H) true (\%)', fontsize = 12)
    plt.ylabel('ROI(H) pred (\%)', fontsize = 12)
    plt.title(f'Batch-level: ROI(H) -- RMSE = {scenario2_metrics_cash["rmse_roi_b"]:.6f}')
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig('Figures/'+filtered+'scenario2cash_batch_roi.png', dpi = 300, format = 'png')
    plt.show()

def scenario2_pre_err_cash(snap, eval_roi_can, test_idx, RNG = 42):
    loan_pred = (snap.loc[test_idx, ['loan_id','batch','out_fix','y_pred_netH']]
                    .dropna(subset=['y_pred_netH'])
                    .copy())
    loan_pred = loan_pred[loan_pred['out_fix'] > 0]
    loan_pred['roi_pred_loan'] = loan_pred['y_pred_netH'] / loan_pred['out_fix']

    rows = []
    for b, g in loan_pred.groupby('batch'):
        mu, lo, hi = wmean_bootstrap(g['roi_pred_loan'], g['out_fix'], B=2000, seed=123)
        rows.append({'batch': b, 'pred_mu': mu, 'pred_lo': lo, 'pred_hi': hi})
    pred_boot = pd.DataFrame(rows).dropna()


    plot_df = (eval_roi_can[['batch','true_roiH']]
            .merge(pred_boot, on='batch', how='inner')
            .dropna(subset=['true_roiH','pred_mu']))

    return plot_df