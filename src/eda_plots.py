import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.lines import Line2D


"""
This file is dedicated to host the plotting functions used in the EDA notebook.
"""

def plot_mean_roi_since_allow(df, rolling=7, min_batches=2, level=0.95, shaded=True, plot = True
):
    
    """
    This function computes the mean ROI between all batches in our dataset, as well as the 95% CI of the curve.

    *Inputs:
        - df: loan level dataframe
        - rolling: parameter used for suavization of the mean by the computation of a rolling average of the ROI over 'rolling' days.
        - min_batches: minimum number of batches required to compute the average
        - level: level of the CI
        - shaded: specifies if the error will be shown as a shaded region (True) or as vertical error bars (False)
        - plot: if True, the function will automatically plot the mean ROI curve.

    *Outputs:
        - x: days since batch is allowed
        - y_plot: average computed between all batches
        - lo_plot: lower boundary of the CI
        - hi_plot: higher boundary of the CI

    -> Besides returning the four previously mentioned outputs, this function automatically plots the ROI curve if plot = True
    """
    df_ = df.copy()

    # ---------- Agregating df_ from loan to batch level
    cf_batch = (df_.groupby(['batch','date'], as_index=False)
                  .agg(inflow=('inflow','sum'),
                       outflow=('outflow','sum'),
                       billing=('billing','sum'))
                  .sort_values(['batch','date'], kind='mergesort'))

    cf_batch['cum_in']   = cf_batch.groupby('batch', sort=False)['inflow'].cumsum()
    cf_batch['cum_out']  = cf_batch.groupby('batch', sort=False)['outflow'].cumsum()
    cf_batch['cum_bill'] = cf_batch.groupby('batch', sort=False)['billing'].cumsum()
    den = cf_batch['cum_out'].replace(0, np.nan)
    cf_batch['roi_cum_batch_weighted'] = (cf_batch['cum_in'] + cf_batch['cum_bill'] - cf_batch['cum_out']) / den
    cf_batch['roi_cum_batch_weighted'] = cf_batch['roi_cum_batch_weighted'].fillna(-1.0)

    # ---------- Computing the days since each batch was allowed
    allow_map = (df_.drop_duplicates('batch')[['batch','allowlisted_date']]
                   .set_index('batch')['allowlisted_date'])
    cf_batch['allowlisted_date'] = cf_batch['batch'].map(allow_map)
    cf_batch['days_since_allowed'] = (cf_batch['date'] - cf_batch['allowlisted_date']).dt.days

    # mantém apenas valores válidos
    valid = np.isfinite(cf_batch['roi_cum_batch_weighted'])
    cfv = cf_batch.loc[valid, ['batch','days_since_allowed','roi_cum_batch_weighted']]

    # ---------- We take the ROI per day
    per_day = (
        cfv.groupby('days_since_allowed')['roi_cum_batch_weighted']
                 .apply(lambda s: np.asarray(s.values, dtype=float))
                 .reset_index(name='vals')
                 )

    # ---------- Filter by the 'min_batches' parameter
    per_day['n'] = per_day['vals'].apply(len)
    per_day = per_day[per_day['n'] >= int(min_batches)].sort_values('days_since_allowed')
    x = per_day['days_since_allowed'].to_numpy()
    n = per_day['n'].to_numpy()

    # ---------- Central point taken by computing the mean
    mu = np.array([v.mean() for v in per_day['vals']])

    lo = hi = None

    # ---------- CI: mean ± z * (sd/sqrt(n))
    z = norm.ppf(0.5 + level/2.0)
    sd = np.array([v.std(ddof=1) if len(v) > 1 else 0.0 for v in per_day['vals']])
    sem = np.where(n > 0, sd / np.sqrt(n), np.nan)
    lo = mu - z * sem
    hi = mu + z * sem

    # ---------- Optional rolling window suavization. Set rolling = 0/None to not include it
    def _roll(a):
        if rolling and int(rolling) > 0:
            r = int(rolling)
            return pd.Series(a).rolling(r, min_periods=max(1, r//2)).mean().to_numpy()
        return a

    y_plot = _roll(mu)
    lo_plot = _roll(lo) if lo is not None else None
    hi_plot = _roll(hi) if hi is not None else None

    # ---------- plot ----------
    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(x, y_plot*100.0, lw=2, label='Average ROI')
        if (lo_plot is not None) and (hi_plot is not None):
            if shaded:
                plt.fill_between(x, lo_plot*100.0, hi_plot*100.0, alpha=0.18,
                                label=f"IC {int(level*100)}\%")
            else:
                # barras de erro verticais (pode ficar poluído se houver muitos pontos)
                yerr = np.vstack([(y_plot - lo_plot)*100.0, (hi_plot - y_plot)*100.0])
                plt.errorbar(x, y_plot*100.0, yerr=yerr, fmt='none', alpha=0.5, capsize=2)

        plt.axhline(0, color='k', ls='--', lw=1)
        plt.axvline(0, color='k', ls=':', lw=1)
        plt.title("Average ROI between all batches")
        plt.xlabel("Days since batch was allowed", fontsize = 12)
        plt.ylabel("ROI (\%)", fontsize = 12)
        plt.legend()
        plt.tight_layout()
        plt.savefig('Figures/'+'batch_mean_roi.png', dpi = 300, format = 'png')
        plt.show()

    return x, y_plot, lo_plot, hi_plot

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

def plot_batch_loan_history(df, batch_list, savefig = True):

    """
    This function is dedicated to plot the loal leven history of a batch.
    It will output one figure for each batch in our dataset, and each figure will be composed of two plots
    -> Plot 1: net cash of each loan vs. date
    -> plot 2: ROI of each loan vs date

    *Inputs:
        df: loan level dataframe
        batch_list: list with the batch name for which you desire a plot
    """

    for i in tqdm(range(len(batch_list))):
        batch_ = batch_list[i]
        fig, axs = plt.subplots(2, sharex=True)
        df_ = df.query('batch == @batch_')
        loans_ids = df_['loan_id'].unique()

        for b in loans_ids:
            c = df_[df_['loan_id']==b]

            axs[0].plot(c['date'], c.sort_values('date')['net_cash'], alpha = 0.3, color = list(mcolors.TABLEAU_COLORS.keys())[i])
            axs[1].plot(c['date'], c.sort_values('date')['roi_cum'], alpha = 0.3, color = list(mcolors.TABLEAU_COLORS.keys())[i])

        axs[0].axhline(0, color='k', ls='--')
        axs[1].axhline(0, color='k', ls='--')
        axs[0].grid()
        axs[1].grid()
        axs[0].set_ylabel(r'Net Cash', fontsize = 12)
        axs[1].set_ylabel(r'Roi (\%)', fontsize = 12)
        plt.title('Batch ' + str(i+1))
        plt.xlabel('Date', fontsize = 12)

        for ax in axs:
            ax.label_outer()
        plt.tight_layout()
        if savefig:
            plt.savefig('Figures/'+'batch_' + str(i+1) + '_loan_history.png', dpi = 300, format = 'png')
        plt.show()

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

def corr_loan_params_final_day(df, cols = None, plot = True):

    """
    This function computes the correlation of a set of features of each loan on their last day
    The last day of a loan is either the day on which it was completely repaid or simply the last day they appear
    in the repayments database
    The objective is to check if there is any clear relation between some of the parameters engineered from the data

    *Inputs:
        - df: loan level dataframe
        - cols: list with the names of the columns of the dataframe df in which you are interested in checking for correlation
            -> If cols = None, this function will try to use a placeholder set of columns defined bellow.
        - plot: if plot = True, this function will also automatically plot the correlation matrix
    *Outputs:
        - corr: correlation matrix between the features

    Beyond corr, this function also automatically plots the correlation matrix if plot = True
    """

    #-------------- Taking the last day of each loan
    last_idx = df.loc[df['date'].notna()].groupby('loan_id')['date'].idxmax()
    last_rows = df.loc[last_idx].sort_values('loan_id').reset_index(drop=True)

    if cols == None:
        columns = ['outflow', 'inflow', 'billing', 'cum_in', 'cum_bill', 'out', 'roi_cum', 'net_in', 'net_cash', 'days_since_allowed', 'days_per_loan', 'roi_cum_batch_weighted']
    else:
        columns = cols

    #-------------- Computing correlation
    corr = last_rows[columns].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    if plot:
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            corr, mask = mask,
            vmin=-1, vmax=1, center=0, cmap='coolwarm',
            square=True, annot=True, fmt='.2f', linewidths=.5,
            cbar_kws={'label': 'Correlation'}
        )

        ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=13)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.yaxis.label.set_size(13)

        plt.tight_layout()
        plt.show()

    return corr


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------


def plot_hist_days(loan_df, loan_df_filtered, batches):

    """
    This function is dedicated to produce two distribution plots
        1: Distribution of the loan duration
        2: Distribution of how long did it take for a the loan to be created since it was allowed

    This function shows these distribution plots comparing two different datasets: a full dataset, composed of all loans
    and a filtered dataset, from which the loans with the largest loan_amount were removed

    *Inputs:
        - loan_df: complete loan dataframe
        - loan_df_filtered: loan dataframe with the most expensive loans removed
        - batches: list of batch names
    """

    df = loan_df.copy()
    df2 = loan_df_filtered.copy()

    tab_colors = list(mcolors.TABLEAU_COLORS.keys())  # ['tab:blue', 'tab:orange', ...]

    for i in tqdm(range(len(batches))):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        batch_ = batches[i]
        df_  = df.query('batch == @batch_')
        df2_ = df2.query('batch == @batch_')
        col = tab_colors[i]

        # --- apenas KDE (barras invisíveis) + estilo via kde_kws ---
        # Painel 1: days_per_loan
        sns.histplot(df_['days_per_loan'], ax=axs[0], stat='percent', bins=40,color=col, kde=True, alpha=0, edgecolor=None,
            line_kws={'linewidth':2, 'color':col}
        )
        sns.histplot(
            df2_['days_per_loan'], ax=axs[0], stat='percent', bins=40, color=col, kde=True, alpha=0, edgecolor=None,
            line_kws={'linestyle':'--',  'linewidth':2, 'color':col}
        )

        # Painel 2: days_since_allowed
        sns.histplot(df_['days_since_allowed'], ax=axs[1], stat='percent', bins=40, color=col, kde=True, alpha=0, edgecolor=None,
            line_kws={'linewidth':2, 'color':col}
        )
        sns.histplot(
            df2_['days_since_allowed'], ax=axs[1], stat='percent', bins=40, color=col, kde=True, alpha=0, edgecolor=None,
            line_kws={'linestyle':'--',  'linewidth':2, 'color':col}
        )

        axs[0].text(0.7, 0.95, s = r'C-DF: 75\% of loans last up to ' + str(int(df_['days_per_loan'].describe()['75%'])) + ' days',
            ha='center', va='center', transform=axs[0].transAxes, fontsize=12)
        axs[0].text(0.7, 0.85, s = r'F-DF: 75\% of loans last up to ' + str(int(df2_['days_per_loan'].describe()['75%'])) + ' days',
            ha='center', va='center', transform=axs[0].transAxes, fontsize=12)
        
        axs[1].text(0.55, 0.95,s = r'C-DF: 75\% of loans are taken up to ' + str(int(df_['days_since_allowed'].describe()['75%'])) + ' days after allowed',
            ha='center', va='center', transform=axs[1].transAxes, fontsize=12)
        axs[1].text(0.55, 0.85,s = r'F-DF: 75\% of loans are taken up to ' + str(int(df2_['days_since_allowed'].describe()['75%'])) + ' days after allowed',
            ha='center', va='center', transform=axs[1].transAxes, fontsize=12)
        
        # Eixos/labels
        axs[0].set_xlabel('Loan length (days)', fontsize=12)
        axs[1].set_xlabel('Days until taking loan', fontsize=12)
        axs[0].set_ylabel('Percent', fontsize=12)
        axs[1].set_ylabel('Percent', fontsize=12)

        # Legenda consistente (linhas somente)
        handles = [
            Line2D([0], [0], color=col, lw=2, ls='-',  label='Complete DF'),
            Line2D([0], [0], color=col, lw=2, ls='--', label='Filtered DF'),
        ]
        axs[0].legend(handles=handles, loc='center right')
        axs[1].legend(handles=handles, loc='center right')

        plt.tight_layout()
        plt.show()