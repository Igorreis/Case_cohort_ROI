import sys
sys.path.append('..')  # Adjust the path as needed

from src.read_data import *
from src.frame_creation import *
from src.scenario2_modeling_functions import *
from src.scenario1_modeling_functions import *
from src.run_models import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

H = 730
t = 200
db_path = "../database.db"


print('***Reading database and building DataFrames***')


allowlist, loans, repayments = load_tables("database.db")

loan_CF = loan_cum_frame(allowlist, loans, repayments)
CF = rebuild_CF_from_loanCF(loan_CF)
loan_CF_filtered, removal_summary, batch_thresholds = remove_top_loans_per_batch(loan_CF,p=0.90)

batches = allowlist['batch'].unique()

scenario1_out, scenario1_all_met, scenario1_Model_n = run_scenario1(loan_CF, loan_CF_filtered, batches, H, t)
scenario2_out, scenario2_all_met, scenario2_Model_n = run_scenario2(loan_CF, loan_CF_filtered, batches, H, t)

all_out = scenario1_out + scenario2_out

all_met = scenario1_all_met + scenario2_all_met

all_Model = scenario1_Model_n + scenario2_Model_n

metrics = [metric['comp_metric'] for metric in all_met]

print('------------------------------------------------------------------------------')
print('BEST MODEL:')
print('------------>' + all_Model[np.argmin(metrics)])

print('------------------------------------------------------------------------------')

print(all_out[np.argmin(metrics)].to_string())


