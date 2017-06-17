import scipy.stats as stats
import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("paths_to_progress_csvs", nargs="+", help="All the csvs")
parser.add_argument("--range_start", type=int, default=-1)
parser.add_argument("--range_end", type=int, default=100000000)


args = parser.parse_args()
assert len(args.paths_to_progress_csvs) == 2

avg_rets = []
std_dev_rets = []
trajs = []

data = pd.read_csv(args.paths_to_progress_csvs[0])

a_means = data["AverageReturn"][max(args.range_start,0):min(args.range_end, len(data["AverageReturn"]))]
a_stds = data["StdReturn"][max(args.range_start,0):min(args.range_end, len(data["AverageReturn"]))]
n_as = data["NumTrajs"][max(args.range_start,0):min(args.range_end, len(data["AverageReturn"]))]

args.paths_to_progress_csvs
data = pd.read_csv(args.paths_to_progress_csvs[1])

b_means = data["AverageReturn"][max(args.range_start,0):min(args.range_end, len(data["AverageReturn"]))]
b_stds = data["StdReturn"][max(args.range_start,0):min(args.range_end, len(data["AverageReturn"]))]
n_bs = data["NumTrajs"][max(args.range_start,0):min(args.range_end, len(data["AverageReturn"]))]

# Do a T - test
ts, ps = [],[]

for a_mean, a_std, n_a, b_mean, b_std, n_b in zip(a_means, a_stds, n_as, b_means, b_stds, n_bs):
    t, p = stats.ttest_ind_from_stats(a_mean, a_std, n_a, b_mean, b_std, n_b, equal_var=False)
    ts.append(t)
    ps.append(p)

print("t=%f,p=%f" % (np.mean(ts), np.mean(ps)))
