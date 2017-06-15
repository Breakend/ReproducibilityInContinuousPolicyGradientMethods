import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from itertools import cycle

from numpy import genfromtxt


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("csvs_to_compile", nargs='+', help="The csvs to compile")
parser.add_argument("ave_out", help="the output file")

args = parser.parse_args()

data_frames = []
for f in args.csvs_to_compile:

    data = pd.read_csv(f)
    data_frames.append(data)


df = pd.concat(data_frames, axis=1)
# import pdb; pdb.set_trace()

# df = df.swaplevel(0, 1, axis=1).sortlevel(axis=1)
foo = df.groupby(level=0, axis=1).mean()

foo.to_csv(args.ave_out)
