import numpy as np
def read_data():
    raw_data=np.loadtxt("task1a_lm1d3za/train.csv",dtype=np.str,delimiter=",")
    data=raw_data[1:,2:].astype(np.float)
    label=raw_data[1:,1].astype(np.float)
    return data,label
