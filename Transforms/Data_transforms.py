import numpy as np


def group_U(dataframe,N_points=3):
    X_r_i=(
        lambda u,x,d: (
        np.concatenate((np.concatenate(*u),np.array(d["U"]))),
        np.concatenate((np.concatenate(*x),np.array(d["X"])))
        )
          )
    f=(lambda d:X_r_i(*tuple(zip(dataframe.sample(n=N_points,replace=True)[["U","X"]].values.T)),d))
    dataframe["U"],dataframe["X"]=zip(*(dataframe[["U","X"]].apply(f,axis=1)))
    return dataframe