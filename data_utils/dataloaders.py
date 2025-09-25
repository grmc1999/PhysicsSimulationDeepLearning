import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from einops import rearrange, repeat
import torch
import h5py


class PDEDataset(Dataset):
    def __init__(self,training_mode,prediction_horizon,sequence_dim):
        """
        NOTE: at this point data should be expressed in a 3-dimensional structure with defined boundary points

        training_mode: defines the mode in which __getitem__ will yield the data
            'state_prediction'
            'sequence_prediction': yields time or space contiguous data from random starting point to defined horizon
        prediction_horizon: prediction horizon for sequence_prediction
        sequence_dimension: dimension over which the sequence will be taken (this is a "generalization" for time sequences for position (useful for ROM))
        """
        if training_model=="sequence_prediction":
            assert prediction_horizon!=None
            assert sequence_dim!=None
    def get_sequence(self,idx):

    def get_point(self,idx):
    def parse_BC(self):
        """
        input: expressions for functions of BC
        output: assert that output is a function
        """
        pass
    def check_pde(self):
        """
        input: expressions for functions of PDE
        output: assert that output is a function
        """
        pass

class h5Dataset(Dataset):
    def __init__(self,path,training_mode,prediction_horizon,sequence_dim,simulation_time_step):
        """
        this function samples as tensor with data from measurements
        tensor dimensions should be flexible
            - "x y z t"
            - "x y z [t_1 ... t_n]"
            - "x y [t_1 ... t_n]"
        """
        self.dt=simulation_time_step
        if training_model=="sequence_prediction":
            assert prediction_horizon!=None
            self.pred_h=prediction_horizon
            assert sequence_dim!=None
            self.seq_dim=sequence_dim
        if path.split(".")[-1]=="h5":
            self.data=path
        else:
            self.data=glob.glob(os.path.join(path,"*.h5"))
#        self.get_prop = lambda com_prop: com_prop.split("_")[0]
        self.get_df_props_names = lambda dataFrame: (np.unique(np.array(list(map(lambda com_prop: com_prop.split("_")[0],list(dataFrame.keys())[:])))))

    def get_one_dataFrame(self,field):
        group = f[field]
        item = group["block0_items"][()]
        data = group["block0_values"][()]
        cols_list=list(map(lambda c: c.decode("utf-8"),item.tolist()))
        return pd.DataFrame(data=data,columns=cols_list)

    def h52dataFrames(self,h5file):
        df_features=self.get_one_dataframe("features")
        df_outout=self.get_one_dataframe("outout")
        return df_features,df_outout
    
    def 

    def dataFrame2Tensor(self,dataFrames):
        """
        at this point the time step has already been defined in seconds
        """
        dynamic_props = np.unique(np.array(list(map(lambda com_prop: com_prop.split("_")[0],list(dataFrame[0].keys())[18:]))))
        static_props = list(dataFrame[0].keys())[18:])
        well_props = np.unique(np.array(list(map(lambda com_prop: com_prop.split("_")[0],list(dataFrame[1].keys())[:]))))

        #XYKIJ=features_data[["GridCentroidX","GridCentroidY","Z","PermeabilityI","PermeabilityJ","PermeabilityK"]].values
        XYKIJ=features_data[self.selectec_static].values
        XYKIJ=features_data[self.selectec_static].values
        SVP=np.stack(list(map(lambda t: features_data[list(map(lambda p: p+"_"+str(t),dynamic_props))].values,np.arange(1,241))),axis=2)



    def __len__(self):
        if isinstance(self.data,list):
            len(self.data)*
        else:
    def __getitem__(self,idx):
        

class pandasDataset(Dataset):
???END
