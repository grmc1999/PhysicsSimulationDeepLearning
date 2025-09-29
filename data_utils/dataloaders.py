import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from einops import rearrange, repeat
import torch
import h5py
from functools import reduce
import glob


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
        if training_mode=="sequence_prediction":
            assert prediction_horizon!=None
            assert sequence_dim!=None
    def get_sequence(self,idx):
        pass
    def get_point(self,idx):
        pass
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
    def __init__(self,
                 path,
                 training_mode,
                 prediction_horizon,
                 sequence_dim,
                 simulation_time_step,
                 start_time=1,
                 position_fields=["GridCentroidX","GridCentroidY"],
                 files_extension="h5"):
        """
        this function samples as tensor with data from measurements
        tensor dimensions should be flexible
            - "x y z t"
            - "x y z [t_1 ... t_n]"
            - "x y [t_1 ... t_n]"

        start_time: first step of the simulation (for now just useful to parse GEM simulations)
        position_fields: fields that contain real coordinates  (for now just useful to parse GEM simulations)
        """
        self.dt=simulation_time_step
        self.t0 = start_time
        if training_mode=="sequence_prediction":
            assert prediction_horizon!=None
            self.pred_h=prediction_horizon
            assert sequence_dim!=None
            self.seq_dim=sequence_dim
        if path.split(".")[-1]==files_extension:
            self.data=path
        else:
            self.data=glob.glob(os.path.join(path,f"*.{files_extension}"))
#        self.get_prop = lambda com_prop: com_prop.split("_")[0]
        self.get_df_props_names = lambda dataFrame: (np.unique(np.array(list(map(lambda com_prop: com_prop.split("_")[0],list(dataFrame.keys())[:])))))

        self.position_fields=position_fields

    def get_one_dataFrame(self,filename,field):
        f = h5py.File(filename)
        group = f[field]
        item = group["block0_items"][()]
        data = group["block0_values"][()]
        cols_list=list(map(lambda c: c.decode("utf-8"),item.tolist()))
        return pd.DataFrame(data=data,columns=cols_list)

    def h52dataFrames(self,h5file):
        df_features=self.get_one_dataframe(h5file,"features")
        df_outout=self.get_one_dataframe(h5file,"outout")
        return df_features,df_outout

    def dataFrame2Tensor(self,dataFrames):
        """
        TODO: think in optimization to avoid loading the entire dataset
        at this point the time step has already been defined in seconds
        """
        X = pd.DataFrame(np.unique(dataFrames[0][list(dataFrames[0].keys())[3:]].values,axis=0), columns=list(dataFrames[0].keys())[3:])
        dynamic_prop_ref=list(X.keys())[15:]

        dynamic_prop=list(np.unique(np.array(list(map(lambda com_prop: com_prop.split("_")[0],dynamic_prop_ref)))))
        tf=np.max(np.unique(np.array(list(map(lambda com_prop: float(com_prop.split("_")[1]),dynamic_prop_ref)))))
        static_prop=list(X.keys())[:15]
        # Processing of dependent variables, in space and time
        Xs=X[static_prop]
        Xt=X[dynamic_prop_ref]

        X=repeat(Xs.values[:,:],"p v -> t p v",t=int(tf))
        T=repeat(np.array([t for t in range(1,int(tf)+1)]),"t -> t p 1",p=X.shape[1])
        XT=np.concatenate([X,T],axis=-1)
        
        SVP=np.stack(list(map(lambda t: Xt[list(map(lambda p: p+"_"+str(int(t)),dynamic_prop))].values,np.arange(1,tf+1))),axis=2)
        Xts=rearrange(SVP,"p v t-> (t p) v")
        XT=rearrange(XT,"t p v -> (t p) v")
        
        Xst=pd.DataFrame(
            np.concatenate([XT,Xts],axis=-1),
            columns= static_prop + ["t"] + dynamic_prop
            )

        
        BC=dataFrames[0][list(dataFrames[0].keys())[:6]+ self.position_fields [dataFrames[0]["completation"]==1.0]]
        U=dataFrames[1]
        U=U.T[BC["well"].values.astype(int)-1].T
        
        u_dynamic_prop_ref=list(U.keys())
        u_dynamic_prop=list(np.unique(np.array(list(map(lambda com_prop: com_prop.split("_")[0],u_dynamic_prop_ref)))))
        SVP=np.stack(list(map(
                          lambda t: U[list(map(lambda p: p+"_"+str(int(t)),u_dynamic_prop))].values,
                          np.arange(1,tf+1))),axis=2)
        
        X=repeat(BC[["X","Y","Z"]+ self.position_fields ].values[:,:],"p v -> t p v",t=int(tf))
        T=repeat(np.array([t for t in range(1,int(tf)+1)]),"t -> t p 1",p=X.shape[1])
        XT=np.concatenate([X,T],axis=-1)
        XT=rearrange(XT,"t p v -> (t p) v")
        Uts=rearrange(SVP,"p v t-> (t p) v")
        
        Ust=pd.DataFrame(
            np.concatenate([XT,Uts],axis=-1),
            columns= ["X","Y","Z"] + self.position_fields + ["t"] + u_dynamic_prop
            )
        return Xst,Ust



    def is_data_list(self):
        if isinstance(self.data,list):
            return True
        else:
            return False
    
    def __len__(self):
        pass
    def __getitem__(self,idx):
       """
       Depending on the method this function should return a point, a sequence or an entire case of simulation
       """
       pass

class pandasDataset(h5Dataset):
    def __init__(self,**args):
        super().__init__(**args,files_extension="csv")

class pointbasedh5Dataset(h5Dataset):
    def __init__(self,**args):
        super().__init__(**args)
    
    def __len__(self):
        # get list of files
        if self.is_data_list():
            return reduce(list(map(lambda d: len(self.dataFrame2Tensor(self.h52dataFrames(d))[0]),self.data)),lambda x,y:x+y)
        else:
            Xst,Ust=self.dataFrame2Tensor(self.h52dataFrames(self.data))
            return len(Xst)
        
    def __getitem__(self,idx):
        if self.is_data_list():
            data_frame_list_counts = np.cumsum(
                    np.array(
                        list(map(lambda d: len(self.dataFrame2Tensor(self.h52dataFrames(d))[0]),self.data))
                        )
                    )
            mod_idx = idx - data_frame_list_counts[np.sum(data_frame_list_counts < idx)]
            x=self.dataFrame2Tensor(self.h52dataFrame(self.data[np.sum(data_frame_list_counts < idx) - 1 ]))[0].T[mod_idx]
            u=self.dataFrame2Tensor(self.h52dataFrame(self.data[np.sum(data_frame_list_counts < idx) - 1 ]))[1].T[mod_idx]
            return x,u
        else:
            Xst,Ust=self.dataFrame2Tensor(self.h52dataFrames(self.data))[0]
            return Xst.T[idx],Ust.T[idx]

class pointbasedPandasDataset(pandasDataset):
    def __len__(self):
        # get list of files
        if self.is_data_list():
            return reduce(list(map(lambda d: len(pd.read_csv(d)),self.data)),lambda x,y:x+y)
        else:
            return len(pd.read_csv(self.data))
        
    def __getitem__(self,idx):
        if self.is_data_list():
            data_frame_list_counts = np.cumsum(
                    np.array(
                        list(map(lambda d: len(pd.read_csv(d)),self.data))
                        )
                    )
            mod_idx = idx - data_frame_list_counts[np.sum(data_frame_list_counts < idx)]
            x=pd.read_csv(self.data[np.sum(data_frame_list_counts < idx) - 1 ]).T[mod_idx]
            return x
        else:
            Xst = pd.read_csv(self.data)
            return Xst.T[idx]

class simulationbasedh5Dataset(h5Dataset):
    def __init__(self,**args):
        super().__init__(**args)
        assert isinstance(self.data)
    
    def __len__(self):
        # get list of files
        if self.is_data_list():
            return len(self.data)
        
    def __getitem__(self,idx):
        if self.is_data_list():
            Xst,Ust=self.dataFrame2Tensor(self.h52dataFrames(self.data[idx]))
            return Xst.T,Ust.T
        else:
            Xst,Ust=self.dataFrame2Tensor(self.h52dataFrames(self.data))
            return Xst.T,Ust.T
