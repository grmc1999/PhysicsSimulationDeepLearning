import torch
import pandas as pd
import numpy as np


class Trainer(object):
    def __init__(self,model_instance,data_path,batch_size,optimizer):
        self.model=model_instance
        self.data=data_path
        self.batch_size=batch_size

        self.transform_U=(lambda d:(d['u']))
        self.transform_X=(lambda d:[d['x'],d['y']])
        self.data=self.data_preprocessing()
        self.optimizer=optimizer

        #self.transform=lambda d:(d['u'],[d['x'],d['y']])
        

    def train(self):
        losses=[]
        self.data.sample(frac=1)
        #for U,X in self.data:
        for i in range(len(self.data)//self.batch_size):
            U=self.data["U"][i*self.batch_size:(i+1)*self.batch_size]
            X=self.data["X"][i*self.batch_size:(i+1)*self.batch_size]
            u=torch.tensor(np.expand_dims(U.values,axis=0).T,dtype=torch.float)
            x=torch.tensor(np.stack(X.values),requires_grad=True,dtype=torch.float)
            total_loss=self.model.compute_loss(x,u)

            loss=total_loss["total_loss"]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            list(total_loss[k].cpu().detach() for k in total_loss.keys())
            losses.append(total_loss)
        return losses

    #def test(self):
    #def train_test(self,epochs):
    #def checkpoint_model(self):
    #def load_model(self):
    def data_preprocessing(self):
        data=pd.read_csv(self.data)
        data["U"]=data.apply(self.transform_U ,axis=1)
        data["X"]=data.apply(self.transform_X ,axis=1)
        return data
        #return pd.read_csv(self.data)
