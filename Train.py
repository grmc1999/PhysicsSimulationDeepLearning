import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from DL_models.Models.GAN import *
from DL_models.Models.PINN import *
from DL_models.PINNS.utils import derivatives, vector_jacobian, vector_grad, x_grad
from DL_models.PINNS.Residual_losses import incompresibble_fluid_loss,two_phase_darcy_flow_loss,one_phase_darcy_flow_loss,incompresibble_fluid_3D_loss

from Transforms.Data_transforms import *

import fire
import json

class Trainer(object):
    def __init__(self,model_instance,data_path,batch_size,optimizer=None,data_dir=None,scope_agent=None,scope_loss=None,fraction_list=[0.8],
    transform_U=(lambda d:([d['u']])),transform_X=(lambda d:[d['x'],d['y']]),dataset_trasnform=None,device="cpu"
    ):
        self.model=model_instance
        self.data=data_path
        self.batch_size=batch_size

        self.transform_U=(eval(transform_U) if isinstance(transform_U,str) else transform_U)
        self.transform_X=(eval(transform_X) if isinstance(transform_X,str) else transform_X)
        self.dataset_trasnform=(eval(dataset_trasnform) if isinstance(dataset_trasnform,str) else dataset_trasnform)
        self.data=self.data_preprocessing()
        self.data_test=self.data_train=self.data
        self.optimizer=optimizer

        self.scope_agent=scope_agent
        self.scope_loss=scope_loss
        self.data_dir=data_dir
        self.fraction_list=fraction_list

        self.model.to(device)
        self.device=device

        #self.transform=lambda d:(d['u'],[d['x'],d['y']])
        

    def get_batch_mean(self,losses_dict):
        return np.mean(np.array(list(map( lambda d:d[self.scope_loss] ,losses_dict))))

    def train(self):
        losses=[]
        self.data_train.sample(frac=1)
        #for U,X in self.data:
        for i in range(len(self.data_train)//self.batch_size):
            U=self.data_train["U"][i*self.batch_size:(i+1)*self.batch_size]
            X=self.data_train["X"][i*self.batch_size:(i+1)*self.batch_size]
            u=torch.tensor(np.stack(U.values,axis=0),dtype=torch.float).to(self.device)
            x=torch.tensor(np.stack(X.values),requires_grad=True,dtype=torch.float).to(self.device)
            self.model.train()
            total_loss=self.model.compute_loss(x,u)

            loss=total_loss["total_loss"]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for k in total_loss.keys():
                total_loss[k]=total_loss[k].cpu().detach()

            losses.append(total_loss)
        return losses

    def test(self):
        losses=[]
        self.data_test.sample(frac=1)
        #for U,X in self.data:
        for i in range(len(self.data_test)//self.batch_size):
            U=self.data_test["U"][i*self.batch_size:(i+1)*self.batch_size] # [batch ]
            X=self.data_test["X"][i*self.batch_size:(i+1)*self.batch_size]
            u=torch.tensor(np.stack(U.values,axis=0),dtype=torch.float).to(self.device)
            x=torch.tensor(np.stack(X.values),requires_grad=True,dtype=torch.float).to(self.device)
            self.model.eval()
            total_loss=self.model.compute_loss(x,u)

            loss=total_loss["total_loss"]

            for k in total_loss.keys():
                total_loss[k]=total_loss[k].cpu().detach()
            losses.append(total_loss)
        return losses
    def epochs_train_test(self,epochs,pref=""):
        losses={}
        best_result=1e10
        for epoch in tqdm(range(epochs)):
            train_losses=self.train()
            test_losses=self.test()
            losses.update({epoch:{
                "test":test_losses,
                "train":train_losses
                }})

            batch_mean=self.get_batch_mean(test_losses)

            # Save best model
            if batch_mean<best_result:

                best_result=batch_mean
                best_model=self.model.state_dict()
                torch.save(best_model,"{fname}.pt".format(fname=os.path.join(self.data_dir,"best")))

            # Save losses
            np.save(os.path.join(self.data_dir,pref+"loss_results"+'.npy'),losses)

            # 
            
            # Schedule functions
        return losses

    def data_size_test(self,epochs):
        torch.save(self.model.state_dict(),"{fname}.pt".format(fname=os.path.join(self.data_dir,"initial_state")))
        for percentaje in tqdm(self.fraction_list):
            self.model.load_state_dict(torch.load("{fname}.pt".format(fname=os.path.join(self.data_dir,"initial_state")), weights_only=True))
            tqdm.write("training with "+str(percentaje))
            self.data_train=self.data[:int(len(self.data)*percentaje)]
            tqdm.write("train size: "+str(len(self.data_train)))
            self.data_test=self.data[int(len(self.data)*percentaje):]
            tqdm.write("test size: "+str(len(self.data_test)))
            self.epochs_train_test(epochs,pref=str(int(percentaje*100))+"_")
            
    #def checkpoint_model(self):
    #def load_model(self):
    def data_preprocessing(self):
        """
        Apply transforms for U and X
        both transforms should map tuple to list as (x,y,z) --> [[x,y,z]] # [n_points=1,coordinates]
        U: (u_1,u_2,...,u_n) --> [[u_1,u_2,...,u_n]]
        X: (x,y,z) --> [[x,y,z]] (common)
        """
        data=pd.read_csv(self.data)
        data["U"]=data.apply(self.transform_U ,axis=1)
        data["X"]=data.apply(self.transform_X ,axis=1)
        if self.dataset_trasnform!= None:
            data=self.dataset_trasnform(data)
        return data
        #return pd.read_csv(self.data)

class Dual_optimizer_trainer(Trainer):

    #def __init__(self,model_instance,data_path,batch_size,optimizer,sub_steps,data_dir=None,scope_agent=None,scope_loss=None,fraction_list=[0.8]):
    def __init__(self,model_instance,data_path,batch_size,optimizer,sub_steps,**args):
    #    super().__init__(model_instance,data_path,batch_size,data_dir=data_dir,scope_agent=scope_agent,scope_loss=scope_loss,fraction_list=fraction_list)
        super().__init__(model_instance,data_path,batch_size,**args)

        self.discriminator_optimizer=optimizer[0]
        self.generator_optimizer=optimizer[1]
        self.discriminator_sub_steps=sub_steps[0]
        self.generator_sub_steps=sub_steps[1]

        

    def get_batch_mean(self,losses_dict):
        return np.mean(np.array(list(map(lambda d:d[self.scope_loss],losses_dict[self.scope_agent]))))

    def train(self):
        losses_dis=[]
        losses_gen=[]
        self.data_train.sample(frac=1)
        #for U,X in self.data_train:
        for i in range(len(self.data_train)//self.batch_size + int(0 if len(self.data_test)%self.batch_size==0 else 1)):
            U=self.data_train["U"][i*self.batch_size:(i+1)*self.batch_size]
            X=self.data_train["X"][i*self.batch_size:(i+1)*self.batch_size]
            u=torch.tensor(np.stack(U.values,axis=0),dtype=torch.float).to(self.device)
            x=torch.tensor(np.stack(X.values),requires_grad=True,dtype=torch.float).to(self.device)
            #total_loss=self.model.compute_loss(x,u)

            for i in range(self.discriminator_sub_steps):
                # Shuffle in b dimension
                total_loss=self.model.compute_loss(x,u)
                loss=total_loss["Discriminator_loss"]
                self.discriminator_optimizer.zero_grad()
                loss.backward()
                self.discriminator_optimizer.step()

                for k in total_loss.keys():
                    total_loss[k]=total_loss[k].cpu().detach()
                losses_dis.append(total_loss)

            for i in range(self.generator_sub_steps):
                # Shuffle in b dimension
                total_loss=self.model.compute_loss(x,u)
                loss=total_loss["Generator_loss"]
                self.generator_optimizer.zero_grad()
                loss.backward()
                self.generator_optimizer.step()

                for k in total_loss.keys():
                    total_loss[k]=total_loss[k].cpu().detach()
                losses_gen.append(total_loss)


            #print("train")
            #print(len(losses_dis))
        return {"discriminative_losses":losses_dis,"generative_losses":losses_gen}

    def test(self):
        losses_dis=[]
        losses_gen=[]
        self.data_test.sample(frac=1)
        #for U,X in self.data_test:
        for i in range(len(self.data_test)//self.batch_size + int(0 if len(self.data_test)%self.batch_size==0 else 1) ):
            U=self.data_test["U"][i*self.batch_size:(i+1)*self.batch_size]
            X=self.data_test["X"][i*self.batch_size:(i+1)*self.batch_size]
            u=torch.tensor(np.stack(U.values,axis=0),dtype=torch.float).to(self.device) # [batch, n_positions, pde_values]
            x=torch.tensor(np.stack(X.values),requires_grad=True,dtype=torch.float).to(self.device) # [batch, n_positions, position_values]
            #total_loss=self.model.compute_loss(x,u)

            for i in range(self.discriminator_sub_steps):
                # Shuffle in b dimension
                total_loss=self.model.compute_loss(x,u)
                loss=total_loss["Discriminator_loss"]

                for k in total_loss.keys():
                    total_loss[k]=total_loss[k].cpu().detach()
                losses_dis.append(total_loss)

            for i in range(self.generator_sub_steps):
                # Shuffle in b dimension
                total_loss=self.model.compute_loss(x,u)
                loss=total_loss["Generator_loss"]

                for k in total_loss.keys():
                    total_loss[k]=total_loss[k].cpu().detach()
                losses_gen.append(total_loss)


            #print("test")
            #print(len(losses_dis))
        return {"discriminative_losses":losses_dis,"generative_losses":losses_gen}


class Dual_optimizer_LBFGS_trainer(Dual_optimizer_trainer):
    
    def train(self):
        losses_dis=[]
        losses_gen=[]
        self.data_train.sample(frac=1)
        #for U,X in self.data_train:
        for i in range(len(self.data_train)//self.batch_size + int(0 if len(self.data_test)%self.batch_size==0 else 1)):
            U=self.data_train["U"][i*self.batch_size:(i+1)*self.batch_size]
            X=self.data_train["X"][i*self.batch_size:(i+1)*self.batch_size]
            u=torch.tensor(np.stack(U.values,axis=0),dtype=torch.float).to(self.device)
            x=torch.tensor(np.stack(X.values),requires_grad=True,dtype=torch.float).to(self.device)
            #total_loss=self.model.compute_loss(x,u)

            for i in range(self.discriminator_sub_steps):
                # Shuffle in b dimension

                def closure():
                    self.discriminator_optimizer.zero_grad()
                    total_loss=self.model.compute_loss(x,u)
                    loss=total_loss["Discriminator_loss"]
                    loss.backward()
                    return loss

                self.discriminator_optimizer.step(closure)

                total_loss=self.model.compute_loss(x,u)
                loss=total_loss["Discriminator_loss"]

                for k in total_loss.keys():
                    total_loss[k]=total_loss[k].cpu().detach()
                losses_dis.append(total_loss)

                

            for i in range(self.generator_sub_steps):
                # Shuffle in b dimension

                def closure():
                    self.generator_optimizer.zero_grad()
                    total_loss=self.model.compute_loss(x,u)
                    loss=total_loss["Generator_loss"]
                    loss.backward()
                    return loss

                self.generator_optimizer.step(closure)
                
                total_loss=self.model.compute_loss(x,u)
                loss=total_loss["Generator_loss"]

                for k in total_loss.keys():
                    total_loss[k]=total_loss[k].cpu().detach()
                losses_gen.append(total_loss)

        return {"discriminative_losses":losses_dis,"generative_losses":losses_gen}


class Launch_train(object):
    def launch(self,directory,epochs):
        self.exp_data=json.load(open(os.path.join(directory,"config.json")))
        self.instantiate_model()
        self.exp_data["trainer"]["trainer_args"]["optimizer"]=eval(self.exp_data["trainer"]["trainer_args"]["optimizer"])
        self.Trainer=getattr(sys.modules[__name__],self.exp_data["trainer"]["trainer_type"])(
            self.model,
            data_dir=directory,
            #data_path=directory,
            **self.exp_data["trainer"]["trainer_args"]
        )
        self.Trainer.epochs_train_test(epochs)

    def launch_data_test(self,directory,epochs):
        self.exp_data=json.load(open(os.path.join(directory,"config.json")))
        self.instantiate_model()
        self.exp_data["trainer"]["trainer_args"]["optimizer"]=eval(self.exp_data["trainer"]["trainer_args"]["optimizer"])
        self.Trainer=getattr(sys.modules[__name__],self.exp_data["trainer"]["trainer_type"])(
            self.model,
            data_dir=directory,
            #data_path=directory,
            **self.exp_data["trainer"]["trainer_args"]
        )
        self.Trainer.data_size_test(epochs)
    
    def instantiate_model(self):
        for k in self.exp_data["model"]["args"].keys():
            self.exp_data["model"]["args"][k]=eval(self.exp_data["model"]["args"][k])
        self.model=getattr(sys.modules[__name__],self.exp_data["model"]["name"])(**self.exp_data["model"]["args"])

    
    


if __name__=="__main__":
    #launch command python Train.py launch --directory "path/to/exps" --epochs 200
    fire.Fire(Launch_train)
