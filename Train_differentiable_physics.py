import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from DL_models.Models.GAN import *
from DL_models.Models.PINN import *
from DL_models.PINNS.utils import derivatives

from Transforms.Data_transforms import *

import fire
import json

from phi.torch.flow import vec,UniformGrid, Field, tensor
from random import choice,sample
from Physical_models.Differentiable_simulation import physical_model,Space2Tensor,Tensor2Space
from copy import copy
from einops import rearrange
from scipy import ndimage
from random import randint

class SOL_trainer(object):
    def __init__(self,boundary,model,optimizer,simulation_steps,spatial_step,time_step,Initial_conditions=vec(x=tensor(0.0),y=tensor(0.0)),coarse_to_fine_timefactor=1/4,co2gt_spatial_factor=4):
      self.boundary = boundary
      self.co2gt_spatial_factor=co2gt_spatial_factor
      self.spatial_step=spatial_step
      self.co2gt_time_factor=coarse_to_fine_timefactor
      self.co_dt=time_step
      self.gt_dt=self.co_dt*self.co2gt_time_factor
      self.IC=Initial_conditions

      self.geo_co=UniformGrid(x=self.spatial_step, y=self.spatial_step)
      self.geo_gt=UniformGrid(x=self.spatial_step*self.co2gt_spatial_factor, y=self.spatial_step*self.co2gt_spatial_factor)
      self.v_co=Field(self.geo_co,values=self.IC,boundary=self.boundary) # add initial conditions
      self.v_gt=Field(self.geo_gt,values=self.IC,boundary=self.boundary) # add initial conditions

      self.ph_model_co=physical_model(self.v_co,dt=self.co_dt)
      self.ph_model_gt=physical_model(self.v_gt,dt=self.gt_dt)

      self.init_states_gt=[self.v_gt]
      for i in range(50):
        self.init_states_gt.append(self.ph_model_gt.step(self.init_states_gt[-1]))

      self.n_steps=simulation_steps
      self.st_model=model

      self.loss=(lambda y_,y: torch.sum((y-y_)**2)/self.n_steps)

      self.optimizer=optimizer

      self.alpha=1

    def forward_prediction_correction(self):
      #print(f"prediction correction simulation")

      states_pred=[self.v_co]
      states_corr=[Tensor2Space(self.st_model(Space2Tensor(self.v_co,self.geo_co)),self.geo_co)]

      states_pred=[self.v_co+states_corr[-1]]

      # For steps in correction run (4 in example) (incidencia nos iniciais)
      for i in range(self.n_steps):

        # Step last in states_pred
        states_pred.append(self.ph_model_co.step(states_pred[-1]))
        # Correct with model of last states_pred
        states_corr.append(Tensor2Space(self.st_model(Space2Tensor(states_pred[-1],self.geo_co)),self.geo_co))

        # Sum correction to last in states pred
        states_pred[-1]=states_pred[-1]+states_corr[-1]

      states_pred=list(map(lambda corr:Space2Tensor(corr,self.geo_gt),states_pred))

      return states_pred,states_corr

    def forward_fine_grained(self):
      states_gt=[Space2Tensor(self.v_gt,self.geo_gt)]

      #print(f"fine grained simulation")
      for i in range(int(self.n_steps/self.co2gt_time_factor)):
        ##print(f"fine grained step {i}")
        self.v_gt=self.ph_model_gt.step(self.v_gt)
        if i%int(1/self.co2gt_time_factor)==0:
          states_gt.append(Space2Tensor(self.v_gt,self.geo_gt))
      return states_gt

    def train(self,epochs):
      losses=[]
      for i in range(epochs):
        #print(f"epoch {i}")
        gt_batch=[]
        co_batch=[]
        for b in range(5):
          self.v_gt=choice(self.init_states_gt)
          states_gt=self.forward_fine_grained()
          self.v_co=Field(values=Tensor2Space(states_gt[0].detach(),self.geo_co),geometry=self.geo_co,boundary=self.boundary)
          states_pred,states_corr=self.forward_prediction_correction()
          gt_batch=gt_batch+states_gt
          co_batch=co_batch+states_pred

        states_pred=torch.concat(states_pred,axis=0)
        states_gt=torch.concat(states_gt,axis=0)
        loss=self.loss(states_pred,states_gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        losses.append(loss.cpu().detach().numpy())
      return losses

      def test(self,epochs):
        losses=[]
        for i in range(epochs):
          gt_batch=[]
          co_batch=[]
          for b in range(5):
            self.v_gt=choice(self.init_states_gt)
            states_gt=self.forward_fine_grained()
            self.v_co=Field(values=Tensor2Space(states_gt[0].detach(),self.geo_co),geometry=self.geo_co,boundary=self.boundary)
            states_pred,states_corr=self.forward_prediction_correction()
            gt_batch=gt_batch+states_gt
            co_batch=co_batch+states_pred

          states_pred=torch.concat(states_pred,axis=0)
          states_gt=torch.concat(states_gt,axis=0)
          loss=self.loss(states_pred,states_gt)

          losses.append(loss.cpu().detach().numpy())
        return losses
      

class SOL_trainer_darcyflow(object):
    def __init__(self,ph_model,coarse_model,model,optimizer,simulation_steps,time_step,gt_factor,batch_size,train_horizon):

      self.Space2Tensor= lambda Space,geometry:Space2Tensor(Space,geometry,space_signature='x,y,vector',tensor_signature="b x y c ->b c x y")
      self.Tensor2Space= lambda Tensor,geometry:Tensor2Space(Tensor,geometry,space_signature="x:s,y:s",tensor_signature="x y -> x y")
      
      self.co_dt=time_step
      self.gt_dt=time_step*gt_factor
      self.gt_factor=gt_factor

      self.v_co=coarse_model  # (Field (x y) Field (x y))

      self.v_gt=copy(coarse_model)

      self.init_states_gt=[self.v_gt]
      self.ph_model=ph_model

      for i in range(train_horizon):
        self.init_states_gt.append(self.ph_model.RK4(*self.init_states_gt[-1],1e5)) # list [ horizon (Field[x y] Field[x y] dtF dtF) ]

      self.n_steps=simulation_steps
      self.st_model=model
      self.batch_size=batch_size

      #self.loss=(lambda y_,y: torch.sum((y[:,:2]-y_[:,:2])**2)/self.n_steps)
      self.loss=(lambda y_,y: torch.sum((y[:,:]-y_[:,:])**2)/self.n_steps)

      self.optimizer=optimizer

      self.alpha=1

    def forward_prediction_correction(self):

      states_pred=[self.v_co]
      
      correction=self.st_model(
          self.Space2Tensor(self.v_co[0],self.v_co[0].geometry),
          self.Space2Tensor(self.v_co[1],self.v_co[1].geometry),
          self.Space2Tensor(self.v_co[2],self.v_co[2].geometry),
          self.Space2Tensor(self.v_co[3],self.v_co[3].geometry)
          )

      states_corr=[(
        self.Tensor2Space(correction[0][0],self.v_co[0].geometry),
        self.Tensor2Space(correction[1][0],self.v_co[0].geometry),
        self.Tensor2Space(correction[2][0],self.v_co[0].geometry),
        self.Tensor2Space(correction[3][0],self.v_co[0].geometry),
        )]

      states_pred=[(
        self.v_co[0]+states_corr[-1][0],
        self.v_co[1]+states_corr[-1][1],
        self.v_co[2],#+states_corr[-1][2],
        self.v_co[3],#+states_corr[-1][3],
        )]

      # For steps in correction run (4 in example) (incidencia nos iniciais)
      for i in range(self.n_steps):

        # Step last in states_pred
        states_pred.append(self.ph_model.RK4(*states_pred[-1],self.co_dt))
        # Correct with model of last states_pred
        correction=self.st_model(
          self.Space2Tensor(states_pred[-1][0],self.v_co[0].geometry),
          self.Space2Tensor(states_pred[-1][1],self.v_co[1].geometry),
          self.Space2Tensor(states_pred[-1][2],self.v_co[2].geometry),
          self.Space2Tensor(states_pred[-1][3],self.v_co[3].geometry)
          )
        states_corr.append((
          self.Tensor2Space(correction[0][0],self.v_co[0].geometry),
          self.Tensor2Space(correction[1][0],self.v_co[1].geometry),
          self.Tensor2Space(correction[2][0],self.v_co[2].geometry),
          self.Tensor2Space(correction[3][0],self.v_co[3].geometry)
          ))
        # Sum correction to last in states pred
        states_pred[-1]=(
          states_pred[-1][0]+states_corr[-1][0],
          states_pred[-1][1]+states_corr[-1][1],
          states_pred[-1][2], #+states_corr[-1][2],
          states_pred[-1][3] #+states_corr[-1][3]
                         )

      #states_pred=list(map(lambda corr:self.Space2Tensor(corr,self.v_co[0].geometry),states_pred))
      states_pred=list(map(lambda corr:
                           (
                             self.Space2Tensor(corr[0],self.v_co[0].geometry),
                             self.Space2Tensor(corr[1],self.v_co[1].geometry),
                             self.Space2Tensor(corr[2],self.v_co[2].geometry),
                             self.Space2Tensor(corr[3],self.v_co[3].geometry)
                             ),states_pred)) # b c x y

      return states_pred,states_corr

    def roll_to_batch(self,roll):
      """
      conver roll to batch
      roll list of tuples of tensors [H (T_space1 T_space1)]
      """
      return torch.concat(list(map( lambda ss:torch.concat(ss,axis=1),roll )),axis=0)
      
    def forward_fine_grained(self):
      states_gt=[(
        self.Space2Tensor(self.v_gt[0],self.v_co[0].geometry),
        self.Space2Tensor(self.v_gt[1],self.v_co[0].geometry),
        self.Space2Tensor(self.v_gt[2],self.v_co[0].geometry),
        self.Space2Tensor(self.v_gt[3],self.v_co[0].geometry),
         )]

      for i in range(int(self.n_steps/self.gt_factor)):
        self.v_gt=self.ph_model.RK4(*self.v_gt,self.gt_dt)
        if i%int(1/self.gt_factor)==0:
          states_gt.append((
            self.Space2Tensor(self.v_gt[0],self.v_co[0].geometry),
            self.Space2Tensor(self.v_gt[1],self.v_co[0].geometry),
            self.Space2Tensor(self.v_gt[2],self.v_co[0].geometry),
            self.Space2Tensor(self.v_gt[3],self.v_co[0].geometry)
            ))
      return states_gt # list [ fine_grained (Field[x y] Field[x y]) ]

    def train(self,epochs):
      losses=[]
      for i in range(epochs):
        print(f"epoch {i}")
        gt_batch=[]
        co_batch=[]

        batch_init_ind=sample(list(range(len(self.init_states_gt))),self.batch_size)
        for i in batch_init_ind:
          print(f"making batch {i}")
          self.v_gt=self.init_states_gt[i]
          states_gt=self.forward_fine_grained() # list [ fine_grained (Field[x y] Field[x y] dtF dtF) ]

          self.v_co=(
              self.Tensor2Space(states_gt[0][0][0,0].detach(),self.v_co[0].geometry), # Field[x y]
              self.Tensor2Space(states_gt[0][1][0,0].detach(),self.v_co[0].geometry), # Field[x y]
              self.Tensor2Space(states_gt[0][1][0,0].detach(),self.v_co[0].geometry), # dtF
              self.Tensor2Space(states_gt[0][1][0,0].detach(),self.v_co[0].geometry) # dtF
              )
          states_pred,states_corr=self.forward_prediction_correction()
          gt_batch=gt_batch+states_gt
          co_batch=co_batch+states_pred
        
        states_pred=self.roll_to_batch(states_pred)
        states_gt=self.roll_to_batch(states_gt)
        loss=self.loss(states_pred,states_gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(loss)

        losses.append(loss.cpu().detach().numpy())
      return losses

      def test(self,epochs):
        losses=[]
        for i in range(epochs):
          #self.alpha=self.alpha*(i/epochs)
          states_pred,states_corr=self.forward_prediction_correction()
          states_gt=self.forward_fine_grained()

          states_pred=torch.concat(states_pred,axis=0)
          states_gt=torch.concat(states_gt,axis=0)
          loss=self.loss(states_pred,states_gt)

          losses.append(loss.cpu().detach().numpy())
        return losses

class data_based_SOL(SOL_trainer_darcyflow):
    def __init__(self,ph_model,coarse_model,gt_data,model,optimizer,simulation_steps,time_step,gt_factor,batch_size):
        self.Space2Tensor= lambda Space,geometry:Space2Tensor(Space,geometry,space_signature='x,y,vector',tensor_signature="b x y c ->b c x y")
        self.Tensor2Space= lambda Tensor,geometry:Tensor2Space(Tensor,geometry,space_signature="x:s,y:s",tensor_signature="x y -> x y")

        self.co_dt=time_step
        self.gt_dt=time_step*gt_factor
        self.gt_factor=gt_factor

        self.v_co=coarse_model  # (Field (x y) Field (x y))

        self.init_states_gt=gt_data
        self.ph_model=ph_model

        self.n_steps=simulation_steps
        self.st_model=model
        self.batch_size=batch_size
        self.loss=(lambda y_,y: torch.sum((y[:,:]-y_[:,:])**2)/self.n_steps)

        self.optimizer=optimizer

        self.gt_data_normalization()

    def transform_phase_pressure2Sw(self,Fields):
      P_w=Fields[:,0]
      P_o=Fields[:,1]
      return self.ph_model.Sw_Pc_f(self.ph_model.compute_p_c(P_w,P_o))

    def transform_phase_pressure2Pt(self,Fields):
        P_w=Fields[:,0]
        P_o=Fields[:,1]
        return P_w+P_o

    def transform_sw_pt2pa(self,Sw,pt):
      pc=Pc_f(Sw)
      p_o=(pt+pc)/2
      p_w=(pt-pc)/2
      return p_o,p_w
    
    def gt_data_normalization(self):
      self.init_states_gt.x=((self.init_states_gt.x.values/50-1)*0.5).astype(int)
      self.init_states_gt.y=(np.abs(self.init_states_gt.y.values/50-1)*0.5).astype(int)-1
      self.init_states_gt.t=(self.init_states_gt.t.values-1).astype(int)

    def prop_to_time_tensor_prop(self,t,prop):
      self.init_states_gt[self.init_states_gt.t==t]
      data_tensor=np.ndarray([self.init_states_gt.x.max()+1,self.init_states_gt.y.max()+1])
      data_tensor[self.init_states_gt.x.values.astype(int),self.init_states_gt.y.values.astype(int)]=self.init_states_gt[prop].values

      return data_tensor
    
    def forward_fine_grained(self,t,prop):
      v_gt=[]
      for i in range(self.n_steps+1):
        self.init_states_gt[self.init_states_gt.t==(t+i)]
        data_tensor=np.ndarray([int(self.init_states_gt.x.max()+1),int(self.init_states_gt.y.max()+1),len(prop)])
        data_tensor[self.init_states_gt.x.values.astype(int),self.init_states_gt.y.values.astype(int)]=self.init_states_gt[prop].values
        # TODO should return list of n Ws[Field], P[Field]
        if self.v_co[0].numpy("x,y").shape!=data_tensor.shape:
          data_tensor=ndimage.zoom(data_tensor,
                                   (self.v_co[0].numpy("x,y").shape[0]/data_tensor.shape[0],self.v_co[0].numpy("x,y").shape[1]/data_tensor.shape[1],1.0)
                                   )

        data_tensor=torch.from_numpy(data_tensor)
        v_gt.append(data_tensor)
      
      return v_gt # list [ fine_grained (Field[x y] Field[x y]) ]
    
    def forward_prediction_correction(self):

      states_pred=[self.v_co]
      
      correction=self.st_model(
          self.Space2Tensor(self.v_co[0],self.v_co[0].geometry),
          self.Space2Tensor(self.v_co[1],self.v_co[1].geometry),
          self.Space2Tensor(self.v_co[2],self.v_co[2].geometry),
          self.Space2Tensor(self.v_co[3],self.v_co[3].geometry)
          )

      states_corr=[(
        self.Tensor2Space(correction[0][0],self.v_co[0].geometry),
        self.Tensor2Space(correction[1][0],self.v_co[0].geometry),
        self.Tensor2Space(correction[2][0],self.v_co[0].geometry),
        self.Tensor2Space(correction[3][0],self.v_co[0].geometry),
        )]

      states_pred=[(
        self.v_co[0]+states_corr[-1][0],
        self.v_co[1]+states_corr[-1][1],
        self.v_co[2]+states_corr[-1][2],
        self.v_co[3]+states_corr[-1][3],
        )]

      # For steps in correction run (4 in example) (incidencia nos iniciais)
      for i in range(self.n_steps*self.gt_factor):

        # Step last in states_pred
        states_pred.append(self.ph_model.RK4(*states_pred[-1],self.co_dt))
        # Correct with model of last states_pred
        correction=self.st_model(
          self.Space2Tensor(states_pred[-1][0],self.v_co[0].geometry),
          self.Space2Tensor(states_pred[-1][1],self.v_co[1].geometry),
          self.Space2Tensor(states_pred[-1][2],self.v_co[2].geometry),
          self.Space2Tensor(states_pred[-1][3],self.v_co[3].geometry)
          )
        states_corr.append((
          self.Tensor2Space(correction[0][0],self.v_co[0].geometry),
          self.Tensor2Space(correction[1][0],self.v_co[1].geometry),
          self.Tensor2Space(correction[2][0],self.v_co[2].geometry),
          self.Tensor2Space(correction[3][0],self.v_co[3].geometry)
          ))
        # Sum correction to last in states pred
        # TODO: apply data transform here recieving tuple of states the functions should reduce the states to the needed
        states_tuple=(states_pred[-1][0]+states_corr[-1][0],states_pred[-1][1]+states_corr[-1][1],states_pred[-1][2],states_pred[-1][3],)

        states_pred[-1]=states_tuple

      states_pred=list(map(lambda corr:
                           (
                             self.Space2Tensor(corr[0],self.v_co[0].geometry),# TODO Implement transformation for fields in target
                             self.Space2Tensor(corr[1],self.v_co[1].geometry),
                             self.Space2Tensor(corr[2],self.v_co[2].geometry),
                             self.Space2Tensor(corr[3],self.v_co[3].geometry)
                             ),states_pred)) # b c x y

      return states_pred[::int(self.gt_factor)],states_corr[::int(self.gt_factor)] # Check

    def train(self,epochs):
      losses=[]
      for i in range(epochs):
        gt_batch=[]
        co_batch=[]

        batch_init_ind=sample(list(range(len(np.unique(self.init_states_gt.t)))),self.batch_size)
        for i in batch_init_ind:
          #self.v_gt=self.init_states_gt[i]
          states_gt=self.forward_fine_grained(i,["P","Ws"]) # list [ fine_grained (Field[x y] Field[x y] dtF dtF) ]

          self.v_co=(
              self.Tensor2Space(self.transform_sw_pt2pa(states_gt[0][:,:,1]*1e3,states_gt[0][:,:,0])[1].detach(),self.v_co[0].geometry), # Field[x y]
              self.Tensor2Space(self.transform_sw_pt2pa(states_gt[0][:,:,1]*1e3,states_gt[0][:,:,0])[0].detach(),self.v_co[0].geometry), # Field[x y]
              self.Tensor2Space(torch.zeros_like(self.transform_sw_pt2pa(states_gt[0][:,:,1]*1e3,states_gt[0][:,:,0])[1].detach()),self.v_co[0].geometry), # dtF
              self.Tensor2Space(torch.zeros_like(self.transform_sw_pt2pa(states_gt[0][:,:,1]*1e3,states_gt[0][:,:,0])[1].detach()),self.v_co[0].geometry) # dtF
              )
          states_pred,states_corr=self.forward_prediction_correction()
          gt_batch=gt_batch+states_gt
          co_batch=co_batch+states_pred
        
        
        states_pred=self.roll_to_batch(states_pred) # [H T_space_1 T_space_2 T_space_3 T_space_4]
        
        Sw=self.ph_model.Sw_Pc_f(torch.clip(states_pred[:,1]-states_pred[:,0],min=self.ph_model.Pc_f(1-self.ph_model.Pc_args["SOR"]),max=self.ph_model.Pc_f(self.ph_model.Pc_args["SWR"])))
        Pt=states_pred[:,0]+states_pred[:,1]

        states_gt=rearrange(torch.stack(states_gt,axis=0),"b x y c ->b c x y")

        loss=self.loss(
          torch.stack((Pt,Sw),axis=1),
          states_gt
          )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(loss)

        losses.append(loss.cpu().detach().numpy())
      return losses

      def test(self,epochs):
        losses=[]
        for i in range(epochs):
          #self.alpha=self.alpha*(i/epochs)
          states_pred,states_corr=self.forward_prediction_correction()
          states_gt=self.forward_fine_grained()

          states_pred=torch.concat(states_pred,axis=0)
          states_gt=torch.concat(states_gt,axis=0)
          loss=self.loss(states_pred,states_gt)

          losses.append(loss.cpu().detach().numpy())
        return losses
      
from phi.torch.flow import fluid,Solve
      

<<<<<<< HEAD
<<<<<<< HEAD
class PINNS_based_SOL_trainer_VP(object):
=======
class PINNS_based_SOL_trainer(object):
>>>>>>> 9eddd057af8c8926d3684d5e4107767e8a7513cb
=======
class PINNS_based_SOL_trainer(object):
>>>>>>> 9eddd057af8c8926d3684d5e4107767e8a7513cb
    def __init__(self,field,model,optimizer,simulation_steps,time_step,loss):

      self.dt=time_step
      self.v=field

      self.ph_model=physical_model(self.v,dt=self.dt)

      self.init_states_gt=[self.v]
      self.T=[0.0]

      for i in range(50):
        self.init_states_gt.append(self.ph_model.step(self.init_states_gt[-1]))
        self.T.append(self.T+self.dt)

      self.n_steps=simulation_steps
      self.st_model=model
      self.loss=loss
      self.optimizer=optimizer

<<<<<<< HEAD
<<<<<<< HEAD
      self.geometry=self.v[0].geometry

    def generate_postion_time_code(self,field,t):
      X=field.geometry.center.native("x,y")
      T=torch.ones(X.shape[:2]+(1,))*t
      XT=torch.concat((X,T),axis=-1)
      return XT

=======
>>>>>>> 9eddd057af8c8926d3684d5e4107767e8a7513cb
=======
>>>>>>> 9eddd057af8c8926d3684d5e4107767e8a7513cb
    def forward_prediction_correction(self):
      #print(f"prediction correction simulation")

      states_pred=[self.v]
<<<<<<< HEAD
<<<<<<< HEAD
      #Up=Space2Tensor(states_pred[-1],self.geometry)
      # TODO ADD POSITION AND TIME ENCODING use selt.t
      XT=self.generate_postion_time_code(states_pred[-1][0],self.t)
      Up=torch.concat(tuple(map(lambda T:Space2Tensor(T,self.geometry),states_pred[-1])),axis=-1)
      XTUp_1=torch.concat((XT,Up),axis=-1) # [X Y T U P]
      XTUp=self.st_model(XTUp_1)
      
      states_in=[tuple(map(lambda T:Tensor2Space(T,self.geometry),torch.split(XTUp_1,1,dim=-1)))]
      states_corr=[tuple(map(lambda T:Tensor2Space(T,self.geometry),torch.split(XTUp,1,dim=-1)))]
      states_pred=[map(lambda x,y:x+y,self.v,states_corr[-1])]
=======
=======
>>>>>>> 9eddd057af8c8926d3684d5e4107767e8a7513cb
      Up=Space2Tensor(states_pred[-1],self.v.geometry)
      # TODO ADD POSITION AND TIME ENCODING use selt.t
      states_corr=[Tensor2Space(self.st_model(Up),self.v.geometry)]

      states_pred=[self.v+states_corr[-1]]
<<<<<<< HEAD
>>>>>>> 9eddd057af8c8926d3684d5e4107767e8a7513cb
=======
>>>>>>> 9eddd057af8c8926d3684d5e4107767e8a7513cb

      # For steps in correction run (4 in example) (incidencia nos iniciais)
      for i in range(self.n_steps):

        # Step last in states_pred
        states_pred.append(self.ph_model.step(states_pred[-1]))
        # Correct with model of last states_pred
<<<<<<< HEAD
<<<<<<< HEAD
        XT=self.generate_postion_time_code(states_pred[-1][0],self.t+self.dt*(i+1))
        Up=torch.concat(tuple(map(lambda T:Space2Tensor(T,self.geometry),states_pred[-1])),axis=-1)
        XTUp_1=torch.concat((XT,Up),axis=-1) # [X Y T U P]
        XTUp=self.st_model(XTUp_1)
        # TODO ADD POSITION AND TIME ENCODING use (selt.t + (i+1)*self.dt)
        #states_corr=[tuple(map(lambda T:Tensor2Space(T,self.geometry),(XTUp[:,:,0],XTUp[:,:,1])))]
        states_in.append(tuple(map(lambda T:Tensor2Space(T,self.geometry),torch.split(XTUp_1,1,dim=-1))))
        states_corr.append(tuple(map(lambda T:Tensor2Space(T,self.geometry),torch.split(XTUp,1,dim=-1))))
        states_pred.append(map(lambda x,y:x+y,states_pred[-1],states_corr[-1]))

      states_pred=list(map(lambda corr:Space2Tensor(corr,self.geometry),states_pred))

      return states_pred,states_corr,states_in
=======
=======
>>>>>>> 9eddd057af8c8926d3684d5e4107767e8a7513cb
        Up=Space2Tensor(states_pred[-1],self.v.geometry)
        # TODO ADD POSITION AND TIME ENCODING use (selt.t + (i+1)*self.dt)
        states_corr.append(Tensor2Space(self.st_model(Up),self.v.geometry))

        # Sum correction to last in states pred
        states_pred[-1]=states_pred[-1]+states_corr[-1]

      states_pred=list(map(lambda corr:Space2Tensor(corr,self.v.geometry),states_pred))

      return states_pred,states_corr
<<<<<<< HEAD
>>>>>>> 9eddd057af8c8926d3684d5e4107767e8a7513cb
=======
>>>>>>> 9eddd057af8c8926d3684d5e4107767e8a7513cb

    def train(self,epochs):
      losses=[]
      for i in range(epochs):
        print(f"epoch {i}")
        gt_batch=[]
        co_batch=[]
        for b in range(5):

          random_idx=randint(len(self.init_states_gt))
          self.v=self.init_states_gt[random_idx]
          self.t=self.T[random_idx]
<<<<<<< HEAD
<<<<<<< HEAD
          states_pred,states_corr,states_in=self.forward_prediction_correction()
=======
          states_pred,states_corr=self.forward_prediction_correction()
>>>>>>> 9eddd057af8c8926d3684d5e4107767e8a7513cb
=======
          states_pred,states_corr=self.forward_prediction_correction()
>>>>>>> 9eddd057af8c8926d3684d5e4107767e8a7513cb
          #gt_batch=gt_batch+states_gt
          co_batch=co_batch+states_pred

        states_pred=torch.concat(states_pred,axis=0) # [B X Y U]
        #states_gt=torch.concat(states_gt,axis=0)
        #loss=self.loss(states_pred,states_gt)
<<<<<<< HEAD
<<<<<<< HEAD
        loss=self.loss(states_in,states_pred)
=======
        loss=self.loss(states_pred)
>>>>>>> 9eddd057af8c8926d3684d5e4107767e8a7513cb
=======
        loss=self.loss(states_pred)
>>>>>>> 9eddd057af8c8926d3684d5e4107767e8a7513cb

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        losses.append(loss.cpu().detach().numpy())
      return losses

      def test(self,epochs):
        losses=[]
        for i in range(epochs):
          gt_batch=[]
          co_batch=[]
          for b in range(5):
            self.v_gt=choice(self.init_states_gt)
            states_gt=self.forward_fine_grained()
            self.v_co=Field(values=Tensor2Space(states_gt[0].detach(),self.geo_co),geometry=self.geo_co,boundary=self.boundary)
            states_pred,states_corr=self.forward_prediction_correction()
            gt_batch=gt_batch+states_gt
            co_batch=co_batch+states_pred

          states_pred=torch.concat(states_pred,axis=0)
          states_gt=torch.concat(states_gt,axis=0)
          loss=self.loss(states_pred,states_gt)

          losses.append(loss.cpu().detach().numpy())
        return losses