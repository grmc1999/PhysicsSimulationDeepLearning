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
      
from phi.torch.flow import fluid,Solve
      

class PINNS_based_SOL_trainer(object):
    def __init__(self,field,physical_model,statistical_model,optimizer,simulation_steps,time_step,loss):

      self.dt=time_step
      self.v=field

      self.ph_model=physical_model(self.v,dt=self.dt)

      self.init_states_gt=[self.v]
      self.T=[0.0]

      for i in range(50):
        self.init_states_gt.append(self.ph_model.step(self.init_states_gt[-1]))
        self.T.append(self.T[-1]+self.dt)

      self.n_steps=simulation_steps
      self.st_model=statistical_model
      self.loss=loss
      self.optimizer=optimizer

      self.geometry=self.v.geometry

    def generate_postion_time_code(self,field,t):  # Re implement if more dimensions are needed
      X=field.geometry.center.native("x,y")
      T=torch.ones(X.shape[:2]+(1,))*t
      XT=rearrange(torch.concat((X,T),axis=-1),"x y c -> c x y").unsqueeze(0)
      return XT
    
    def correct(self,states_pred):
      XT=self.generate_postion_time_code(states_pred[-1],self.t)
      #Up=torch.concat(tuple(map(lambda T:Space2Tensor(T,self.geometry),states_pred[-1])),axis=-1)
      Up=Space2Tensor(states_pred[-1],self.geometry)
      print("Up",Up.shape)
      print("XT",XT.shape)
      XTUp_1=torch.concat((XT,Up),axis=1) # [X Y T U P]
      # TODO: implement a method to be re implemented for other architecures
      XTUp=self.st_model(XTUp_1)
      return XTUp_1,XTUp
      
      

    def forward_prediction_correction(self):

      states_pred=[self.v]

      XTUp_1,XTUp=self.correct(states_pred)

      #XT=self.generate_postion_time_code(states_pred[-1][0],self.t)
      #Up=torch.concat(tuple(map(lambda T:Space2Tensor(T,self.geometry),states_pred[-1])),axis=-1)
      #XTUp_1=torch.concat((XT,Up),axis=-1) # [X Y T U P]
      ## TODO: implement a method to be re implemented for other architecures
      #XTUp=self.st_model(XTUp_1)
      
      states_in=[tuple(map(lambda T:Tensor2Space(T,self.geometry),torch.split(XTUp_1,1,dim=-1)))]
      states_corr=[tuple(map(lambda T:Tensor2Space(T,self.geometry),torch.split(XTUp,1,dim=-1)))]
      states_pred=[map(lambda x,y:x+y,self.v,states_corr[-1])]

      Up=Space2Tensor(states_pred[-1],self.v.geometry)
      states_corr=[Tensor2Space(self.st_model(Up),self.v.geometry)]

      states_pred=[self.v+states_corr[-1]]


      # For steps in correction run (4 in example) (incidencia nos iniciais)
      for i in range(self.n_steps):

        # Step last in states_pred
        states_pred.append(self.ph_model.step(states_pred[-1]))
        # Correct with model of last states_pred
        XTUp_1,XTUp=self.correct(states_pred)

        #XT=self.generate_postion_time_code(states_pred[-1][0],self.t+self.dt*(i+1))
        #Up=torch.concat(tuple(map(lambda T:Space2Tensor(T,self.geometry),states_pred[-1])),axis=-1)
        #XTUp_1=torch.concat((XT,Up),axis=-1) # [X Y T U P]
        #XTUp=self.st_model(XTUp_1)
        #states_corr=[tuple(map(lambda T:Tensor2Space(T,self.geometry),(XTUp[:,:,0],XTUp[:,:,1])))]
        states_in.append(tuple(map(lambda T:Tensor2Space(T,self.geometry),torch.split(XTUp_1,1,dim=-1))))
        states_corr.append(tuple(map(lambda T:Tensor2Space(T,self.geometry),torch.split(XTUp,1,dim=-1))))
        states_pred.append(map(lambda x,y:x+y,states_pred[-1],states_corr[-1]))

      states_pred=list(map(lambda corr:Space2Tensor(corr,self.geometry),states_pred))

      return states_pred,states_corr,states_in

    def train(self,epochs):
      losses=[]
      for i in range(epochs):
        print(f"epoch {i}")
        gt_batch=[]
        co_batch=[]
        for b in range(5):

          random_idx=randint(0,len(self.init_states_gt)-1)
          self.v=self.init_states_gt[random_idx]
          self.t=self.T[random_idx]
          states_pred,states_corr,states_in=self.forward_prediction_correction()
          co_batch=co_batch+states_pred

        states_pred=torch.concat(states_pred,axis=0) # [B X Y U]
        loss=self.loss(states_in,states_pred)


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