import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from DL_models.Models.GAN import *
from DL_models.Models.PINN import *
from DL_models.PINNS.utils import derivatives

from Physical_models.Differentiable_simmulation import physical_model

from Transforms.Data_transforms import *

import fire
import json

from phi.torch.flow import vec,UniformGrid, Field, tensor
import choice
from Physical_models.Differentiable_simulation import physical_model,Space2Tensor,Tensor2Space

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