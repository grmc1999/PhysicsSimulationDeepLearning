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

class SOL_trainer(object):
    def __init__(self,model,optimizer,boundary_conditions,initial_conditions=vec(x=tensor(0.0),y=tensor(0.0)),
                 simulation_steps=4,spatial_step=128,time_step=0.1,coarse_to_fine_timefactor=1/4,co2gt_spatial_factor=4):
      self.boundary = boundary_conditions
      self.initial_conditions=initial_conditions
      self.co2gt_spatial_factor=co2gt_spatial_factor
      self.spatial_step=spatial_step
      self.co2gt_time_factor=coarse_to_fine_timefactor
      self.co_dt=time_step
      self.gt_dt=self.co_dt*self.co2gt_time_factor

      self.geo_co=UniformGrid(x=self.spatial_step, y=self.spatial_step)
      self.geo_gt=UniformGrid(x=self.spatial_step*self.co2gt_spatial_factor, y=self.spatial_step*self.co2gt_spatial_factor)
      self.v_co=Field(self.geo_co,values=self.initial_conditions,boundary=self.boundary) # add initial conditions
      self.v_gt=Field(self.geo_gt,values=self.initial_conditions,boundary=self.boundary) # add initial conditions

      self.ph_model_co=physical_model(self.v_co,dt=self.co_dt)
      self.ph_model_gt=physical_model(self.v_gt,dt=self.gt_dt)

      self.n_steps=simulation_steps
      self.st_model=model

      self.loss=(lambda y_,y: torch.mean((y_-y)**2))

      self.optimizer=optimizer

      self.alpha=1

    def forward_prediction_correction(self):
      print(f"prediction correction simulation")
      #states_pred=[self.v_co]
      states_pred=[self.v_co + Tensor2Space(self.st_model(Space2Tensor(self.v_co,self.geo_co)))]
      #states_corr=[Space2Tensor(self.v_co,self.geo_co)+self.st_model(Space2Tensor(states_pred[-1],self.geo_co))]
      #states_corr=[self.st_model(Space2Tensor(states_pred[-1],self.geo_gt))]
      states_corr=[Tensor2Space(self.st_model(Space2Tensor(self.v_co,self.geo_co)))]

      # For steps in correction run (4 in example) (incidencia nos iniciais)
      for i in range(self.n_steps):
        print(f"step {i}")
        # Simulation step: run step method in physics class and add to list states_predicted
        #corrected_space = states_pred[-1] + Tensor2Space(self.st_model(Space2Tensor(states_pred[-1],self.geo_gt)))
        states_pred.append(self.ph_model_co.step(
            states_pred[-1] + Tensor2Space(self.st_model(Space2Tensor(states_pred[-1],self.geo_co)))
            ))

        # Prediction step: run model
        #states_corr.append(Space2Tensor(corrected_space,self.geo_gt))
        #states_corr.append(self.st_model(Space2Tensor(states_pred[-1],self.geo_gt)))
        states_corr.append(Tensor2Space(self.st_model(Space2Tensor(states_pred[-1],self.geo_co))))
        plot(Field(
          values=states_corr[-1],
          geometry=self.geo_gt,boundary=self.boundary))
        plt.show()


      print(f"alpha value: {self.alpha}")
      #states_corr=list(map(lambda corr,pred:self.alpha*corr + Space2Tensor(pred,self.geo_gt),states_corr,states_pred))
      states_pred=list(map(lambda corr:Space2Tensor(corr,self.geo_gt),states_pred))

      self.v_co=Field(values=Tensor2Space(states_pred[-1].detach().clone()),geometry=self.geo_gt,boundary=self.boundary).sample(self.geo_co)

      self.v_co=Field(values=self.v_co,geometry=self.geo_co,boundary=self.boundary)

      return states_pred,states_corr

    def forward_fine_grained(self):
      states_gt=[Space2Tensor(self.v_gt,self.geo_gt)]

      print(f"fine grained simulation")
      for i in range(int(self.n_steps/self.co2gt_time_factor)):
        #print(f"fine grained step {i}")
        self.v_gt=self.ph_model_gt.step(self.v_gt)
        if i%int(1/self.co2gt_time_factor)==0:
          states_gt.append(Space2Tensor(self.v_gt,self.geo_gt))
      return states_gt

    def train(self,epochs):
      losses=[]
      for i in range(epochs):
        print(f"epoch {i}")
        #self.alpha=(i/(epochs))
        #self.alpha=(i/(20))
        states_gt=self.forward_fine_grained()
        states_pred,states_corr=self.forward_prediction_correction()

        states_pred=torch.concat(states_pred,axis=0)
        states_gt=torch.concat(states_gt,axis=0)
        loss=self.loss(states_pred,states_gt)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=False)
        self.optimizer.step()

        losses.append(loss.cpu().detach().numpy())
        print(loss)
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