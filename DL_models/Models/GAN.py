import sys
import os
import torch
from .MLP import MLP
sys.path.append(os.path.join("..","..","DL_models"))
from DL_models.PINNS.losses import PDE_res,Discriminator_loss,Generator_loss,PDE_GAN_loss
from .PINN import PINN_base


class GAN_PI(PINN_base):
    def __init__(self,G_params,P_params,D_params,args_Gen,args_PDE_res,args_PDE_sup,distribution_args,weights={"generative_posterior_loss":1.,
                                                                "generative_entropy_loss":1.,
                                                                "PDE_residual_loss":1.,
                                                                "PDE_supervised_loss":1.}):
        super(GAN_PI,self).__init__()
        self.G_model=MLP(**G_params)
        self.P_model=MLP(**P_params)
        self.D_model=MLP(**D_params)
        self.loss=PDE_GAN_loss(args_Gen,args_PDE_res,args_PDE_sup,weights=weights)
        self.distribution_args=distribution_args
        self.u_dims=self.G_model.layer_sizes[-1]
    
    def Generate(self,z,X):
        torch.concatenate([z,X],axis=1)
        u=self.G_model(torch.concatenate([z,X],axis=1))

    def Generate_forward(self,X):
        #z=torch.normal(**self.distribution_args)
        z=torch.normal(
            mean=torch.ones([X.shape[0],self.u_dims]),
            std=torch.ones([X.shape[0],self.u_dims])
        )
        u=self.G_model(torch.concatenate([z,X],axis=1))
        return u
    
    def Posterior_forward(self,X,u_):
        z=self.P_model(torch.concatenate([u_,X],axis=1))
        return z

    def Discriminate(self,u):
        y=self.D_model(u)
        return y

    def compute_loss(self,X,U):
        logits_G=self.Generate_forward(X)
        logits_P=self.Posterior_forward(X,logits_G)
        logits_F=self.Discriminate(logits_G)
        logits_R=self.Discriminate(U)
        return self.loss(logits_G,logits_P,logits_F,logits_R,X,U)


