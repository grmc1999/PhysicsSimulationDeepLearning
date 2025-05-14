import torch
from DL_models.PINNS.losses import PDE_res,Discriminator_loss,Generator_loss,PDE_GAN_loss

class PINN_base(torch.nn.Module):
    def __init__(self):
        """
        Provides basic functionalities for PDE behavior
        """
        super(PINN_base,self).__init__()
        print("PINNS_init")
    def derivatives(self,u,x,n):
        print(n)
        if n==0:
            return u
        else:
            du=torch.autograd.grad(u,x,
            grad_outputs=torch.ones_like(x).to(u.device),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
            )[0]
        return self.derivatives(du,x,n-1)


class PINN_vanilla(PINN_base):
    def __init__(self,u_dim,args_PDE_res):
        super(GAN_PI_base,self).__init__()
        self.loss=PDE_res(args_PDE_res)
        self.u_dims=u_dim

    def Generate_forward(self,X):
        #z=torch.normal(**self.distribution_args)
        z=torch.normal(
            #list(torch.ones_like(torch.zeros((3,4,6))).shape[:-1])+[4]
            mean=torch.zeros(list(X.shape[:-1])+[self.u_dims]),
            std=torch.ones(list(X.shape[:-1])+[self.u_dims])
        ).to(X.device)
        u=self.G_model(torch.concatenate([z,X],axis=-1))
        return u

    def compute_loss(self,X,U):
        U_=self.Generate_forward(X)
        return self.loss(X,U)