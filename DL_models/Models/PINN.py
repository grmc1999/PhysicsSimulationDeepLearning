import torch

class PINN_base(torch.nn.Module):
    def __init__(self):
        """
        Provides basic functionalities for PDE behavior
        """
        super(PINN_base,self).__init__()
    def derivatives(self,u,x,n):
        if n==0:
            return u
        else:
            du=torch.autograd.grad(u,x,
            grad_outputs=torch.ones_like(x).to(device),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
            )[0]
        return self.derivatives(du,x,n-1)
