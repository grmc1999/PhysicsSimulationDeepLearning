from phi import physics
from phi.torch.flow import diffuse, advect, Solve, fluid, math


class physical_model(object):
  def __init__(self,domain,dt):
    #self.v0=v0
    self.domain=domain
    self.dt=dt

  def momentum_eq(self,u, u_prev, dt, diffusivity=0.01):
    diffusion_term = dt * diffuse.differential(u, diffusivity, correct_skew=False)
    advection_term = dt * advect.differential(u, u_prev, order=1)
    return u + advection_term + diffusion_term


  def implicit_time_step(self, v, dt):
      v = math.solve_linear(self.momentum_eq, v, Solve(x0=v), u_prev=v, dt=-dt)
      v, p = fluid.make_incompressible(v,solve=Solve('CG-adaptive', 1e-5))
      return v