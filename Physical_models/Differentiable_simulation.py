from phi import physics
from phi.torch.flow import diffuse, advect, Solve, fluid, math,Field, unstack,stack,batch,field,vec
from einops import rearrange
import anisotropic_diffusion
from phiml.math import sum as phi_sum


class physical_model(object):
  def __init__(self,domain,dt):
    #self.v0=v0
    self.domain=domain
    self.dt=dt
    self.p=None

  def momentum_eq(self,u, u_prev, dt, diffusivity=0.01):
    diffusion_term = dt * diffuse.implicit(u,diffusivity, dt=dt,correct_skew=False)
    advection_term = dt * advect.semi_lagrangian(u, u_prev,dt)
    return u + advection_term + diffusion_term


  def implicit_time_step(self, v, dt):
      v = math.solve_linear(self.momentum_eq, v, Solve('CG-adaptive',
                                                    1e-2,
                                                    1e-2,x0=v), u_prev=v, dt=-dt)
      v,p = fluid.make_incompressible(v,solve=Solve('CG-adaptive',
                                                    1e-2,
                                                    1e-2,
                                                          #x0=self.p
                                                          ))
      return v

  def step(self,v):
    return self.implicit_time_step(v,self.dt)

def Space2Tensor(Space,geometry,space_signature='x,y,vector,',tensor_signature="b x y c->b c x y"):
  return rearrange(Space.sample(geometry).native(space_signature).unsqueeze(0),tensor_signature)


def Tensor2Space(Tensor,geometry,tensor_signature='c x y->x y c',space_signature="x:s,y:s,vector:c"):
  return Field(geometry=geometry,values=math.wrap(rearrange(Tensor[0],'c x y->x y c'),"x:s,y:s,vector:c"))


import sympy


SWC=sympy.symbols("S_{wc}")
SOR=sympy.symbols("S_{or}")
Sw=sympy.symbols("S_w")
lam=sympy.symbols("\lambda")
Pi=sympy.symbols("P_i")
K_rw0=sympy.symbols("k_{rw0}")
K_ro0=sympy.symbols("k_{ro0}")

Pc_=sympy.symbols("P_c")

Sc=(Sw-SWC)/(1-SWC-SOR)
Pc=Pi*Sc**(-1/lam)

Sw_Pc=(1-SWC-SOR)*((Pc_/Pi)**(-1*lam))+SWC
dScdPc=sympy.diff(Sw_Pc,Pc_)


K_rw=K_rw0*Sc**((2+3*lam)/(lam))
K_ro=K_ro0*((1-Sc)**2)*(1-Sc**((2+lam)/(lam)))

K_w=lambda K_l,p_c,mu,por:stack(
    [stack([K_l*K_rw_f(Sw_Pc_f(p_c))/(por*mu*dScdPc_f(p_c)),math.zeros_like(p_c)],batch("k") ),
    stack([math.zeros_like(p_c),K_l*K_rw_f(Sw_Pc_f(p_c))/(por*mu*dScdPc_f(p_c))],batch("k") )],batch("KK"))

K_o=lambda K_l,p_c,mu,por:stack(
    [stack([K_l*K_ro_f(Sw_Pc_f(p_c))/(por*mu*dScdPc_f(p_c)),math.zeros_like(p_c)],batch("k") ),
    stack([math.zeros_like(p_c),K_l*K_ro_f(Sw_Pc_f(p_c))/(por*mu*dScdPc_f(p_c))],batch("k") )],batch("KK"))

dK_w=lambda K_l,p_c,mu,por:stack(
    [stack([K_l*dK_rw_f(Sw_Pc_f(p_c))/(por*mu),math.zeros_like(p_c)],batch("dk") ),
    stack([math.zeros_like(p_c),K_l*dK_rw_f(Sw_Pc_f(p_c))/(por*mu)],batch("dk") )],batch("dKK"))

dK_o=lambda K_l,p_c,mu,por:stack(
    [stack([K_l*dK_ro_f(Sw_Pc_f(p_c))/(por*mu),math.zeros_like(p_c)],batch("dk") ),
    stack([math.zeros_like(p_c),K_l*dK_ro_f(Sw_Pc_f(p_c))/(por*mu)],batch("dk") )],batch("dKK"))

# Contraction operations

# dK_a = dK_o(p_c) or dK_a = dK_w(p_c)
grad_phi_dK = lambda phi_a,dK_a:(math.dot(
    field.spatial_gradient(phi_a,phi_a.boundary).sample(phi_a.geometry),"vector",
    dK_a,"dKK"))

class two_phase_flow_fake(object):
  def __init__(self,phi_w,phi_o,dt,w_advection_solver,o_advection_solver):
    #self.v0=v0
    self.phi_w=phi_w
    self.phi_o=phi_o
    self.dt=dt
    self.p=None
    self.w_advection_solver=w_advection_solver
    self.o_advection_solver=o_advection_solver


  def compute_p_c(self,phi_w,phi_o):
    p_c=phi_w.sample(phi_w.geometry) -\
    phi_o.sample(phi_o.geometry)
    return p_c

  def compute_convective_velocity(self,phi_a,phi_b,dK_a):
    p_c=self.compute_p_c(self.phi_w,self.phi_o)
    convective_velocity = grad_phi_dK(phi_a,dK_a(p_c))\
                         - grad_phi_dK(phi_b,dK_a(p_c))

    V=unstack(convective_velocity,"dk")
    convective_velocity=Field(self.phi_o.geometry,values=vec(x=V[0],y=V[1]))
    return convective_velocity

  #def compute_anisotropic_viscosity_effect(self):
    # reformulate differential solver
    
  def phi_w_momentum_eq(self,phi_w,phi_o, dt):
    #grad_phi_w=field.spatial_gradient(self.phi_w,self.phi_w.boundary)
    p_c=self.compute_p_c(phi_w,phi_o)
    w_advection_term = dt * advect.semi_lagrangian((phi_o),
                                                    self.compute_convective_velocity(phi_w,phi_o,dK_w),
                                                    dt).sample(phi_w.geometry)
    o_advection_term = dt * advect.semi_lagrangian((phi_w),
                                                    self.compute_convective_velocity(phi_o,phi_w,dK_o),
                                                    dt).sample(phi_w.geometry)
    w_diffusion_term = dt * anisotropic_diffusion.implicit(phi_w,K_w(p_c), dt=dt,correct_skew=False).sample(phi_w.geometry)
    o_diffusion_term = dt * anisotropic_diffusion.implicit(phi_o,K_o(p_c), dt=dt,correct_skew=False).sample(phi_w.geometry)

    return phi_w + phi_w.with_values(w_advection_term + o_advection_term) + phi_w.with_values(w_diffusion_term - o_diffusion_term)
  
  def phi_o_momentum_eq(self,phi_o,phi_w, dt):
    #grad_phi_w=field.spatial_gradient(phi_w,phi_w.boundary)
    p_c=self.compute_p_c(phi_w,phi_o)
    w_advection_term = dt * advect.semi_lagrangian((phi_o),
                                                    self.compute_convective_velocity(phi_o,phi_w,dK_w),
                                                    dt).sample(phi_o.geometry)
    o_advection_term = dt * advect.semi_lagrangian((phi_w),
                                                    self.compute_convective_velocity(phi_w,phi_o,dK_o),
                                                    dt).sample(phi_o.geometry)
    w_diffusion_term = dt * anisotropic_diffusion.implicit(phi_w,K_w(p_c), dt=dt,correct_skew=False).sample(phi_o.geometry)
    o_diffusion_term = dt * anisotropic_diffusion.implicit(phi_o,K_o(p_c), dt=dt,correct_skew=False).sample(phi_o.geometry)

    return phi_o + phi_o.with_values(w_advection_term + o_advection_term) + phi_o.with_values(o_diffusion_term - w_diffusion_term)


  def implicit_time_step(self, phi_w,phi_o, dt):
    new_phi_w = math.solve_linear(self.phi_w_momentum_eq, phi_w, self.w_advection_solver(phi_w),phi_o, dt=-dt)
    new_phi_o = math.solve_linear(self.phi_o_momentum_eq, phi_o, self.o_advection_solver(phi_o),phi_w, dt=-dt)
    return new_phi_w,new_phi_o
  

class two_phase_flow_SF(object):
  def __init__(self,phi_w,phi_o,dtphi_w_1,dtphi_o_1,dt,w_advection_solver,o_advection_solver):
    #self.v0=v0
    self.phi_w=phi_w
    self.phi_o=phi_o
    self.dtphi_o_1=dtphi_o_1
    self.dtphi_w_1=dtphi_w_1
    self.dt=dt
    self.p=None
    self.w_advection_solver=w_advection_solver
    self.o_advection_solver=o_advection_solver


  def compute_p_c(self,phi_w,phi_o):
    p_c=phi_o.sample(phi_o.geometry) -\
    phi_w.sample(phi_w.geometry)
    return p_c

  def compute_convective_velocity(self,phi_a,phi_b,dK_a,dK_b):
    p_c=self.compute_p_c(self.phi_w,self.phi_o)
    convective_velocity = grad_phi_dK(phi_a,dK_a(p_c))\
                         - grad_phi_dK(phi_b,dK_b(p_c))

    V=unstack(convective_velocity,"dk")
    convective_velocity=Field(self.phi_o.geometry,values=vec(x=V[0],y=V[1]))
    return convective_velocity

  #def compute_anisotropic_viscosity_effect(self):
    # reformulate differential solver
    
  def phi_w_momentum_eq(self,phi_w,phi_o, dt):
    #grad_phi_w=field.spatial_gradient(self.phi_w,self.phi_w.boundary)
    p_c=self.compute_p_c(phi_w,phi_o)
    w_advection_term = dt * advect.semi_lagrangian((phi_w),
                                                    self.compute_convective_velocity(phi_w,phi_o,dK_w,dK_o),
                                                    dt).sample(phi_w.geometry)

    w_diffusion_term = dt * anisotropic_diffusion.implicit(phi_w,K_w(p_c), dt=dt,correct_skew=False).sample(phi_w.geometry)
    #o_diffusion_term = dt * anisotropic_diffusion.implicit(phi_o,K_o(p_c), dt=dt,correct_skew=False).sample(phi_w.geometry)

    pressure_chage_term = dt * (self.dtphi_o_1/(dsdpc(p_c)))

    return phi_w + phi_w.with_values(pressure_chage_term) + phi_w.with_values(w_advection_term) - phi_w.with_values(w_diffusion_term)
  
  def phi_o_momentum_eq(self,phi_o,phi_w, dt):
    #grad_phi_w=field.spatial_gradient(phi_w,phi_w.boundary)
    p_c=self.compute_p_c(phi_w,phi_o)

    o_advection_term = dt * advect.semi_lagrangian((phi_o),
                                                    self.compute_convective_velocity(phi_w,phi_o,dK_w,dK_o),
                                                    dt).sample(phi_o.geometry)

    o_diffusion_term = dt * anisotropic_diffusion.implicit(phi_o,K_o(p_c), dt=dt,correct_skew=False).sample(phi_o.geometry)

    pressure_chage_term = dt * (self.dtphi_w_1/(dsdpc(p_c)))

    return phi_o + phi_o.with_values(pressure_chage_term) + phi_o.with_values(o_advection_term) - phi_o.with_values(o_diffusion_term)
  
  def compute_phi_k(self,phi_w,phi_o,phi_w_1,phi_o_1,dt):
    return (phi_w-phi_w_1)/dt,(phi_o-phi_o_1)/dt


  def implicit_time_step(self, phi_w,phi_o, dt):
    new_phi_w = math.solve_linear(self.phi_w_momentum_eq, phi_w, self.w_advection_solver(phi_w),phi_o, dt=-dt)
    new_phi_o = math.solve_linear(self.phi_o_momentum_eq, phi_o, self.o_advection_solver(phi_o),phi_w, dt=-dt)
    self.dtphi_w_1,self.dtphi_o_1=self.compute_phi_k(new_phi_w,new_phi_o,phi_w,phi_o, dt)
    return new_phi_w,new_phi_o
  


class two_phase_flow_RD(object):
  def __init__(self,phi_w,phi_o,dtphi_w_1,dtphi_o_1,dt,w_advection_solver,o_advection_solver):
    #self.v0=v0
    self.phi_w=phi_w
    self.phi_o=phi_o
    self.dtphi_o_1=dtphi_o_1
    self.dtphi_w_1=dtphi_w_1
    self.dt=dt
    self.p=None
    self.w_advection_solver=w_advection_solver
    self.o_advection_solver=o_advection_solver
    self.K_o=K_o
    self.K_w=K_w


  def compute_p_c(self,phi_w,phi_o):
    p_c=phi_o.sample(phi_o.geometry) -\
    phi_w.sample(phi_w.geometry)
    return p_c

  def compute_convective_velocity(self,phi_a,phi_b,dK_a,dK_b):
    p_c=self.compute_p_c(self.phi_w,self.phi_o)
    convective_velocity = grad_phi_dK(phi_a,dK_a(p_c))\
                         - grad_phi_dK(phi_b,dK_b(p_c))

    V=unstack(convective_velocity,"dk")
    convective_velocity=Field(self.phi_o.geometry,values=vec(x=V[0],y=V[1]))
    return convective_velocity

  #def compute_anisotropic_viscosity_effect(self):
    # reformulate differential solver
    
  def phi_w_momentum_eq(self,phi_w,phi_o, dt):
    #grad_phi_w=field.spatial_gradient(self.phi_w,self.phi_w.boundary)
    p_c=self.compute_p_c(phi_w,phi_o)

    w_advection_term = phi_sum(self.compute_convective_velocity(phi_w,phi_o,dK_w,dK_o)*phi_w.gradient(),"vector").sample(phi_w.geometry)

    x,y=unstack(phi_sum(self.K_o(p_c),"KK"),"k")
    spatial_diffusion=Field(phi_w.geometry,values=vec(x=x,y=y))
    w_diffusion_term=phi_w.with_values(phi_sum(phi_w.gradient(2)*spatial_diffusion,"vector"))

    pressure_chage_term = (self.dtphi_o_1)

    return phi_w + dt * (phi_w.with_values(pressure_chage_term) + phi_w.with_values(w_advection_term) - phi_w.with_values(w_diffusion_term))
  
  def phi_o_momentum_eq(self,phi_o,phi_w, dt):
    #grad_phi_w=field.spatial_gradient(phi_w,phi_w.boundary)
    p_c=self.compute_p_c(phi_w,phi_o)

    w_advection_term = phi_sum(self.compute_convective_velocity(phi_w,phi_o,dK_w,dK_o)*phi_w.gradient(),"vector").sample(phi_o.geometry)

    x,y=unstack(phi_sum(self.K_w(p_c),"KK"),"k")
    spatial_diffusion=Field(phi_o.geometry,values=vec(x=x,y=y))
    w_diffusion_term=phi_o.with_values(phi_sum(phi_w.gradient(2)*spatial_diffusion,"vector"))

    pressure_chage_term = (self.dtphi_w_1)

    return phi_o + dt * (phi_o.with_values(pressure_chage_term) + phi_o.with_values(w_advection_term) - phi_o.with_values(w_diffusion_term))
  
  def compute_phi_k(self,phi_w,phi_o,phi_w_1,phi_o_1,dt):
    return (phi_w-phi_w_1)/dt,(phi_o-phi_o_1)/dt


  def implicit_time_step(self, phi_w,phi_o, dt):
    new_phi_w = self.phi_w_momentum_eq(phi_w,phi_o, dt)
    new_phi_o = self.phi_o_momentum_eq(phi_o,phi_w, dt)
    self.dtphi_w_1,self.dtphi_o_1=self.compute_phi_k(new_phi_w,new_phi_o,phi_w,phi_o, dt)
    return new_phi_w,new_phi_o