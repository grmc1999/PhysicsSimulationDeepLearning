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
  
from phiml import math as pmath
import sympy

from phiml.math import sum as phi_sum
from phiml import math as pmath
import numpy as np
from scipy import ndimage
import sympy
from Differentiable_simulation import SWC,SOR,Sw,lam,Pi,K_rw0,K_ro0,Pc_,Sw_Pc

K_s=K_s * 9.869233e-13

class two_phase_flow_RD(object):
  def __init__(self,phi_w,phi_o,dtphi_w_1,dtphi_o_1,dt,por,mu_w,mu_o,K_s,kr_w,kr_o,Pc_args):
    #self.v0=v0
    self.phi_w=phi_w
    self.phi_o=phi_o
    self.dtphi_o_1=dtphi_o_1
    self.dtphi_w_1=dtphi_w_1
    self.dt=dt
    self.p=None
    self.K_l=K_s
    self.por=por
    self.mu_w=mu_w
    self.mu_o=mu_o

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

    self.Pc_f=lambda sw: sympy.lambdify((Sw,SOR,SWC,lam,Pi),Pc)(sw,*tuple(Pc_args.values()))
    self.Sw_Pc_f=lambda sw: sympy.lambdify((Pc_,SOR,SWC,lam,Pi),Sw_Pc)(sw,*tuple(Pc_args.values()))
    
    self.K_rw_f=lambda sw: sympy.lambdify((Sw,SOR,SWC,lam,Pi,K_rw0),K_rw)(sw,*tuple(Pc_args.values()),kr_w)
    self.K_ro_f=lambda sw: sympy.lambdify((Sw,SOR,SWC,lam,Pi,K_ro0),K_ro)(sw,*tuple(Pc_args.values()),kr_o)
    self.dK_rw_f=lambda sw: sympy.lambdify((Sw,SOR,SWC,lam,Pi,K_rw0),sympy.diff(K_rw,Sw))(sw,*tuple(Pc_args.values()),0.3)
    self.dK_ro_f=lambda sw: sympy.lambdify((Sw,SOR,SWC,lam,Pi,K_ro0),sympy.diff(K_ro,Sw))(sw,*tuple(Pc_args.values()),0.5)

    
    self.dScdPc_f=lambda sw: sympy.lambdify((Pc_,SOR,SWC,lam,Pi),dScdPc)(sw,*tuple(Pc_args.values()))

    self.K_w=lambda K_l,p_c:stack(
          [stack([K_l*self.K_rw_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_w*self.dScdPc_f(p_c)),math.zeros_like(p_c)],batch("k") ),
          stack([math.zeros_like(p_c),K_l*self.K_rw_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_w*self.dScdPc_f(p_c))],batch("k") )],batch("KK"))

    self.K_o=lambda K_l,p_c:stack(
        [stack([K_l*self.K_ro_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_o*self.dScdPc_f(p_c)),math.zeros_like(p_c)],batch("k") ),
        stack([math.zeros_like(p_c),K_l*self.K_ro_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_o*self.dScdPc_f(p_c))],batch("k") )],batch("KK"))

    self.dK_w=lambda K_l,p_c:stack(
        [stack([K_l*self.dK_rw_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_w),math.zeros_like(p_c)],batch("dk") ),
        stack([math.zeros_like(p_c),K_l*self.dK_rw_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_w)],batch("dk") )],batch("dKK"))

    self.dK_o=lambda K_l,p_c:stack(
        [stack([K_l*self.dK_ro_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_o),math.zeros_like(p_c)],batch("dk") ),
        stack([math.zeros_like(p_c),K_l*self.dK_ro_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_o)],batch("dk") )],batch("dKK"))
    
    self.grad_phi_dK = lambda phi_a,dK_a:(math.dot(field.spatial_gradient(phi_a,phi_a.boundary).sample(phi_a.geometry),"vector",dK_a,"dKK"))


  def compute_p_c(self,phi_w,phi_o):
    p_c=phi_o.sample(phi_o.geometry) -\
    phi_w.sample(phi_w.geometry)
    p_c=pmath.clip(p_c,lower_limit=self.Pc_f(1-Sw_args["SOC"]),upper_limit=self.Pc_f(Sw_args["SWR"]))
    return p_c

  def compute_convective_velocity(self,phi_a,phi_b,dK_a,dK_b):
    p_c=self.compute_p_c(phi_a,phi_b)
    convective_velocity = self.grad_phi_dK(phi_a,dK_a(self.K_l,p_c))\
                         - self.grad_phi_dK(phi_b,dK_b(self.K_l,p_c))

    V=unstack(convective_velocity,"dk")
    convective_velocity=Field(self.phi_o.geometry,values=vec(x=V[0],y=V[1]))
    return convective_velocity
  
  def phi_w_pde(self,phi_w,phi_o,dtphi_o):
    p_c=self.compute_p_c(phi_w,phi_o)
    
    w_advection_term = phi_sum(
      self.compute_convective_velocity(phi_w,phi_o,self.dK_w,self.dK_w)*phi_w.gradient(),
      "vector").sample(phi_w.geometry)
    
    x,y=unstack(phi_sum(self.K_w(self.K_l,p_c),"KK"),"k")
    spatial_diffusion=Field(phi_w.geometry,values=vec(x=x,y=y))
    w_diffusion_term=phi_w.with_values(phi_sum(phi_w.gradient(2)*spatial_diffusion,"vector"))

    pressure_chage_term = (dtphi_o.values)

    return phi_w.with_values(pressure_chage_term) + phi_w.with_values(w_advection_term) - phi_w.with_values(w_diffusion_term)
  
  def phi_o_pde(self,phi_o,phi_w,dtphi_w):
    p_c=self.compute_p_c(phi_w,phi_o)
    
    w_advection_term = phi_sum(
      self.compute_convective_velocity(phi_w,phi_o,self.dK_o,self.dK_o)*phi_o.gradient(),
      "vector").sample(phi_o.geometry)
    
    x,y=unstack(phi_sum(self.K_o(self.K_l,p_c),"KK"),"k")
    spatial_diffusion=Field(phi_o.geometry,values=vec(x=x,y=y))
    w_diffusion_term=phi_o.with_values(phi_sum(phi_o.gradient(2)*spatial_diffusion,"vector"))

    pressure_chage_term = (dtphi_w.values)

    return phi_o.with_values(pressure_chage_term) + phi_o.with_values(w_advection_term) - phi_o.with_values(w_diffusion_term)
  
  def RK4(self,phi_w,phi_o,dt):

    K_o1=self.phi_o_pde(phi_o,phi_w,self.dtphi_w_1)
    K_w1=self.phi_w_pde(phi_w,phi_o,self.dtphi_o_1)

    K_o2=self.phi_o_pde(phi_o+0.5*K_o1.values*dt,phi_w+0.5*K_w1.values*dt,K_w1)
    K_w2=self.phi_w_pde(phi_w+0.5*K_w1.values*dt,phi_o+0.5*K_o1.values*dt,K_o1)

    K_o3=self.phi_o_pde(phi_o+0.5*K_o2.values*dt,phi_w+0.5*K_w2.values*dt,K_w2)
    K_w3=self.phi_w_pde(phi_w+0.5*K_w2.values*dt,phi_o+0.5*K_o2.values*dt,K_o2)

    K_o4=self.phi_o_pde(phi_o+K_o3.values*dt,phi_w+K_w3.values*dt,K_w3)
    K_w4=self.phi_w_pde(phi_w+K_w3.values*dt,phi_o+K_o3.values*dt,K_o3)

    self.dtphi_o_1 = (1/6)  * (K_o1 + 2*K_o2 + 2*K_o3 + K_o4)
    self.dtphi_w_1 = (1/6)  * (K_w1 + 2*K_w2 + 2*K_w3 + K_w4)

    phi_o = phi_o.with_values(pmath.finite_fill(pmath.clip(phi_o.values + dt * (1/6)  * (K_o1 + 2*K_o2 + 2*K_o3 + K_o4).values,lower_limit=0.0,upper_limit=1e6)))
    phi_w = phi_w.with_values(pmath.finite_fill(pmath.clip(phi_w.values + dt * (1/6)  * (K_w1 + 2*K_w2 + 2*K_w3 + K_w4).values,lower_limit=0.0,upper_limit=phi_o.values)))
    return phi_w,phi_o
  
  def compute_phi_k(self,phi_w,phi_o,phi_w_1,phi_o_1,dt):
    return (phi_w-phi_w_1)/dt,(phi_o-phi_o_1)/dt


  def implicit_time_step(self, phi_w,phi_o, dt):
    new_phi_o = phi_o + dt * self.phi_o_pde(phi_o,phi_w,self.dtphi_w_1)
    new_phi_w = phi_w + dt * self.phi_w_pde(phi_w,phi_o,self.dtphi_o_1)
    self.dtphi_w_1,self.dtphi_o_
  

from copy import copy
class two_phase_flow_RD_TBK(two_phase_flow_RD):
    def __init__(self,phi_w,phi_o,dtphi_w_1,dtphi_o_1,dt,por,mu_w,mu_o,Pc_args,K_s,krwo):
        super().__init__(phi_w,phi_o,dtphi_w_1,dtphi_o_1,dt,por,mu_w,mu_o,K_s=K_s,Pc_args=Pc_args,kr_w=0.3,kr_o=0.3)
        self.krwo=krwo

        self.K_w=lambda K_l,p_c:stack(
              [stack([K_l*self.K_rw_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_w*self.dScdPc_f(p_c)),math.zeros_like(p_c)],batch("k") ),
              stack([math.zeros_like(p_c),K_l*self.K_rw_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_w*self.dScdPc_f(p_c))],batch("k") )],batch("KK"))

        self.K_o=lambda K_l,p_c:stack(
            [stack([K_l*self.K_ro_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_o*self.dScdPc_f(p_c)),math.zeros_like(p_c)],batch("k") ),
            stack([math.zeros_like(p_c),K_l*self.K_ro_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_o*self.dScdPc_f(p_c))],batch("k") )],batch("KK"))

        self.dK_w=lambda K_l,p_c:stack(
            [stack([K_l*self.dK_rw_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_w),math.zeros_like(p_c)],batch("dk") ),
            stack([math.zeros_like(p_c),K_l*self.dK_rw_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_w)],batch("dk") )],batch("dKK"))

        self.dK_o=lambda K_l,p_c:stack(
            [stack([K_l*self.dK_ro_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_o),math.zeros_like(p_c)],batch("dk") ),
            stack([math.zeros_like(p_c),K_l*self.dK_ro_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_o)],batch("dk") )],batch("dKK"))

        self.grad_phi_dK = lambda phi_a,dK_a:(math.dot(field.spatial_gradient(phi_a,phi_a.boundary).sample(phi_a.geometry),"vector",dK_a,"dKK"))

    def K_rw_f(self,x):
        krwo_=copy(self.krwo)
        i1=np.argmin(np.abs(self.krwo[:,0]-x))
        x1=krwo_[i1,0]
        y1=krwo_[i1,1]
        krwo_[i1,0]=1e6
        i2=np.argmin(np.abs(krwo_[:,0]-x))
        x2=krwo_[i2,0]
        y2=krwo_[i2,1]
        dy=(y1-y2)
        dx=(x1-x2)
        y=y1+(dy/dx)*(x-x1)
        return np.clip(y,0.0,1.0)
    
    def K_ro_f(self,x):
        krwo_=copy(self.krwo)
        i1=np.argmin(np.abs(krwo_[:,0]-x))
        x1=krwo_[i1,0]
        y1=krwo_[i1,2]
        krwo_[i1,0]=1e6
        i2=np.argmin(np.abs(krwo_[:,0]-x))
        x2=krwo_[i2,0]
        y2=krwo_[i2,2]
        dy=(y1-y2)
        dx=(x1-x2)
        y=y1+(dy/dx)*(x-x1)
        return np.clip(y,0.0,1.0)
    
    def dK_rw_f(self,x):
        krwo_=copy(self.krwo)
        i1=np.argmin(np.abs(krwo_[:,0]-x))
        x1=krwo_[i1,0]
        y1=krwo_[i1,1]
        krwo_[i1,0]=1e6
        i2=np.argmin(np.abs(krwo_[:,0]-x))
        x2=krwo_[i2,0]
        y2=krwo_[i2,1]
        dy=(y1-y2)
        dx=(x1-x2)
        y=y1+(dy/dx)*(x-x1)
        return (dy/dx)
    
    def dK_ro_f(self,x):
        krwo_=copy(self.krwo)
        i1=np.argmin(np.abs(krwo_[:,0]-x))
        x1=krwo_[i1,0]
        y1=krwo_[i1,2]
        krwo_[i1,0]=1e6
        i2=np.argmin(np.abs(krwo_[:,0]-x))
        x2=krwo_[i2,0]
        y2=krwo_[i2,2]
        dy=(y1-y2)
        dx=(x1-x2)
        y=y1+(dy/dx)*(x-x1)
        return (dy/dx)