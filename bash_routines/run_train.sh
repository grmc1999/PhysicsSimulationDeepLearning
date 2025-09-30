#!/bin/bash

#root='/share_zeta/Proxy-Sim/PhysicsSimulationDeepLearning'
root=$1
cd $root
exp_name='GAN_PI_MultiP5_PointNet_LL_PDELL_D2_G2'
exp_name='GAN_PI_LL_s10_PDELL_D_os_1_G_os_2'
exp_name='GAN_PI_LL_s10_PDELL_D_os_3_G_os_5'
exp_name='GAN_PI_LL_s10_PDELL_D_os_1_G_os_2' # NOT STARTED
#exp_name='GAN_PI_LL_s10_PDELL_D_os_2_G_os_1' # IN PROCESS
#exp_name='GAN_PI__MultiP20_PointNet_LL_PDELL'

data_name='GAN_PI'
#data_name='2D_Poisson/2D_poisson_eq_D_N_R_quad_f_Linear_Linear_64x64'
#data_name='2D_Poisson/2D_poisson_eq_Dirichlet_Const_Quad_f_Const_64x64'
#data_name='3D_Poisson/3D_poisson_eq_Dirichlet_BC_64x64x64'
#data_name='2D_NS/2D_NS_eq_Dirichlet_BC_S10x10_T10s'
#data_name='3D_NS/3D_poisson_eq_Dirichlet_BC_64x64x64'
#data_name='2D_2P_Darcy_flow/2D_NS_eq_Dirichlet_BC_S10x10_T10s'
#data_name='2D_1P_Darcy_flow_SD/3D_poisson_eq_Dirichlet_BC_64x64x64'
#data_name='2D_1P_non_linear_Darcy_flow_SD/3D_poisson_eq_Dirichlet_BC_64x64x64'
#data_name='3D_1P_Darcy_flow_SD/3D_poisson_eq_Dirichlet_BC_64x64x64'
#data_name='3D_1P_non_linear_Darcy_flow_SD/3D_poisson_eq_Dirichlet_BC_64x64x64'
#data_name='3D_Poisson/3D_poisson_eq_Dirichlet_BC_64x64x64'
#data_name='2D_NS/2D_NS_eq_Dirichlet_BC_S10x10_T10s'
#data_name='3D_NS/3D_poisson_eq_Dirichlet_BC_64x64x64'
data_name='2D_2P_Darcy_flow/2D_NS_eq_Dirichlet_BC_S10x10_T10s'


echo $data_name
echo $exp_name
python3 Train.py launch --directory $root'/Exps/'$data_name'/'$exp_name --epochs 1000


#python Train.py launch_data_test --directory $root'/Exps/Data_sensibility_exps/GAN_PI/'$exp_name --epochs 500
