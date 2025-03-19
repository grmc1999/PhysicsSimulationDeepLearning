#!/bin/bash

root='/share_zeta/Proxy-Sim/PhysicsSimulationDeepLearning'
cd $root
exp_name='GAN_PI_LL_s10_PDELL_D_os_3_G_os_5'
echo $exp_name
#python Train.py launch --directory $root'/Exps/GAN_PI/$exp_name' --epochs 800

python Train.py launch_data_test --directory $root'/Exps/Data_sensibility/GAN_PI/'$exp_name --epochs 500
