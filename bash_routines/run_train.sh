#!/bin/bash

root='/share_zeta/Proxy-Sim/PhysicsSimulationDeepLearning'
cd $root
exp_name='GAN_PI_MultiP5_PointNet_LL_PDELL_D2_G2_opt_LBFGS'
echo $exp_name
python Train.py launch --directory $root'/Exps/GAN_PI/'$exp_name --epochs 200


#python Train.py launch_data_test --directory $root'/Exps/Data_sensibility_exps/GAN_PI/'$exp_name --epochs 500
