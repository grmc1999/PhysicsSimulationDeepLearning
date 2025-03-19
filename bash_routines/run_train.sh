#!/bin/bash

root='/share_zeta/Proxy-Sim/PhysicsSimulationDeepLearning'
cd $root

#python Train.py launch --directory $root'/Exps/GAN_PI/GAN_PI_w_l3' --epochs 800

python Train.py launch_data_test --directory $root'/Exps/Data_sensibility/GAN_PI/GAN_PI_LL_s10_PDELL' --epochs 500
