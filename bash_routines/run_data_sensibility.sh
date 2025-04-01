#!/bin/bash

root='/share_zeta/Proxy-Sim/PhysicsSimulationDeepLearning'
cd $root

python Train.py launch_data_test --directory $root'/Exps/Data_sensibility_exps/GAN_PI/GAN_PI_LL_s10_PDELL' --epochs 500
