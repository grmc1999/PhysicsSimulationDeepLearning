#!/bin/bash

root='/share_zeta/Proxy-Sim/PhysicsSimulationDeepLearning'
cd $root

python Train.py launch --directory $root'/Exps/GAN_PI/GAN_PI_1' --epochs 200
