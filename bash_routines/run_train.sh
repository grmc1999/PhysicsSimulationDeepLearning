#!/bin/bash

root='/home/gmorenoc/code_/'
cd $root'Encoding_Methods_For_VAE'

python Multi_train.py train --data_exp_path $root'Local_exps/Exps/MNIST' --module_iteration_path VAE_ViT \
				--dataset_dir $root'Localdata/MNIST' \
				--dataset_name 'MNIST'
#python Multi_train.py train --data_exp_path $root'Local_exps/Exps/CelebA_64' --module_iteration_path VAE_DNN \
#				--dataset_dir $root'Localdata/CelebA_64' \
#				--dataset_name 'Flowers102'
