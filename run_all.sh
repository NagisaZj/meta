#! /bin/bash
source activate py3
CUDA_VISIBLE_DEVICES=0 nohup python ppo2_new.py --network mlp --optim_stepsize 6e-4 --clip_param 0.1 > try0mlp-cp-0.1-lr-6e-4.out &
CUDA_VISIBLE_DEVICES=0 nohup python ppo2_new.py --network cnn --optim_stepsize 6e-4 --clip_param 0.1 > try0cnn-cp-0.1-lr-6e-4.out &

CUDA_VISIBLE_DEVICES=1 nohup python ppo2_new.py --network mlp --optim_stepsize 6e-4 --clip_param 0.2 > try1mlp-cp-0.2-lr-6e-4.out &
CUDA_VISIBLE_DEVICES=1 nohup python ppo2_new.py --network cnn --optim_stepsize 6e-4 --clip_param 0.2 > try1cnn-cp-0.2-lr-6e-4.out &

CUDA_VISIBLE_DEVICES=2 nohup python ppo2_new.py --network mlp --optim_stepsize 6e-4 --clip_param 0.3 > try2mlp-cp-0.3-lr-6e-4.out &
CUDA_VISIBLE_DEVICES=2 nohup python ppo2_new.py --network cnn --optim_stepsize 6e-4 --clip_param 0.3 > try2cnn-cp-0.3-lr-6e-4.out &


CUDA_VISIBLE_DEVICES=3 nohup python ppo2_new.py --network mlp --optim_stepsize 2e-4 --clip_param 0.1 > try3mlp-cp-0.1-lr-2e-4.out &
CUDA_VISIBLE_DEVICES=3 nohup python ppo2_new.py --network cnn --optim_stepsize 2e-4 --clip_param 0.1 > try3cnn-cp-0.1-lr-2e-4.out &

CUDA_VISIBLE_DEVICES=4 nohup python ppo2_new.py --network mlp --optim_stepsize 2e-4 --clip_param 0.2 > try4mlp-cp-0.2-lr-2e-4.out &
CUDA_VISIBLE_DEVICES=4 nohup python ppo2_new.py --network cnn --optim_stepsize 2e-4 --clip_param 0.2 > try4cnn-cp-0.2-lr-2e-4.out &

CUDA_VISIBLE_DEVICES=5 nohup python ppo2_new.py --network mlp --optim_stepsize 2e-4 --clip_param 0.3 > try5mlp-cp-0.3-lr-2e-4.out &
CUDA_VISIBLE_DEVICES=5 nohup python ppo2_new.py --network cnn --optim_stepsize 2e-4 --clip_param 0.3 > try5cnn-cp-0.3-lr-2e-4.out &


CUDA_VISIBLE_DEVICES=6 nohup python ppo2_new.py --network mlp --optim_stepsize 1e-3 --clip_param 0.1 > try6mlp-cp-0.1-lr-1e-3.out &
CUDA_VISIBLE_DEVICES=6 nohup python ppo2_new.py --network cnn --optim_stepsize 1e-3 --clip_param 0.1 > try6cnn-cp-0.1-lr-1e-3.out &

CUDA_VISIBLE_DEVICES=7 nohup python ppo2_new.py --network mlp --optim_stepsize 1e-3 --clip_param 0.2 > try7mlp-cp-0.2-lr-1e-3.out &
CUDA_VISIBLE_DEVICES=7 nohup python ppo2_new.py --network cnn --optim_stepsize 1e-3 --clip_param 0.2 > try7cnn-cp-0.2-lr-1e-3.out &

CUDA_VISIBLE_DEVICES=8 nohup python ppo2_new.py --network mlp --optim_stepsize 1e-3 --clip_param 0.3 > try8mlp-cp-0.3-lr-1e-3.out &
CUDA_VISIBLE_DEVICES=8 nohup python ppo2_new.py --network cnn --optim_stepsize 1e-3 --clip_param 0.3 > try8cnn-cp-0.3-lr-1e-3.out &


CUDA_VISIBLE_DEVICES=9 nohup python ppo2_new.py --network mlp --optim_stepsize 6e-5 --clip_param 0.1 > try9mlp-cp-0.1-lr-6e-5.out &
CUDA_VISIBLE_DEVICES=9 nohup python ppo2_new.py --network cnn --optim_stepsize 6e-5 --clip_param 0.1 > try9cnn-cp-0.1-lr-6e-5.out &

CUDA_VISIBLE_DEVICES=1 nohup python ppo2_new.py --network mlp --optim_stepsize 6e-5 --clip_param 0.2 > try11mlp-cp-0.2-lr-6e-5.out &
CUDA_VISIBLE_DEVICES=1 nohup python ppo2_new.py --network cnn --optim_stepsize 6e-5 --clip_param 0.2 > try11cnn-cp-0.2-lr-6e-5.out &

CUDA_VISIBLE_DEVICES=2 nohup python ppo2_new.py --network mlp --optim_stepsize 6e-5 --clip_param 0.3 > try22mlp-cp-0.3-lr-6e-5.out &
CUDA_VISIBLE_DEVICES=2 nohup python ppo2_new.py --network cnn --optim_stepsize 6e-5 --clip_param 0.3 > try22cnn-cp-0.3-lr-6e-5.out &