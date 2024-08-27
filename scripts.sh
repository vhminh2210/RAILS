#/bin/sh
# python GraphEnc/setup.py build_ext --inplace 2> /dev/null
python main.py --modeltype BC_LOSS \
                --epoch 10 \
                --root datasets \
                --dataset d1 \
                --n_layers 2 \
                --neg_sample 64 \
                --sim_mode user_embedding \
                --freeze_epoch 5 \
                --cuda -1 \
                --episode_max 64\
                --step_max 4096 \
                --memory 8192 \
                --nov_beta 0.0 \
                --agent_batch 32 \
                --dqn_mode ddqn \
                --eta 1.0 \
                --agent_lr 0.001