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
                --episode_max 512\
                --step_max 4096 \
                --memory 16384 \
                --nov_beta 0.0 \
                --agent_batch 64 \
                --dqn_mode ddqn \
                --eta 1.0 \
                --agent_lr 1e-4 \
                --replace_freq 5000 \
                --tau 2e-4 \
                --gamma 0.9