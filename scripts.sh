#/bin/sh
python GraphEnc/setup.py build_ext --inplace 2> /dev/null
python main.py --modeltype BC_LOSS \
                --epoch 10 \
                --root datasets \
                --dataset amazon-book \
                --n_layers 2 \
                --neg_sample 64 \
                --sim_mode user_embedding \
                --freeze_epoch 5 \
                --cuda -1 \
                --episode_max 256\
                --step_max 2048 \
                --memory 16394 \
                --nov_beta 0.0 \
                --agent_batch 128 \
                --dqn_mode ddqn \
                --eta 1.0 \
                --agent_lr 1e-4 \
                --replace_freq 100 \
                --tau 0.01 \
                --gamma 0.9 \
                --cql_mode cql_H \
                --dueling_dqn \
                --n_augment 10