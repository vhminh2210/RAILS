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
                --step_max 4086 \
                --memory 16380 \
                --nov_beta 0.0 \
                --agent_batch 64 \
                --dqn_mode ddqn \
                --eta 1.0 \
                --agent_lr 1e-4 \
                --replace_freq 1000 \
                --tau 0.001 \
                --gamma 0.9 \
                --cql_mode cql_Rho