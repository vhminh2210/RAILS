#/bin/sh
# python GraphEnc/setup.py build_ext --inplace 2> /dev/null
python main.py --modeltype BC_LOSS \
                --cuda -1 \
                --epoch 409 \
                --pretrained_graph \
                --enc_batch_size 2048 \
                --enc_lr 5e-5 \
                --root datasets \
                --dataset d1 \
                --n_layers 2 \
                --neg_sample 128 \
                --freeze_epoch 5 \
                --sim_mode user_embedding \
                --epoch_max 10 \
                --episode_max 64\
                --step_max 16 \
                --memory 16394 \
                --nov_beta 0.0 \
                --agent_batch 2048 \
                --dqn_mode ddqn \
                --eta 1.0 \
                --agent_lr 1e-5 \
                --replace_freq 10000 \
                --num_hidden 256 \
                --tau 0.0001 \
                --gamma 0.999 \
                --cql_mode cql_H \
                --dueling_dqn \
                --n_augment 10 \
                --n_aug_scale 5 \
                --rare_thresh 0.1 \
                --seq_ratio 0.3 \
                --rare_ratio 0.2 \
                --rand_ratio 0.5 \
                --topk 10 \
                --eval_freq 50 \
                --episode_batch 2 \
                --eval_graph \
                # --all_episodes