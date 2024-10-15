#/bin/sh
# python GraphEnc/setup.py build_ext --inplace 2> /dev/null
python main.py --modeltype BC_LOSS \
                --cuda -1 \
                --epoch 100 \
                --enc_batch_size 1024 \
                --enc_lr 1e-4 \
                --root datasets/d2-fold \
                --dataset Round2 \
                --n_layers 2 \
                --neg_sample 128 \
                --freeze_epoch 5 \
                --sim_mode user_embedding \
                --epoch_max 10 \
                --episode_max 64\
                --step_max 8 \
                --memory 16384 \
                --nov_beta 0.0 \
                --agent_batch 1024 \
                --dqn_mode ddqn \
                --eta 1.0 \
                --agent_lr 1e-3 \
                --replace_freq -2 \
                --num_hidden 256 \
                --tau 8e-4 \
                --gamma 0.999 \
                --cql_mode cql_H \
                --cql_alpha 5. \
                --user_lam 0.75 \
                --dueling_dqn \
                --n_augment 3 \
                --n_aug_scale 5 \
                --rare_thresh 0.1 \
                --seq_ratio 0.3 \
                --rare_ratio 0.2 \
                --rand_ratio 0.5 \
                --topk 10 \
                --eval_freq 50 \
                --episode_batch 64 \
                --eval_query \
                --all_episodes \
                # --epsilon 0.75 \
                # --n_proposal 500 \
                # --action_proposal