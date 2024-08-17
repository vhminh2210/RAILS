#/bin/sh
python GraphEnc/setup.py build_ext --inplace 2> /dev/null
python main.py --modeltype BC_LOSS \
                --root datasets \
                --dataset d1 \
                --n_layers 2 \
                --neg_sample 1 \
                --sim_mode user_embedding \
                --epoch 5 \
                --cuda -1