#/bin/sh
echo $PWD
python main.py --modeltype LGN \
                --root datasets \
                --dataset d1 \
                --n_layers 2 \
                --neg_sample 1 \
                --sim_mode user_embedding \
                --cuda -1