# RAILS
Code repository for the paper **Bake Two Cakes with One Oven: RL for Defusing Popularity Bias and Cold-start in Third-Party Library Recommendations**

<p align="center">
    <img src="https://github.com/vhminh2210/GraphICS/blob/main/graphics.png" height="50%">
</p>

---

## Dependencies

This implementation builds upon the following repositories:

- **[BC-Loss](https://github.com/anzhang314/BC-Loss):** Used for the `LightGCN` backbone trained with `BC-Loss`.
- **[DNaIR](https://github.com/shixiaoyu0216/DNaIR):** Provides the `DQN` environment setup.

---

## Overview

The GraphICS framework consists of two main stages:

1. **Encoder Pretraining:** The encoder is pretrained following the format defined in [BC-Loss](https://github.com/anzhang314/BC-Loss).
2. **RL Agent Training:** The pretrained encoder is used for reinforcement learning.

### Dataset Format

The dataset directory must include the following files:

- `train.txt`
- `val.txt` *(Required for encoder training and loading)* 
- `test.txt`
- `query.txt` *(Optional)* contains known cold-start interaction history for cold-start setting

Each file should follow this format:

```
<user-id> <item1-id> <item2-id> ...
<user-id> <item3-id> <item4-id> <item5-id> ...
```

All `id` values should be numerical and **zero-based**. Refer to provided datasets (`DS1`, `DS2`, `DS3`) for more details.

### Evaluation Modes

The script allows different evaluation modes via command-line flags:

- `--eval_graph`: Evaluate the pretrained encoder.
- `--eval_query`: Use a cold-start recommendation scheme.
- *(Default mode)*: Uses interaction-split schema.

---

## Usage Examples

### 1. End-to-End Training

This command trains both the encoder and the RL agent sequentially. It is the **recommended** approach for running experiments with `GraphICS`.

```sh
python GraphEnc/setup.py build_ext --inplace 2> /dev/null
python main.py --modeltype BC_LOSS \
                --cuda -1 \  # Use CPU (Set to device ID, e.g., 0, for GPU)
                --num_workers 4 \
                --root datasets \
                --dataset d1 \
                --epoch 100 \
                --enc_batch_size 1024 \
                --enc_lr 1e-4 \
                --n_layers 2 \
                --neg_sample 128 \
                --freeze_epoch 5 \  # End of encoder parameters
                --sim_mode user_embedding \  # Start of RL parameters
                --epoch_max 1 \
                --step_max 4 \
                --memory 16384 \
                --nov_beta 0.0 \
                --agent_batch 1024 \
                --eta 1.0 \
                --agent_lr 1e-3 \
                --replace_freq -1 \
                --num_hidden 256 \
                --tau 1e-4 \
                --gamma 0.999 \
                --cql_mode cql_H \
                --cql_alpha 5.5 \
                --user_lam 0.5 \
                --dqn_mode ddqn \
                --dueling_dqn \
                --n_augment 3 \
                --n_aug_scale 5 \
                --rare_thresh 0.1 \
                --seq_ratio 0.3 \
                --rare_ratio 0.2 \
                --rand_ratio 0.5 \
                --topk 10 \
                --episode_batch 128 \
                --all_episodes 
```

### 2. Using a Pretrained Encoder

If an encoder has already been trained, it can be loaded directly for RL training:

```sh
python GraphEnc/setup.py build_ext --inplace 2> /dev/null
python main.py --modeltype BC_LOSS \
                --pretrained_graph \
                --ckpt_dir weights/d2-fold/Round3/BC_LOSS-LGN \
                --ckpt n_layers=2tau1=0.07tau2=0.1w=0.5 \  # Specify checkpoint
                --cuda -1 \
                --root datasets \
                --dataset d1 \  # End of encoder parameters
                --sim_mode user_embedding \  # Start of RL parameters
                ...  # Add similar hyperparameters
```
