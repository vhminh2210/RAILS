import math
import numpy as np
from numpy.linalg import norm
import torch

class Env():
    def __init__(self, user, observation_data, I,
                 item_sim_matrix, item_pop_dict, quality_dict, mask_list, sim_mode, repr_user, item_emb, args):
        self.observation = np.array(observation_data)
        self.n_observation = len(self.observation)
        self.action_space = I
        self.n_actions = len(self.action_space)
        self.user = user
        self.item_sim_matrix = item_sim_matrix
        self.item_pop_dict = item_pop_dict
        self.quality_dict = quality_dict
        self.mask_list = mask_list
        self.sim_mode = sim_mode
        self.repr_user = repr_user
        self.item_emb = item_emb
        self.K = args.topk
        self.eta = args.eta
        self.args = args

    def build_state(self, last_obs):
        if self.args.sim_mode == 'stats':
            return np.array(last_obs)

        elif self.args.sim_mode == 'item_embedding':
            assert self.item_emb is not None
            state = 0
            n_obs = len(last_obs)
            for i in range(n_obs):
                obs = last_obs[i]
                state += (self.args.eta ** (n_obs - i - 1)) * self.item_emb(torch.IntTensor([obs]))
            return state / torch.linalg.norm(state)

        elif self.args.sim_mode == 'user_embedding':
            assert self.repr_user is not None
            state = 0
            n_obs = len(last_obs)
            for i in range(n_obs):
                obs = last_obs[i]
                state += (self.args.eta ** (n_obs - i - 1)) * self.repr_user(torch.IntTensor([obs]))
            return state / torch.linalg.norm(state)

        else:
            raise NotImplementedError(f'sim_mode {args.sim_mode} not found!')

    def reset(self, observation):
        self.observation = observation
        self.n_observation = len(self.observation)
        return self.observation

    def step(self, action):
        '''
        The function returns (new_state, returned_reward, status) after action [action]
        '''
        done = False
        so = self.observation
        s = self.build_state(self.observation)

        # If action is chosen in the previous step (due to similarity reasons), discard the step
        if so[-1] == action:
            self.item_sim_matrix[str(so[-1])][str(action)] = 0
            r = -1
        else:
            quality = self.quality_dict[str(action)]
            # Novelty score
            r_div = self.args.nov_beta * quality * 1 / math.log((self.item_pop_dict[str(action)] + 1.1), 10)
            # Similarity score
            r_acc = 0
            for i in range(self.n_observation):
                cur_tensor = torch.IntTensor([so[-(i + 1)]])
                act_tensor = torch.IntTensor([action])
                # Cosine Similarity using item embedding
                if self.sim_mode == 'item_embedding':
                    vi = self.item_emb(cur_tensor).reshape((-1)) # Item embedding
                    va = self.item_emb(act_tensor).reshape((-1))
                    r_acc += (self.eta ** i) * (np.dot(vi, va) / (norm(vi) * norm(va)))
                
                # Cosine Similarity using user-item statistics
                elif self.sim_mode == 'stats':
                    if str(s[-(i + 1)]) in self.item_sim_matrix.keys():
                        # If candidate action has no similarity with past action -> skip this pair
                        if str(action) in self.item_sim_matrix[str(s[-(i + 1)])].keys():
                            r_acc += (self.eta ** i) * self.item_sim_matrix[str(s[-(i + 1)])][str(action)]

                # Cosine Similarity using representative user and item embeddings
                elif self.sim_mode == 'user_embedding':
                    vi = self.repr_user(cur_tensor).reshape((-1)) # Item's representative user
                    va = self.item_emb(act_tensor).reshape((-1))
                    r_acc += (self.eta ** i) * (np.dot(vi, va) / (norm(vi) * norm(va)))

            # Normalized similarity score to [-1, 1]
            r = r_acc / self.n_observation + r_div
        if r > 0:
            # If the reward is positive, append [action] to [observations]
            s_temp_ = np.append(so, action)
            # Remove the oldest [observation]
            observation_ = np.delete(s_temp_, 0, axis=0)
        else:
            observation_ = so
        s_ = self.build_state(so)

        return s_, r, done
