import math
import numpy as np
from numpy.linalg import norm
import torch

class Env():
    def __init__(self, user, observation_data, I,
                 item_pop_dict, mask_list, sim_mode, repr_user, item_emb, wild_items, args):
        self.observation = np.array(observation_data)
        self.n_observation = len(self.observation)
        self.action_space = I
        self.n_actions = len(self.action_space)
        self.user = user
        # self.item_sim_matrix = item_sim_matrix
        self.item_pop_dict = item_pop_dict
        # self.quality_dict = quality_dict
        self.mask_list = mask_list
        self.sim_mode = sim_mode
        self.repr_user = repr_user
        self.item_emb = item_emb
        self.K = args.topk
        self.eta = args.eta
        self.args = args
        self.wild_items = wild_items
        if self.wild_items == None:
            self.wild_items = []

    def build_state(self, last_obs):
        if self.args.sim_mode == 'stats':
            print('WARNING: `stats` similarity mode is deprecated! Please use `item_embedding` or `user_embedding` instead!')
            return np.array(last_obs)

        elif self.args.sim_mode == 'item_embedding':
            assert self.item_emb is not None
            state = 0
            n_obs = len(last_obs)
            cnt = len(last_obs)
            for i in range(n_obs):
                obs = last_obs[i]
                if obs in self.wild_items: 
                    cnt -= 1
                    continue
                state += (self.args.eta ** (n_obs - i - 1)) * self.item_emb(torch.IntTensor([obs]))
            # Absolute cold-start user is undefined
            if cnt == 0:
                return None
            return state / cnt

        elif self.args.sim_mode == 'user_embedding':
            assert self.repr_user is not None
            state = 0
            n_obs = len(last_obs)
            cnt = len(last_obs)
            for i in range(n_obs):
                obs = last_obs[i]
                if obs in self.wild_items:
                    cnt -= 1
                    continue
                try:
                    state += (self.args.eta ** (n_obs - i - 1)) * self.repr_user(torch.IntTensor([obs]))
                except:
                    # Out-of-range obsservation
                    cnt -= 1
                    continue
            # Absolute cold-start user is undefined
            if cnt == 0:
                return None
            return state / cnt

        else:
            raise NotImplementedError(f'sim_mode {self.args.sim_mode} not found!')

    def reset(self, observation):
        self.observation = observation
        self.n_observation = len(self.observation)
        return self.observation

    def step(self, action):
        '''
        The function returns (new_state, returned_reward, status) after action [action]
        '''
        done = False
        so = np.array(self.observation)
        s = self.build_state(self.observation)

        # If action is chosen in the previous step (due to similarity reasons), discard the step
        if so[-1] == action:
            r = -1
            return s, r, done
        else:
            act_tensor = torch.IntTensor([action])
            va = self.item_emb(act_tensor).reshape((-1)) # Action item embedding

            s, va = s.squeeze().to(self.args.device), va.to(self.args.device)

            r =  torch.dot(s, va) # Cold-start user and candidate reward
            
        s_temp_ = np.append(so, action) # Append [action] to [observations]
        observation_ = np.delete(s_temp_, 0, axis=0) # Remove the oldest [observation]
        s_ = self.build_state(observation_) # observation_ is the new state

        done = True

        return s_, r, done
