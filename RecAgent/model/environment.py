import math
import numpy as np
from numpy.linalg import norm
import torch

class Env():
    def __init__(self, user, observation_data, I,
                 item_pop_dict, mask_list, sim_mode, repr_user, item_emb, wild_items, args):
        self.observation = np.array(observation_data)
        self.n_observation = len(self.observation)

        self.args = args

        self.action_space = I # Legacy code, deprecated !
        self.n_actions = len(self.action_space)

        self.n_items = self.args.n_items
        self.n_users = self.args.n_users
        
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

        self.wild_items = wild_items

        if self.wild_items == None:
            self.wild_items = []

        if self.args.sim_mode == 'crossrec':
            assert args.crossrec_bundle is not None
            self.r_bar = args.crossrec_bundle['mean_rating']
            self.tfidf_item = args.crossrec_bundle['tfidf_item']
            self.tfidf_user = args.crossrec_bundle['tfidf_user']
    
    def crossrec_similarity(self, user_p, user_q):
        mask = torch.eq(user_p.squeeze(), user_q.squeeze()).int()
        masked_p = user_p.squeeze() * mask
        masked_q = user_q.squeeze() * mask

        return torch.dot(masked_p, masked_q)
    
    def crossrec_similarity_matrix(self, user_p):
        '''
        user_p: (1, I)
        tfidf_user: (U, I)
        '''
        mask = torch.eq(self.tfidf_user, user_p).int()
        masked_p = user_p * mask # (1, I)
        masked_q = self.tfidf_user * mask # (U, I)

        # print(masked_p.shape, masked_q.shape)

        return (masked_p@masked_q.T).squeeze() # (1, U) --> (U,)
    
    def crossrec_topsim(self, user_p):
        simlist = []
        # pq_list = self.crossrec_similarity_matrix(user_p)[self.user]
        for q in range(self.n_users):
            user_q = self.tfidf_user[q]
            if not (user_q != 0).any():
                continue # Skip non-training users
            simlist.append((q, user_q, self.r_bar[q], self.crossrec_similarity(user_p, user_q)))
            # simlist.append((q, user_q, self.r_bar[q], pq_list[q]))
        
        simlist = sorted(simlist, key= lambda x : -x[2]) # Sorted decreasing by values
        return simlist[ : min(self.args.crossrec_topsim, len(simlist))]

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
                    # state += self.item_emb(torch.IntTensor([obs]))
                except:
                    # Out-of-range obsservation
                    cnt -= 1
                    continue
            # Absolute cold-start user is undefined
            if cnt == 0:
                return None
            return state / cnt
            # return state / torch.linalg.norm(state)

        elif self.args.sim_mode == 'crossrec':
            state = torch.zeros((1, self.n_items))
            n_obs = len(last_obs)
            cnt = len(last_obs)
            for i in range(n_obs):
                obs = last_obs[i]
                if obs in self.wild_items: 
                    cnt -= 1
                    continue
                state[0, obs] = self.tfidf_item[obs]

            # Absolute cold-start user is undefined
            if cnt == 0:
                return None
            
            return state
        
        else:
            raise NotImplementedError(f'sim_mode {self.args.sim_mode} not found!')

    def reset(self, observation):
        self.observation = [x for x in observation if x not in self.wild_items]
        self.n_observation = len(self.observation)
        return self.observation

    def step(self, action, action_type : int = None):
        '''
        The function returns (new_state, returned_reward, status) after action [action]
        '''
        done = False

        # If prj contains only rare observations
        if len(self.observation) == 0:
            return None, None, False

        so = np.array(self.observation)
        s = self.build_state(self.observation)

        # If action is chosen in the previous step (due to similarity reasons), discard the step
        if so[-1] == action:
            r = -1
            return s, r, done
        else:
            # Soft reward
            if self.args.sim_mode == 'crossrec':
                '''
                r = r_(user, action) = r_(p,l)
                '''
                rp_bar = self.n_observation / self.n_items
                topsim = self.crossrec_topsim(s) 

                numer, denom = 0.0, 0.0
                if len(topsim) == 0:
                    return None, None, False
                for candidate in topsim:
                    q, user_q, rq_bar, sim_pq = candidate

                    # Relevance of action in user q recommendation
                    if user_q[action] > 0:
                        r_qa = 1.
                    else:
                        r_qa = 0.

                    numer += (sim_pq * (r_qa - rq_bar))
                    denom += sim_pq
                if denom == 0:
                    return None, None, False
                r = rp_bar + float(numer / denom)
            else:
                act_tensor = torch.IntTensor([action])
                va = self.item_emb(act_tensor).reshape((-1)) # Action item embedding

                s, va = s.squeeze().to(self.args.device), va.to(self.args.device)

                r =  torch.dot(s, va) # Cold-start user and candidate reward

            # Hard reward
            if action_type is not None: 
                r += action_type
            
        s_temp_ = np.append(so, action) # Append [action] to [observations]
        observation_ = np.delete(s_temp_, 0, axis=0) # Remove the oldest [observation]
        s_ = self.build_state(observation_) # observation_ is the new state

        done = True

        return s_, r, done
