import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
import random
import math
import copy
random.seed(101)
# import seaborn as sns

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    Source: https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/05.noisy_net.ipynb
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, embd= None, 
                 dueling= False, noisy_net= False, dropout= 0.0,
                 one_hot= False, num_raw_inputs= 1200, bottleneck= 64):
        '''
        Note: num_inputs (i.e., input state embeddings dimension) is different than n_items (i.e., num_raw_inputs)!
        Normally, num_inputs < hidden_size
        '''
        super(Net, self).__init__()

        self.dueling = dueling
        self.noisy_net = noisy_net
        self.one_hot = one_hot
        self.dropout = nn.Dropout(p= dropout)

        self.softmax = nn.Softmax(dim= 1)
        self.relu = nn.ReLU()

        if self.one_hot:
            num_inputs = bottleneck
            self.prelinear = nn.Linear(num_raw_inputs, num_inputs) # Onehot downscaler

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        if self.noisy_net:
            self.linear1 = NoisyLinear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        if not self.dueling:
            # Non-dueling DQN
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            if self.noisy_net:
                self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.ln2 = nn.LayerNorm(hidden_size)

            self.mu = nn.Linear(hidden_size, num_outputs)
            if self.noisy_net:
                self.mu = NoisyLinear(hidden_size, num_outputs)

            if self.one_hot:
                assert not self.noisy_net # Not yet support noisy net + crossrec
                self.postlinear = nn.Linear(hidden_size, bottleneck)
                self.mu = nn.Linear(bottleneck, num_outputs)

        else:
            # Dueling DQN
            # Consult https://arxiv.org/pdf/1511.06581 for architecture
            try:
                assert hidden_size % 2 == 0
            except:
                raise ValueError('Number of hidden activations must divisible by 2 in Dueling DQN mode!')
            half_size = int(hidden_size / 2)
            
            # Value branch
            self.linear2_V = nn.Linear(hidden_size, half_size)
            if self.noisy_net:
                self.linear2_V = NoisyLinear(hidden_size, half_size)
            self.ln2_V = nn.LayerNorm(half_size)

            self.linear3_V = nn.Linear(half_size, 1)
            if self.noisy_net:
                self.linear3_V = NoisyLinear(half_size, 1)

            # Advantage branch
            self.linear2_A = nn.Linear(hidden_size, half_size)
            if self.noisy_net:
                self.linear2_A = NoisyLinear(hidden_size, half_size)
            self.ln2_A = nn.LayerNorm(half_size)

            self.mu = nn.Linear(half_size, num_outputs)
            if self.noisy_net:
                self.mu = NoisyLinear(half_size, num_outputs)

            if self.one_hot:
                assert not self.noisy_net # Not yet support noisy net + crossrec
                self.postlinear = nn.Linear(half_size, bottleneck)
                self.mu = nn.Linear(bottleneck, num_outputs)

        # Default init for non-noisy nets
        if not self.noisy_net:
            self.mu.weight.data.mul_(0.1)
            self.mu.bias.data.mul_(0.1)

        if embd is not None:
            self.embd = embd.weight.detach()
            try:
                assert self.embd.shape[-1] == num_outputs
            except:
                print('Output size must equal embedding size for dot product')

            try:
                assert not self.one_hot
            except:
                print('Embedding must be None for one-hot encodings!')
        else:
            self.embd = None

        # Consult https://wandb.ai/wandb_fc/tips/reports/How-to-Initialize-Weights-in-PyTorch--VmlldzoxNjcwOTg1
        self.apply(self._init_weights) 

    def _reset_noise(self, module):
        """
        Reset all noisy layers.
        Source: https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/05.noisy_net.ipynb
        """
        if not self.noisy_net:
            return
        if isinstance(module, NoisyLinear):
            module.reset_noise()
            module.reset_noise()

    def reset_noise(self):
        if not self.noisy_net:
            return
        self.apply(self._reset_noise)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, inputs):
        if self.one_hot:
            # Downscaler
            inputs = self.prelinear(inputs)

        if self.dueling:
            # Dueling DQN
            x = inputs
            # x = F.relu(self.ln0(self.linear0(x)))
            x = self.relu(self.ln1(self.dropout(self.linear1(x))))

            # Value branch
            val = self.relu(self.ln2_V(self.dropout(self.linear2_V(x))))
            val = self.linear3_V(val) # (batch_size, 1)

            # Advantage branch
            adv = self.relu(self.ln2_A(self.dropout(self.linear2_A(x))))
            if self.embd is not None:
                adv = (self.mu(adv) @ self.embd.T) # (batch_size, n_action)
            else:
                if self.one_hot:
                    adv = self.postlinear(adv)
                adv = torch.tanh(self.mu(adv)) # (batch_size, num_outputs = n_action)

            # Q-value fusion
            adv_mean = torch.mean(adv, dim= 1, keepdim= True) # (batch_size, 1)
            q_values = adv + (val - adv_mean)
        
        else:
            x = inputs
            # x = F.relu(self.ln0(self.linear0(x)))
            x = self.relu(self.ln1(self.dropout(self.linear1(x))))
            x = self.relu(self.ln2(self.dropout(self.linear2(x))))

            if self.embd is not None:
                q_values = (self.mu(x) @ self.embd.T)
            else:
                if self.one_hot:
                    adv = self.postlinear(adv)
                q_values = torch.tanh(self.mu(x))

        return q_values

class AQLProposalNet(nn.Module):
    def __init__(self, item_embd, args):
        '''
        Proposal network. Not yet support CQL_Rho !
        '''
        super(AQLProposalNet, self).__init__()
        self.embd = item_embd.weight.detach() # (n_action, embd_dim)
        self.args = args

        self.n_exploit = int(args.n_proposal * args.epsilon)
        self.n_explore = int(args.n_proposal - self.n_exploit)

        self.n_action = self.embd.shape[0]
        self.idlist = np.arange(self.n_action)

        self.rng = np.random.default_rng(seed=42)

        self.softmax = nn.Softmax(dim= 1)

    def forward(self, s, mode= 'stochastic', epsi= 0.):
        '''
        s: Batch of input states. Shape = (batch_size, embd_dim)
        self.embd: Item embedding matrix. Shape = (n_action, embd_dim)

        output: Proposal item index
        '''
        logits = (s @ self.embd.T) + epsi # (batch_size, n_action)
        # probs = logits / torch.sum(logits, 0, keepdim= True)
        probs = self.softmax(logits) # (batch_size, n_actions)

        numpy_probs = probs.detach().cpu().numpy()

        idx = []
        
        for i in range(probs.shape[0]):
            # Generate masks
            if mode == 'stochastic':
                # Exploitation
                exploit_id = self.rng.choice(self.idlist, size= self.n_exploit, replace= False, p= numpy_probs[i]).tolist()
                # Exploration
                explore_id = self.rng.choice(self.idlist, size= self.n_explore, replace= False).tolist()
            else:
                # Pure exploitation
                exploit_id = sorted(self.idlist, key= lambda x: -numpy_probs[i, x])[: (self.n_exploit + self.n_explore)]
                explore_id = []

            # Merge
            final_id = exploit_id
            final_id.extend(explore_id)
            final_id = list(set(final_id))

            mask = torch.zeros((self.n_action))
            mask[final_id] = 1

            idx.append(mask)

        return torch.stack(idx, dim= 0) # (batch_size, n_action)

class DQN(object):
    def __init__(self, n_states, n_actions,
                 memory_capacity, lr, epsilon, target_network_replace_freq, batch_size, gamma, tau, K, 
                 embd= None, mode= 'vanilla', args= None,
                 item_pop_dict= None):
        self.n_states = n_states # States size, i.e., observation windows
        self.n_actions = n_actions # Size of action space
        self.lr = lr
        self.epsilon = epsilon
        self.replace_freq = target_network_replace_freq
        self.replace_freq_unit = int(target_network_replace_freq * 0.25)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.train_loss = []
        self.args = args

        self.one_hot = False
        self.num_raw_inputs = -1
        if self.args.sim_mode == 'crossrec':
            self.one_hot = True
            self.num_raw_inputs = self.args.n_items

        if self.args.cuda < 0:
            self.device = 'cpu'
        else:
            self.device = f'cuda:{self.args.cuda}'

        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.state_mean = 0.0
        self.state_std = 1.0
        self.std_smoothing = 1e-5

        self.rare_thresh = self.args.rare_thresh

        self.item_pop_dict = item_pop_dict

        self.num_hidden = self.args.num_hidden

        if embd is None:
            self.eval_net = Net(self.n_states, self.n_actions, self.num_hidden, 
                                dueling= self.args.dueling_dqn, noisy_net= self.args.noisy_net, dropout= self.args.dropout,
                                one_hot= self.one_hot, num_raw_inputs= self.num_raw_inputs)
            self.buffered_net = Net(self.n_states, self.n_actions, self.num_hidden, 
                                    dueling= self.args.dueling_dqn, noisy_net= self.args.noisy_net, dropout= self.args.dropout,
                                    one_hot= self.one_hot, num_raw_inputs= self.num_raw_inputs)
            self.target_net = Net(self.n_states, self.n_actions, self.num_hidden, 
                                  dueling= self.args.dueling_dqn, noisy_net= self.args.noisy_net, dropout= self.args.dropout,
                                  one_hot= self.one_hot, num_raw_inputs= self.num_raw_inputs)
        else:
            self.eval_net = Net(self.n_states, embd.weight.shape[-1], self.num_hidden, 
                                embd= embd.to(self.device), 
                                dueling= self.args.dueling_dqn, noisy_net= self.args.noisy_net, dropout= self.args.dropout,
                                one_hot= self.one_hot, num_raw_inputs= self.num_raw_inputs)
            self.buffered_net = Net(self.n_states, embd.weight.shape[-1], self.num_hidden, 
                                embd= embd.to(self.device), 
                                dueling= self.args.dueling_dqn, noisy_net= self.args.noisy_net, dropout= self.args.dropout,
                                one_hot= self.one_hot, num_raw_inputs= self.num_raw_inputs)
            self.target_net = Net(self.n_states, embd.weight.shape[-1], self.num_hidden, 
                                embd= embd.to(self.device), 
                                dueling= self.args.dueling_dqn, noisy_net= self.args.noisy_net, dropout= self.args.dropout,
                                one_hot= self.one_hot, num_raw_inputs= self.num_raw_inputs)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.max_iter = int(self.args.epoch_max * self.args.episode_max * self.args.step_max / self.args.episode_batch)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 
        #                                                                       T_0 = int(self.args.step_max), verbose= True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = self.max_iter)
        self.loss_func = nn.MSELoss()
        # self.loss_func = nn.HuberLoss()
        hard_update(self.target_net, self.eval_net) # Transfer weights from eval_q_net to target_q_net

        self.learn_step_counter = 0
        self.K = K # topK

        # 0-th dimension for memory parition: 0-sequential, 1-rare, 2-random
        self.memory = [np.zeros((0, self.n_states * 2 + 2))] * 3
        self.mask = [[] for _ in range(3)]

        self.memory_counter = np.array([0, 0, 0])
        self.memory_capacity = memory_capacity
        self.partition = np.array([self.args.seq_ratio, self.args.rare_ratio, self.args.rand_ratio])
        self.pbatch_size = (self.batch_size * self.partition / np.sum(self.partition)).astype('int')
        self.partition = (memory_capacity * self.partition / np.sum(self.partition)).astype('int')

        self.mode = mode

        # Proposal net
        if embd is not None:
            self.proposal_net = AQLProposalNet(embd, self.args)

        # Load components to device
        self.eval_net = self.eval_net.to(self.device)
        self.buffered_net = self.buffered_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        self.loss_func = self.loss_func.to(self.device)

        if embd is not None:
            self.proposal_net = self.proposal_net.to(self.device)

    def choose_action(self, obs, env, mode= 'training'):
        self.setEval()
        if env.args.sim_mode == 'stats':
            obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)
        state = env.build_state(obs)

        # Absolute cold-start user is undefined
        if state is None:
            return []
        
        # state = (state - self.state_mean) / (self.state_std + self.std_smoothing)
        state = state.to(torch.float).to(self.device)
        actions_Q = self.eval_net.forward(state)

        # Action proposal
        if self.args.action_proposal:
            masks = self.proposal_net(state, mode= 'deterministic').to(self.device)
        else:
            masks = 1.

        actions_Q = (actions_Q * masks).cpu().detach().numpy().squeeze() # (n_actions)
        sorted_ids = sorted(range(actions_Q.size), key= lambda x:-actions_Q[x])

        rec_list = []
        cnt = 0

        for i in range(actions_Q.size):
            candidate = sorted_ids[i]
            if candidate in env.mask_list:
                continue
            rec_list.append(candidate)
            cnt += 1
            if cnt == self.args.topk:
                break

        return rec_list

    def check_memory(self):
        # print(len(self.memory[0]), len(self.memory[1]), len(self.memory[2]))
        # print(len(self.memory[0]) + len(self.memory[1]) + len(self.memory[2]), np.sum(self.partition))
        return ((len(self.memory[0]) + len(self.memory[1]) + len(self.memory[2])) >= np.sum(self.partition))

    def align_memory(self):
        # RewardNorm
        current_rewards = [self.memory[x][:, self.n_states + 1:self.n_states + 2] for x in range(3)]
        current_rewards = np.concatenate(current_rewards, axis= 0)
        self.reward_mean = np.mean(current_rewards)
        self.reward_std = np.std(current_rewards)
        print('####################')
        print('Normalize initial rewards ...')
        print('mean:', self.reward_mean, 'std:', self.reward_std)
        for i in range(3):
            self.memory[i][:, self.n_states + 1:self.n_states + 2] -= self.reward_mean
            self.memory[i][:, self.n_states + 1:self.n_states + 2] /= (self.reward_std + self.std_smoothing)
        
        print('Verify normalization ...')
        current_rewards = [self.memory[x][:, self.n_states + 1:self.n_states + 2] for x in range(3)]
        current_rewards = np.concatenate(current_rewards, axis= 0)
        print('mean:', np.mean(current_rewards), 'std:', np.std(current_rewards))

        # StateNorm
        current_states = [self.memory[x][:, :self.n_states] for x in range(3)]
        current_states_ = [self.memory[x][:, self.n_states + 2:] for x in range(3)]
        current_states = np.concatenate(current_states, axis= 0)
        self.state_mean = np.mean(current_states, axis= 0)
        self.state_std = np.std(current_states, axis= 0)
        print('####################')
        print('Normalize initial states ...')
        print('mean_norm:', np.linalg.norm(self.state_mean), 'std_norm:', np.linalg.norm(self.state_std))
        for i in range(3):
            self.memory[i][:, :self.n_states] -= self.state_mean
            self.memory[i][:, :self.n_states] /= (self.state_std + self.std_smoothing)

            self.memory[i][:, self.n_states + 2:] -= self.state_mean
            self.memory[i][:, self.n_states + 2:] /= (self.state_std + self.std_smoothing)

        print('Verify normalization ...')
        current_states = [self.memory[x][:, :self.n_states] for x in range(3)]
        current_states_ = [self.memory[x][:, self.n_states + 2:] for x in range(3)]
        current_states = np.concatenate(current_states, axis= 0)
        new_mean = np.mean(current_states, axis= 0)
        new_std = np.std(current_states, axis= 0)
        print('mean_norm:', np.linalg.norm(new_mean), 'std_norm:', np.linalg.norm(new_std))
        print('####################')

    def store_transition(self, s, a, r, s_, mask):
        # normalized_r = (r - self.reward_mean) / (self.reward_std + self.std_smoothing)
        # normalized_s = (s - self.state_mean) / (self.state_std + self.std_smoothing)
        # normalized_s_ = (s_ - self.state_mean) / (self.state_std + self.std_smoothing)
        transition = np.hstack((s, np.array([a, float(r)]), s_))

        mask = copy.copy(mask)

        # Always store transition into random memory
        # Shuffle memory. Preventing forgetting of interactions from early episodes
        random_index = np.arange(len(self.memory[2]))
        np.random.shuffle(random_index)
        self.memory[2] = self.memory[2][random_index, :]
        if len(self.memory[2]) > 0:
            self.mask[2] = [self.mask[2][x] for x in random_index.tolist()]

        if len(self.memory[2]) < self.partition[2]:
            self.memory[2] = np.append(self.memory[2], [transition], axis=0)
            self.mask[2].append(mask)
        else:
            index = self.memory_counter[2] % self.partition[2]
            self.memory[2][index, :] = transition
            self.mask[2][index] = mask
        self.memory_counter[2] += 1

        # Always store transition into sequential memory
        # Round-Robin
        if len(self.memory[0]) < self.partition[0]:
            self.memory[0] = np.append(self.memory[0], [transition], axis=0)
            self.mask[0].append(mask)
        else:
            index = self.memory_counter[0] % self.partition[0]
            self.memory[0][index, :] = transition
            self.mask[0][index] = mask
        self.memory_counter[0] += 1

        # Rare-action memory
        if self.item_pop_dict[str(a)] <= self.rare_thresh or self.item_pop_dict[str(a)] >= 1. - self.rare_thresh:
        # if self.item_pop_dict[str(a)] >= 1. - self.rare_thresh:
            if len(self.memory[1]) < self.partition[1]:
                self.memory[1] = np.append(self.memory[1], [transition], axis=0)
                self.mask[1].append(mask)
            else:
                actions = np.copy(self.memory[1][:, self.n_states]).reshape((-1)).tolist() # List of rare actions stored
                
                # Replace the most frequent item in the rare-item memory
                # action_ranks = np.array([self.item_pop_dict[str(int(x))] for x in actions])
                # index = np.argmax(action_ranks) 

                # Round-Robin
                index = self.memory_counter[1] % self.partition[1]
                
                self.memory[1][index, :] = transition
                self.mask[1][index] = mask
            self.memory_counter[1] += 1

    def sampling(self):
        samples = []
        splits = []
        _batch_masks = []
        rng = np.random.default_rng()
        for mode in range(3):
            # sample_index = random.sample(list(range(len(self.memory[mode]))), 
            #                                 k= min(self.pbatch_size[mode], len(self.memory[mode])))
            sample_index = rng.choice(np.arange(len(self.memory[mode])), 
                                      size= min(self.pbatch_size[mode], len(self.memory[mode])), 
                                      replace= False)
            # (batch_size, transition_shape)
            batch_memory = self.memory[mode][sample_index, :].reshape((self.pbatch_size[mode], -1))
            batch_mask = [self.mask[mode][x] for x in sample_index.tolist()]
            samples.append(batch_memory)
            _batch_masks.append(batch_mask)
            splits.append(min(self.pbatch_size[mode], len(self.memory[mode])))
        
        batch_memory = np.concatenate(samples, axis= 0)

        batch_masks = torch.ones((batch_memory.shape[0], self.n_actions))
        cnt = 0
        for _masks in _batch_masks:
            for _mask in _masks:
                batch_masks[cnt, _mask] = 0.
                cnt += 1
        
        batch_state = torch.tensor(batch_memory[:, :self.n_states], dtype=torch.float32)
        batch_action = torch.tensor(batch_memory[:, self.n_states:self.n_states + 1].astype(int), dtype=torch.long)
        batch_reward = torch.tensor(batch_memory[:, self.n_states + 1:self.n_states + 2], dtype=torch.float32)
        batch_state_ = torch.tensor(batch_memory[:, -self.n_states:], dtype=torch.float32)

        return batch_masks, batch_state, batch_action, batch_reward, batch_state_, torch.tensor(splits)

    def CQLLoss(self, max_obs, q_values, current_action, buffered_action = None):
        '''
        Consult: https://github.com/BY571/CQL/blob/main/CQL-DQN/agent.py#L45
        Notations follow https://arxiv.org/pdf/2006.04779 

        q_values.shape = batch_size, n_actions # Q_values under current policy
        buffered_action.shape = batch_size, 1
        current_action.shape = batch_size, 1
        '''
        if self.args.cql_mode == 'none':
            return 0
        elif self.args.cql_mode == 'cql_H':
            # Minimize Q-values
            logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True) # batch_size, 1
            # Maximize Q-values under data
            q_a = q_values.gather(1, current_action)
            return (logsumexp - q_a).mean()
        elif self.args.cql_mode == 'cql_Rho':
            try:
                assert buffered_action is not None
            except:
                raise ValueError('Q_values from previous policy must not be None for CQL(rho)!')
            # Maximize Q-values under data
            maximizer = q_values.gather(1, current_action)
            # Minimize Q-values
            minimizer = q_values.gather(1, buffered_action) # batch_size, 1
            return (minimizer - maximizer / max_obs).mean()
        else:
            raise NotImplementedError(f'CQL mode {self.args.cql_mode} not defined!')
        
    def setTrain(self):
        self.eval_net.train()
        self.buffered_net.train()
        self.target_net.eval()

    def setEval(self):
        self.eval_net.eval()
        self.buffered_net.eval()
        self.target_net.eval()

    def net_hard_update(self):
        self.setEval()
        # hard_update(self.target_net, self.eval_net)
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self):
        # Major update: Replace target net by evaluation net
        if self.replace_freq > 0 and self.learn_step_counter % self.replace_freq == 0:
            self.setEval()
            # hard_update(self.target_net, self.eval_net)
            self.target_net.load_state_dict(self.eval_net.state_dict())
            
        self.learn_step_counter += 1

        # Shape of batch_mask: (batch_size, n_actions)
        batch_mask, batch_state, batch_action, batch_reward, batch_state_, splits = self.sampling()

        batch_state = batch_state.to(self.device)
        batch_action = batch_action.to(self.device)
        batch_reward = batch_reward.to(self.device)
        batch_state_ = batch_state_.to(self.device)
        batch_mask = batch_mask.to(self.device)
        splits = splits.to(self.device)

        self.setTrain()

        raw_q_eval = self.eval_net(batch_state) # batch_size, n_actions

        raw_q_eval_next = None
        if self.mode == 'ddqn':
            raw_q_eval_next = self.eval_net(batch_state_) # batch_size, n_action

        buffered_q_eval = None
        if self.args.cql_mode == 'cql_Rho':
            buffered_q_eval = self.buffered_net(batch_state) # batch_size, n_actions

        # Action proposal
        if self.args.action_proposal:
            masks = self.proposal_net(batch_state_).to(self.device)
        else:
            masks = batch_mask # batch_size, n_actions

        if self.args.cql_mode == 'cql_Rho':
            masks = torch.ones_like(masks) # Disable masks for cql_Rho

        # q_next.shape == batch_size, n_actions
        q_next = self.target_net(batch_state_).detach() # detach target_net from gradient updates (?)

        q_eval = raw_q_eval.gather(1, batch_action) # batch_size, 1

        # Apply mask
        if self.mode == 'vanilla':
            q_next = q_next * masks
        elif self.mode == 'ddqn':
            raw_q_eval_next = raw_q_eval_next * masks

        # Double DQN
        if self.mode == 'vanilla':
            if self.args.policy == 'max':
                q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
            elif self.args.policy == 'stochastic':
                margin_q = torch.sum(q_next, 1, keepdim= True)
                raw_q_next = q_next * q_next / margin_q
                q_target = batch_reward + self.gamma * torch.sum(raw_q_next, 1, keepdims= True)
            else:
                raise NotImplementedError(f'DQN policy mode {self.args.policy} not found!')
            
        elif self.mode == 'ddqn':
            if self.args.policy == 'max':
                q_eval_action = raw_q_eval_next.argmax(dim=1).unsqueeze(1)
                # print(batch_reward, q_eval_action, q_next)
                q_target = batch_reward + self.gamma * q_next.gather(1, q_eval_action)
            elif self.args.policy == 'stochastic':
                margin_q = torch.sum(raw_q_eval_next, 1, keepdim= True).detach()
                raw_q_next = q_next * (raw_q_eval_next.clone().detach() / margin_q)
                q_target = batch_reward + self.gamma * torch.sum(raw_q_next, 1, keepdims= True)
            else:
                raise NotImplementedError(f'DQN policy mode {self.args.policy} not found!')
        else:
            raise NotImplementedError(f'DQN update mode {self.mode} not found!')

        # Partition-weighted Bellman loss
        weights = splits / torch.sum(splits)

        # print(q_eval, q_target)

        seq_loss = self.loss_func(q_eval[ : splits[0]], q_target[ : splits[0]].detach())
        rare_loss = self.loss_func(q_eval[splits[0] : splits[0] + splits[1]], q_target[splits[0] : splits[0] + splits[1]].detach())
        rand_loss = self.loss_func(q_eval[splits[0] + splits[1] : ], q_target[splits[0] + splits[1] : ].detach())

        bellman_loss = weights[0] * seq_loss + weights[1] * rare_loss + weights[2] * rand_loss

        # print('bellman:', seq_loss, rare_loss, rand_loss)

        # BETA: MOO-Based Bellman loss
        # mean_loss = (seq_loss + rare_loss + rand_loss) / 3.
        mean_loss = bellman_loss.detach()
        # moo_loss = self.loss_func(seq_loss, mean_loss) +  self.loss_func(rand_loss, mean_loss) + self.loss_func(rare_loss, mean_loss)
        # moo_loss = torch.sqrt(moo_loss / 3.)

        # Conservative Q learning
        buffered_action = None
        if self.args.cql_mode == 'cql_Rho':
            buffered_action = torch.argmax(buffered_q_eval, dim= 1).unsqueeze(dim= 1)
        cql_loss = self.CQLLoss((self.args.max_obs + 1.) / 2., raw_q_eval, batch_action, buffered_action)

        # print('cql:', cql_loss)

        loss = self.args.cql_alpha * cql_loss + 0.5 * bellman_loss
        self.train_loss.append(loss.item())

        # Minor update over eval_net
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.eval_net.parameters(), 1.)
        self.optimizer.step()
        self.scheduler.step()

        # NoisyNet: reset noise
        self.target_net.reset_noise()
        self.eval_net.reset_noise()
        
        # Fuse eval_net into target_net with temperature tau
        soft_update(self.target_net, self.eval_net, self.tau)

        # Update buffer
        if self.args.cql_mode == 'cql_Rho':
            hard_update(self.buffered_net, self.eval_net)

    def stats_plot(self, args, resdict):
        if not os.path.exists('exps'):
            os.mkdir('exps')
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
        root = os.path.join('exps', dt_string)
        os.makedirs(root)

        precision, recall, ndcg = resdict['precision'], resdict['recall'], resdict['ndcg']
        epc, coverage = resdict['epc'], resdict['coverage']
        freq = resdict['freq']
        reclist, testlist, testers = resdict['reclist'], resdict['testlist'], resdict['testers']

        save_pth = os.path.join(root, f'{self.args.dataset}-{self.args.step_max}.step-{self.args.gamma}.gamma.png')
        reduced_loss = [min(100, np.mean(self.train_loss[(x-1) * 4 : x * 4])) for x in range(10, int(len(self.train_loss) / 4))]
        plt.plot(reduced_loss)
        plt.title('Training Loss')
        plt.savefig(save_pth)
        plt.clf()

        # Export args
        with open(os.path.join(root, 'args.txt'), 'w') as f:
            # Release crossrec-bundle
            args.crossrec_bundle = "removed"
            json.dump(args.__dict__, f, indent=2)
            f.close()

        # Export loss
        train_loss = np.array(self.train_loss)
        np.savetxt(os.path.join(root, 'loss.txt'), train_loss)

        # Export item frequency
        freq = np.array(freq)
        np.savetxt(os.path.join(root, 'freq.txt'), freq)

        # Export recommendation list and test list
        n_users = len(testers)
        if len(reclist) != n_users or len(testlist) != n_users:
            print('WARNING: Different number of testers and recommendation/test list!')

        with open(os.path.join(root, 'reclist.txt'), 'w') as file:
            Lines = []
            for i in range(n_users):
                line = [testers[i]]
                line.extend(reclist[i])
                line = ' '.join([str(x) for x in line]) + '\n'
                Lines.append(line)
            file.writelines(Lines)
            file.close()

        with open(os.path.join(root, 'testlist.txt'), 'w') as file:
            Lines = []
            for i in range(n_users):
                line = [testers[i]]
                line.extend(testlist[i])
                line = ' '.join([str(x) for x in line]) + '\n'
                Lines.append(line)
            file.writelines(Lines)
            file.close()

        # Plots
        if precision is not None:
            save_pth = os.path.join(root, f'precision.png')
            plt.plot(precision)
            plt.title(f'Test Precision@{args.topk}')
            plt.savefig(save_pth)
            plt.clf()

        if recall is not None:
            save_pth = os.path.join(root, f'recall.png')
            plt.plot(recall)
            plt.title(f'Test Recall@{args.topk}')
            plt.savefig(save_pth)
            plt.clf()

        if ndcg is not None:
            save_pth = os.path.join(root, f'ndcg.png')
            plt.plot(ndcg)
            plt.title(f'Test NDCG@{args.topk}')
            plt.savefig(save_pth)    
            plt.clf()    

        if epc is not None:
            save_pth = os.path.join(root, f'epc.png')
            plt.plot(epc)
            plt.title(f'Test EPC@{args.topk}')
            plt.savefig(save_pth)    
            plt.clf()   

        if coverage is not None:
            save_pth = os.path.join(root, f'coverage.png')
            plt.plot(coverage)
            plt.title(f'Test Coverage@{args.topk}')
            plt.savefig(save_pth)    
            plt.clf()   

        return root