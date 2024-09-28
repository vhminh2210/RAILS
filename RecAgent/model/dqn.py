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
random.seed(101)
# import seaborn as sns

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, embd= None, dueling= False):
        super(Net, self).__init__()
        # self.linear0 = nn.Linear(num_inputs, hidden_size)
        # self.ln0 = nn.LayerNorm(hidden_size)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.dueling = dueling

        if not self.dueling:
            # Non-dueling DQN
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.ln2 = nn.LayerNorm(hidden_size)

            self.mu = nn.Linear(hidden_size, num_outputs)

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
            self.ln2_V = nn.LayerNorm(half_size)

            self.linear3_V = nn.Linear(half_size, 1)

            # Advantage branch
            self.linear2_A = nn.Linear(hidden_size, half_size)
            self.ln2_A = nn.LayerNorm(half_size)

            self.mu = nn.Linear(half_size, num_outputs)

        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

        # self.softmax = nn.Softmax(dim= 1)

        if embd is not None:
            self.embd = embd.weight.detach()
            try:
                assert self.embd.shape[-1] == num_outputs
            except:
                print('Output size must equal embedding size for dot product')
        else:
            self.embd = None

        # Consult https://wandb.ai/wandb_fc/tips/reports/How-to-Initialize-Weights-in-PyTorch--VmlldzoxNjcwOTg1
        self.apply(self._init_weights) 

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, inputs):
        if self.dueling:
            # Dueling DQN
            x = inputs
            # x = F.relu(self.ln0(self.linear0(x)))
            x = F.relu(self.ln1(self.linear1(x)))

            # Value branch
            val = F.relu(self.ln2_V(self.linear2_V(x)))
            val = self.linear3_V(val) # (batch_size, 1)

            # Advantage branch
            adv = F.relu(self.ln2_A(self.linear2_A(x)))
            if self.embd is not None:
                adv = (self.mu(adv) @ self.embd.T) # (batch_size, n_action)
            else:
                adv = torch.tanh(self.mu(adv)) # (batch_size, num_outputs = n_action)

            # Q-value fusion
            adv_mean = torch.mean(adv, dim= 1, keepdim= True) # (batch_size, 1)
            q_values = adv + (val - adv_mean)
        
        else:
            x = inputs
            # x = F.relu(self.ln0(self.linear0(x)))
            x = F.relu(self.ln1(self.linear1(x)))
            x = F.relu(self.ln2(self.linear2(x)))

            if self.embd is not None:
                q_values = (self.mu(x) @ self.embd.T)
            else:
                q_values = torch.tanh(self.mu(x))

        return q_values


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
            self.eval_net = Net(self.n_states, self.n_actions, self.num_hidden, dueling= self.args.dueling_dqn)
            self.buffered_net = Net(self.n_states, self.n_actions, self.num_hidden, dueling= self.args.dueling_dqn)
            self.target_net = Net(self.n_states, self.n_actions, self.num_hidden, dueling= self.args.dueling_dqn)
        else:
            self.eval_net = Net(self.n_states, embd.weight.shape[-1], self.num_hidden, 
                                embd= embd.to(self.device), dueling= self.args.dueling_dqn)
            self.buffered_net = Net(self.n_states, embd.weight.shape[-1], self.num_hidden, 
                                embd= embd.to(self.device), dueling= self.args.dueling_dqn)
            self.target_net = Net(self.n_states, embd.weight.shape[-1], self.num_hidden, 
                                embd= embd.to(self.device), dueling= self.args.dueling_dqn)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        max_iter = int(self.args.epoch_max * self.args.episode_max * self.args.step_max / self.args.episode_batch)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 
        #                                                                       T_0 = int(self.args.step_max), verbose= True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = max_iter)
        self.loss_func = nn.MSELoss()
        # self.loss_func = nn.HuberLoss()
        hard_update(self.target_net, self.eval_net) # Transfer weights from eval_q_net to target_q_net

        self.learn_step_counter = 0
        self.K = K # topK

        # 0-th dimension for memory parition: 0-sequential, 1-rare, 2-random
        self.memory = [np.zeros((0, self.n_states * 2 + 2))] * 3
        self.memory_counter = np.array([0, 0, 0])
        self.memory_capacity = memory_capacity
        self.partition = np.array([self.args.seq_ratio, self.args.rare_ratio, self.args.rand_ratio])
        self.pbatch_size = (self.batch_size * self.partition / np.sum(self.partition)).astype('int')
        self.partition = (memory_capacity * self.partition / np.sum(self.partition)).astype('int')

        self.mode = mode

        # Load components to device
        self.eval_net = self.eval_net.to(self.device)
        self.buffered_net = self.buffered_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        self.loss_func = self.loss_func.to(self.device)

    def choose_action(self, obs, env, mode= 'training'):
        self.setEval()
        if env.args.sim_mode == 'stats':
            obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)
        state = env.build_state(obs)
        state = (state - self.state_mean) / (self.state_std + self.std_smoothing)
        state = state.to(torch.float).to(self.device)
        actions_Q = self.eval_net.forward(state)

        actions_Q = actions_Q.cpu().detach().numpy().squeeze() # (n_actions)
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

    def store_transition(self, s, a, r, s_):
        normalized_r = (r - self.reward_mean) / (self.reward_std + self.std_smoothing)
        normalized_s = (s - self.state_mean) / (self.state_std + self.std_smoothing)
        normalized_s_ = (s_ - self.state_mean) / (self.state_std + self.std_smoothing)
        transition = np.hstack((normalized_s, np.array([a, float(normalized_r)]), normalized_s_))

        # Always store transition into random memory
        # Shuffle memory. Preventing forgetting of interactions from early episodes
        random_index = np.arange(len(self.memory[2]))
        np.random.shuffle(random_index)
        self.memory[2] = self.memory[2][random_index, :]

        if len(self.memory[2]) < self.partition[2]:
            self.memory[2] = np.append(self.memory[2], [transition], axis=0)
        else:
            index = self.memory_counter[2] % self.partition[2]
            self.memory[2][index, :] = transition
        self.memory_counter[2] += 1

        # Always store transition into sequential memory
        # Round-Robin
        if len(self.memory[0]) < self.partition[0]:
            self.memory[0] = np.append(self.memory[0], [transition], axis=0)
        else:
            index = self.memory_counter[0] % self.partition[0]
            self.memory[0][index, :] = transition
        self.memory_counter[0] += 1

        # Rare-action memory
        if self.item_pop_dict[str(a)] <= self.rare_thresh:
            if len(self.memory[1]) < self.partition[1]:
                self.memory[1] = np.append(self.memory[1], [transition], axis=0)
            else:
                # Replace the most frequent item in the rare-item memory
                actions = np.copy(self.memory[1][:, self.n_states]).reshape((-1)).tolist() # List of rare actions stored
                action_ranks = np.array([self.item_pop_dict[str(int(x))] for x in actions])
                index = np.argmax(action_ranks)
                self.memory[1][index, :] = transition
            self.memory_counter[1] += 1

    def sampling(self):
        samples = []
        splits = []
        for mode in range(3):
            sample_index = random.sample(list(range(len(self.memory[mode]))), 
                                            k= min(self.pbatch_size[mode], len(self.memory[mode])))
            # (batch_size, transition_shape)
            batch_memory = self.memory[mode][sample_index, :].reshape((self.pbatch_size[mode], -1))
            samples.append(batch_memory)
            splits.append(min(self.pbatch_size[mode], len(self.memory[mode])))
        
        batch_memory = np.concatenate(samples, axis= 0)
        
        batch_state = torch.tensor(batch_memory[:, :self.n_states], dtype=torch.float32)
        batch_action = torch.tensor(batch_memory[:, self.n_states:self.n_states + 1].astype(int), dtype=torch.long)
        batch_reward = torch.tensor(batch_memory[:, self.n_states + 1:self.n_states + 2], dtype=torch.float32)
        batch_state_ = torch.tensor(batch_memory[:, -self.n_states:], dtype=torch.float32)

        return batch_state, batch_action, batch_reward, batch_state_, torch.tensor(splits)

    def CQLLoss(self, q_values, current_action, buffered_action = None):
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
            return (minimizer - maximizer).mean()
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

    def learn(self):
        # Major update: Replace target net by evaluation net
        if self.learn_step_counter % self.replace_freq == 0:
            self.setEval()
            self.target_net.load_state_dict(self.eval_net.state_dict())
            
        self.learn_step_counter += 1

        batch_state, batch_action, batch_reward, batch_state_, splits = self.sampling()

        batch_state = batch_state.to(self.device)
        batch_action = batch_action.to(self.device)
        batch_reward = batch_reward.to(self.device)
        batch_state_ = batch_state_.to(self.device)
        splits = splits.to(self.device)

        self.setTrain()

        raw_q_eval = self.eval_net(batch_state) # batch_size, n_actions
        buffered_q_eval = None
        if self.args.cql_mode == 'cql_Rho':
            buffered_q_eval = self.buffered_net(batch_state) # batch_size, n_actions
        q_eval = raw_q_eval.gather(1, batch_action) # batch_size, 1
        # q_next.shape == batch_size, n_actions
        q_next = self.target_net(batch_state_).detach() # detach target_net from gradient updates (?)

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
                q_eval_action = raw_q_eval.argmax(dim=1).unsqueeze(1)
                q_target = batch_reward + self.gamma * q_next.gather(1, q_eval_action)
            elif self.args.policy == 'stochastic':
                margin_q = torch.sum(raw_q_eval, 1, keepdim= True).detach()
                raw_q_next = q_next * (raw_q_eval.clone().detach() / margin_q)
                q_target = batch_reward + self.gamma * torch.sum(raw_q_next, 1, keepdims= True)
            else:
                raise NotImplementedError(f'DQN policy mode {self.args.policy} not found!')
        else:
            raise NotImplementedError(f'DQN update mode {self.mode} not found!')

        # Partition-weighted Bellman loss
        weights = splits / torch.sum(splits)

        seq_loss = self.loss_func(q_eval[ : splits[0]], q_target[ : splits[0]].detach())
        rare_loss = self.loss_func(q_eval[splits[0] : splits[0] + splits[1]], q_target[splits[0] : splits[0] + splits[1]].detach())
        rand_loss = self.loss_func(q_eval[splits[0] + splits[1] : ], q_target[splits[0] + splits[1] : ].detach())

        bellman_loss = weights[0] * seq_loss + weights[1] * rare_loss + weights[2] * rand_loss

        # Conservative Q learning
        buffered_action = None
        if self.args.cql_mode == 'cql_Rho':
            buffered_action = torch.argmax(buffered_q_eval, dim= 1).unsqueeze(dim= 1)
        cql_loss = self.CQLLoss(raw_q_eval, batch_action, buffered_action)

        loss = self.args.cql_alpha * cql_loss + 0.5 * bellman_loss
        self.train_loss.append(loss.item())

        # Minor update over eval_net
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.eval_net.parameters(), 1.)
        self.optimizer.step()
        # self.scheduler.step()
        
        # Fuse eval_net into target_net with temperature tau
        soft_update(self.target_net, self.eval_net, self.tau)

        # Update buffered 
        if self.args.cql_mode == 'cql_Rho':
            hard_update(self.buffered_net, self.eval_net)

    def stats_plot(self, args, precision= None, recall= None, ndcg= None, epc= None, coverage= None):
        if not os.path.exists('exps'):
            os.mkdir('exps')
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
        root = os.path.join('exps', dt_string)
        os.makedirs(root)

        with open(os.path.join(root, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        with open(os.path.join(root, 'loss.txt'), 'w') as f:
            f.write(str(self.train_loss))

        save_pth = os.path.join(root, f'{self.args.episode_max}.episodes-{self.args.step_max}.step-{self.args.gamma}.gamma.png')
        reduced_loss = [np.mean(self.train_loss[x * 256 : (x+1) * 256]) for x in range(4, int(len(self.train_loss) / 256))]
        plt.plot(reduced_loss)
        plt.title('Training Loss')
        plt.savefig(save_pth)
        plt.clf()

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