import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y

nn.LayerNorm = LayerNorm

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, embd= None):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

        self.softmax = nn.Softmax(dim= 1)

        if embd is not None:
            self.embd = embd.weight.detach()
            try:
                assert self.embd.shape[-1] == num_outputs
            except:
                print('Output size must equal embedding size for dot product')
        else:
            self.embd = None

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        q_values = torch.tanh(self.mu(x))
        if self.embd is not None:
            q_values = (q_values @ self.embd.T)
        return q_values


class DQN(object):
    def __init__(self, n_states, n_actions,
                 memory_capacity, lr, epsilon, target_network_replace_freq, batch_size, gamma, tau, K, embd= None):
        self.n_states = n_states # States size, i.e., observation windows
        self.n_actions = n_actions # Size of action space
        self.memory_capacity = memory_capacity
        self.lr = lr
        self.epsilon = epsilon
        self.replace_freq = target_network_replace_freq
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        if embd is None:
            self.eval_net = Net(self.n_states, self.n_actions, 256)
            self.target_net = Net(self.n_states, self.n_actions, 256)
        else:
            self.eval_net = Net(self.n_states, embd.weight.shape[-1], 256, embd= embd)
            self.target_net = Net(self.n_states, embd.weight.shape[-1], 256, embd= embd)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        hard_update(self.target_net, self.eval_net) # Transfer weights from eval_q_net to target_q_net
        if (torch.cuda.is_available()):
            self.eval_net = self.eval_net.cuda()
            self.target_net = self.target_net.cuda()
            self.loss_func = self.loss_func.cuda()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.K = K # topK
        self.memory = np.zeros((0, self.n_states * 2 + 2))

    def choose_action(self, obs, env, I_sim_list):
        if env.args.sim_mode == 'stats':
            obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)
        state = env.build_state(obs)
        if (torch.cuda.is_available()):
            state = state.cuda()
        actions_Q = self.eval_net.forward(torch.tensor(state))

        actions_Q = actions_Q.cpu().detach().numpy()

        temp_actions_Qvalue = []
        for index in range(env.n_actions):
            temp_actions_Qvalue.append(actions_Q[0][index])

        actions_Qvalue = torch.from_numpy(np.array(temp_actions_Qvalue))
        actions_Qvalue = torch.unsqueeze(actions_Qvalue, 0)

        actions_Qvalue_list = actions_Qvalue.tolist()[0]
        sorted_Qvalue = sorted(actions_Qvalue_list, reverse= True)

        rec_list = []
        cnt = 0

        while len(rec_list) < self.K:
            # Exploitation
            if np.random.uniform() < self.epsilon:
                if cnt < len(actions_Qvalue_list):
                    action = actions_Qvalue_list.index(sorted_Qvalue[cnt])
                    cnt += 1
                else:
                    action = np.random.randint(0, self.n_actions)
            # Exploration - epsilon greedy
            else:
                action = np.random.randint(0, self.n_actions)
            # If action is not masked (by being chosen before, etc...)
            if action not in env.mask_list and action not in rec_list:
                rec_list.append(action)
                env.mask_list.append(action)
        return rec_list

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        if len(self.memory) < self.memory_capacity:
            self.memory = np.append(self.memory, [transition], axis=0)
        else:
            index = self.memory_counter % self.memory_capacity
            self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # Major update: Replace target net by evaluation net
        if self.learn_step_counter % self.replace_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(len(self.memory), self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = Variable(torch.tensor(batch_memory[:, :self.n_states], dtype=torch.float32))
        batch_action = Variable(
            torch.tensor(batch_memory[:, self.n_states:self.n_states + 1].astype(int), dtype=torch.long))
        batch_reward = Variable(torch.tensor(batch_memory[:, self.n_states + 1:self.n_states + 2], dtype=torch.float32))
        batch_state_ = Variable(torch.tensor(batch_memory[:, -self.n_states:], dtype=torch.float32))

        if (torch.cuda.is_available()):
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_state_ = batch_state_.cuda()

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_state_).detach() # detach target_net from gradient updates (?)
        q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        # Minor update over eval_net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Fuse eval_net into target_net with temperature tau
        soft_update(self.target_net, self.eval_net, self.tau)
