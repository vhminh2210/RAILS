from cmath import cos
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F


class MF(nn.Module):
    def __init__(self, args, data):
        super(MF, self).__init__()
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.lr = args.enc_lr
        self.emb_dim = args.embed_size
        self.batch_size = args.enc_batch_size
        self.decay = args.regs
        self.device = torch.device(args.cuda) if args.cuda >= 0 else torch.device('cpu')
        self.saveID = args.saveID

        self.train_user_list = data.train_user_list
        self.valid_user_list = data.valid_user_list
        # = torch.tensor(data.population_list).cuda(self.device)
        self.user_pop = torch.tensor(data.user_pop_idx).type(torch.LongTensor).to(self.device)
        self.item_pop = torch.tensor(data.item_pop_idx).type(torch.LongTensor).to(self.device)
        self.user_pop_max = data.user_pop_max
        self.item_pop_max = data.item_pop_max        

        self.embed_user = nn.Embedding(self.n_users, self.emb_dim)
        self.embed_item = nn.Embedding(self.n_items, self.emb_dim)

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)

    # Prediction function used when evaluation
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        users = self.embed_user(torch.tensor(users).to(self.device))
        items = torch.transpose(self.embed_item(torch.tensor(items).to(self.device)), 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()

class LGN(MF):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.Graph = data.getSparseGraph()
        self.n_layers = args.n_layers

    def compute(self):
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items

    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb.squeeze()), dim=1)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).to(self.device)]
        items = torch.transpose(all_items[torch.tensor(items).to(self.device)], 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()

class INFONCE(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1

    def forward(self, users, pos_items, neg_items):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb = F.normalize(users_emb, dim = -1)
        pos_emb = F.normalize(pos_emb, dim = -1)
        neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss

class INFONCE_batch(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1

    def forward(self, users, pos_items):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        numerator = torch.exp(ratings_diag / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))


        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2
        regularizer = regularizer
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss

class BC_LOSS(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau1 = args.tau1
        self.tau2 = args.tau2
        self.w_lambda = args.w_lambda
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.n_users_pop=data.n_user_pop
        self.n_items_pop=data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)
    
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):

        # popularity branch
        users_pop_emb = self.embed_user_pop(users_pop)
        pos_pop_emb = self.embed_item_pop(pos_items_pop)
        neg_pop_emb = self.embed_item_pop(neg_items_pop)

        pos_ratings_margin = torch.sum(users_pop_emb * pos_pop_emb, dim = -1)

        users_pop_emb = F.normalize(users_pop_emb, dim = -1)
        pos_pop_emb = F.normalize(pos_pop_emb, dim = -1)
        neg_pop_emb = F.normalize(neg_pop_emb, dim = -1)

        pos_ratings = torch.sum(users_pop_emb * pos_pop_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_pop_emb, 1), 
                                       neg_pop_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau2)
        denominator = torch.sum(torch.exp(ratings / self.tau2), dim = 1)
        loss2 = self.w_lambda * torch.mean(torch.negative(torch.log(numerator/denominator)))

        # main branch
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb = F.normalize(users_emb, dim = -1)
        pos_emb = F.normalize(pos_emb, dim = -1)
        neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        pos_ratings = torch.cos(torch.arccos(torch.clamp(pos_ratings,-1+1e-7,1-1e-7))+(1-torch.sigmoid(pos_ratings_margin)))
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau1)
        denominator = torch.sum(torch.exp(ratings / self.tau1), dim = 1)
        
        loss1 = (1-self.w_lambda) * torch.mean(torch.negative(torch.log(numerator/denominator)))

        # reg loss
        regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + \
                        0.5 * torch.norm(negEmb0) ** 2
        regularizer1 = regularizer1/self.batch_size

        regularizer2= 0.5 * torch.norm(users_pop_emb) ** 2 + 0.5 * torch.norm(pos_pop_emb) ** 2 + \
                        0.5 * torch.norm(neg_pop_emb) ** 2
        regularizer2  = regularizer2/self.batch_size
        reg_loss = self.decay * (regularizer1+regularizer2)

        reg_loss_freeze=self.decay * (regularizer2)
        reg_loss_norm=self.decay * (regularizer1)

        # main_loss, pop.bias(pb).extractor loss, (main+pb)_reg, pb_reg, main_reg
        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm

    def freeze_pop(self):
        self.embed_user_pop.requires_grad_(False)
        self.embed_item_pop.requires_grad_(False)

class BC_LOSS_batch(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau1 = args.tau1
        self.tau2 = args.tau2
        self.w_lambda = args.w_lambda
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.n_users_pop=data.n_user_pop
        self.n_items_pop=data.n_item_pop

        # Popularity bias extractor
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)
    
    def forward(self, users, pos_items, users_pop, pos_items_pop):
        # pos_items : Positive items
        # popularity branch
        users_pop_emb = self.embed_user_pop(users_pop)
        pos_pop_emb = self.embed_item_pop(pos_items_pop)

        pos_ratings_margin = torch.sum(users_pop_emb * pos_pop_emb, dim = -1)

        users_pop_emb = F.normalize(users_pop_emb, dim = -1)
        pos_pop_emb = F.normalize(pos_pop_emb, dim = -1)

        ratings = torch.matmul(users_pop_emb, torch.transpose(pos_pop_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        numerator = torch.exp(ratings_diag / self.tau2)
        denominator = torch.sum(torch.exp(ratings / self.tau2), dim = 1)
        loss2 = self.w_lambda * torch.mean(torch.negative(torch.log(numerator/denominator)))

        # main branch
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        ratings_diag = torch.cos(torch.arccos(torch.clamp(ratings_diag,-1+1e-7,1-1e-7))+\
                                (1-torch.sigmoid(pos_ratings_margin)))
        
        numerator = torch.exp(ratings_diag / self.tau1)
        denominator = torch.sum(torch.exp(ratings / self.tau1), dim = 1)
        loss1 = (1-self.w_lambda) * torch.mean(torch.negative(torch.log(numerator/denominator)))

        # reg loss
        regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + self.batch_size * 0.5 * torch.norm(posEmb0) ** 2
        regularizer1 = regularizer1/self.batch_size

        regularizer2= 0.5 * torch.norm(users_pop_emb) ** 2 + self.batch_size * 0.5 * torch.norm(pos_pop_emb) ** 2
        regularizer2  = regularizer2/self.batch_size
        reg_loss = self.decay * (regularizer1+regularizer2)

        reg_loss_freeze=self.decay * (regularizer2)
        reg_loss_norm=self.decay * (regularizer1)

        # main_loss, pop.bias(pb).extractor loss, (main+pb)_reg, pb_reg, main_reg
        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm

    def freeze_pop(self):
        '''
        Freeze the gradients of popularity bias extractor
        '''
        self.embed_user_pop.requires_grad_(False)
        self.embed_item_pop.requires_grad_(False)