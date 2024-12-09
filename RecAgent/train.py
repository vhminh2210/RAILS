import random
random.seed(101)

import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait
from .model import environment, dqn
from .util.metrics_util import ndcg_metric, novelty_metric, ils_metric, interdiv_metric

from tqdm import tqdm
import torch
import copy

import os

user_num = 0
precision, ndcg, novelty, coverage, ils, interdiv, recall, epc = [], [], [], [], [], [], [], []
logs = ['########### RL AGENT EVALUATIONS LOGS ###########']

def stateAugment(observations, history_size, n_augment_, freq, args):
    n_obs = len(observations)
    np_observations = np.array(observations)
    p = freq[observations]
    p = p / np.sum(p)

    inv_p = 1. - p
    inv_p = inv_p / np.sum(inv_p)
    # g = np.random.Generator(np.random.PCG64())
    try:
        assert n_obs >= history_size + 1
    except:
        raise ValueError('Sampling history size exceed known observations size!')
    n_augment = n_augment_

    if history_size == n_obs - 1:
        n_augment = min(n_augment, n_obs)

    aug_observations, aug_actions = [], []
    g = np.random.default_rng()

    for i in range(n_augment):
        # Pure random sampling
        if i % 2 == 0:
            idx = random.sample(observations, k= history_size + 1)
            history = idx[:-1]

            # Epsilon greedy
            if random.random() < args.epsilon:
                # Exploration
                while True:
                    action = random.randint(0, args.n_users - 1)
                    if action not in args.wild_items and action not in history:
                        break
            else:
                action = idx[-1] # Exploitation

        # Rare-item seeker
        elif i % 4 == 3:
            # idx = g.choice(n_obs, size= history_size + 1, p= p, replace= False)
            idx = g.choice(n_obs, size= history_size + 1, replace= False)
            samples = sorted(np_observations[idx].tolist(), key= lambda x: -freq[x])
            history = samples[:-1]

            # Epsilon greedy
            if random.random() < args.epsilon:
                # Exploration
                while True:
                    action = random.randint(0, args.n_users - 1)
                    if action not in args.wild_items and action not in history:
                        break
            else:
                action = samples[-1] # Exploitation

        # Popular item seeker
        else:
            # idx = g.choice(n_obs, size= history_size + 1, p= p, replace= False)
            idx = g.choice(n_obs, size= history_size + 1, replace= False)
            samples = sorted(np_observations[idx].tolist(), key= lambda x: freq[x])
            history = samples[:-1]

            # Epsilon greedy
            if random.random() < args.epsilon:
                # Exploration
                while True:
                    action = random.randint(0, args.n_users - 1)
                    if action not in args.wild_items and action not in history:
                        break
            else:
                action = samples[-1] # Exploitation

        aug_actions.append(action)
        aug_observations.append(history)

    return aug_observations, aug_actions

def setInteraction(env, agent, ep_user, train_df, args, freq, augment= True, ckpt= False, evalmode= False):
    user_df = train_df[train_df['user_id'] == ep_user]
    observations = user_df['item_id'].to_list()

    interaction_num = 0
    args = agent.args

    if evalmode or len(observations) < args.min_obs:
        return len(observations) - 1
    
    if args.n_aug_scale < 1:
        size_loader = range(1, len(observations))
    else:
        # tmp = list(range(1, len(observations) - 1))
        tmp = len(observations) - 1
        scale_list = np.arange(min(tmp, 3), tmp).tolist()
        # scale_list = np.arange(int(tmp // 1.5), tmp).tolist()
        # scale_list = np.arange(max(1, tmp - args.n_aug_scale), tmp).tolist()
        scale = min(len(scale_list) - 1, args.n_aug_scale)
        size_loader = random.sample(scale_list, k = max(scale, 0))
        size_loader.append(len(observations) - 1)

        if len(observations) <= 1:
            return 0
    
    for history_size in size_loader:
        if augment:
            aug_obsevations, aug_actions = stateAugment(observations, history_size, args.n_augment, freq, args)
        else:
            if not ckpt:
                aug_obsevations, aug_actions = stateAugment(observations, history_size, int(args.n_augment), freq, args)
            else:
                aug_obsevations, aug_actions = stateAugment(observations, history_size, 1, freq, args)

        for i in range(len(aug_actions)):
            s = np.array(env.reset(aug_obsevations[i]))
            a = int(aug_actions[i])
            s_, r, done = env.step(a)
            # print(aug_obsevations[i], aug_actions[i])
            agent.store_transition(env.build_state(s).reshape((-1)), a, r, s_.reshape((-1)), aug_obsevations[i])
        
        interaction_num += 1
    
    return interaction_num


def recommend_offpolicy(env, agent, last_obs):
    state = np.array(last_obs)
    so = env.reset(state)

    return agent.choose_action(so, env, mode= 'infer')

def recommend_encoder(user_emb, item_weight, last_obs, args):
    '''
    Note: user_emb represent a SINGLE user embedding, NOT the complete user embedding matrix
    '''
    user_emb = user_emb.to(args.device).squeeze()
    # item_weight = item_emb.weight.to(args.device)
    scores = (user_emb @ item_weight.T).squeeze()
    # scores = torch.einsum('d,id->i', user_emb, item_weight)
    sorted_ids = sorted(range(scores.shape[0]), key= lambda x:-scores[x])

    rec_list = []
    cnt = 0

    for idx in sorted_ids:
        if idx in last_obs:
            continue
        rec_list.append(idx)
        cnt += 1
        if cnt == args.topk:
            break
    
    return rec_list

def trainAgent(agent, step_max):
    # for step in tqdm(range(step_max)):
    #     agent.learn()
    for step in range(step_max):
        agent.learn()

def recommender(agent, train_episodes, ep_user, train_df, test_df, train_dict, item_pop_dict,
                max_item_id, mask_list, repr_user, item_emb, episode_id, args,
                min_freq, max_freq, freq, wild_items):
    # Interaction made by [ep_user]
    new_mask_list = copy.copy(mask_list)
    new_mask_list.extend(train_dict[ep_user][:-1])
    # Simulate the enviroment, regarding [ep_user] preferences
    env = environment.Env(ep_user, train_dict[ep_user], list(range(max_item_id + 1)),
                          item_pop_dict, new_mask_list, args.sim_mode, repr_user, item_emb, wild_items, args)

    # Generate transitions (s, a, r, s_) and store in agent replay memory
    interaction_num = setInteraction(env, agent, ep_user, train_df, args, freq)
    if interaction_num <= args.min_obs:
        return None, None, None, None, None
    else:
        global user_num
        user_num += 1

    if episode_id % args.episode_batch == 0 or episode_id == len(train_episodes):
        trainAgent(agent, args.step_max)
        
    # if episode_id % args.eval_freq != 0:
    #     return None, None, None, None, None
    # prec, recall, ndcg, epc, coverage = evaluate(agent, train_episodes, train_df, test_df, train_dict, item_pop_dict,
    #                     max_item_id, mask_list, repr_user, item_emb, args, ckpt= True,
    #                     min_freq= min_freq, max_freq= max_freq)
    # return prec, recall, ndcg, epc, coverage

    return None, None, None, None, None

def evaluate(agent, ep_users, train_df, test_df, train_dict, item_pop_dict,
            max_item_id, mask_list, repr_user, item_emb, args, 
            ckpt= False, encoder= False, user_emb= None,
            min_freq= None, max_freq= None, freq= None, wild_items= None, export_list= False):

    global precision, ndcg, novelty, coverage, ils, interdiv, recall, epc, logs
    precision, ndcg, novelty, coverage, ils, interdiv, recall, epc = [], [], [], [], [], [], [], []

    all_reclist, all_testlist, all_testers = [], [], []

    if encoder:
        assert user_emb is not None

    user_loader = tqdm(ep_users, desc= 'Evaluate ckpt')
    if not ckpt:
        user_loader = ep_users

    if encoder:
        item_weight = item_emb.weight.detach().to(args.device)

    for ep_user in user_loader:
        if not ckpt:
            print('Evaluating user', ep_user)

        last_obs = train_dict[ep_user] # Use all (train) observed history for evaluation
        
        n_obs = len(last_obs)
        cold_obs = last_obs[ : int(n_obs * 0.3)] # Cold observation
        additional_test = last_obs[int(n_obs * 0.3) : ] # The rest observations are added to test set

        ep_mask_list = copy.copy(mask_list)
        
        if args.eval_coldstart:
            query_obs = copy.copy(cold_obs)
        else:
            query_obs = copy.copy(last_obs) # Standard evaluation

        ep_mask_list.extend(query_obs) # Standard evaluation

        # Simulate the enviroment, regarding [ep_user] preferences
        env = environment.Env(ep_user, query_obs, list(range(max_item_id + 1)),
                          item_pop_dict, ep_mask_list, args.sim_mode, repr_user, item_emb, wild_items, args)
        interaction_num = setInteraction(env, agent, ep_user, train_df, args, freq,
                                         augment= False, ckpt= ckpt, evalmode= True)
        if interaction_num <= args.min_obs and not ckpt:
            continue
        
        if not encoder:
            # Generate unseen interaction using learned policy
            rec_list = recommend_offpolicy(env, agent, query_obs)
        else:
            # Generate unseen interaction using pretrained encoder
            rec_list = recommend_encoder(user_emb(torch.IntTensor([ep_user])), item_weight, query_obs, args)
        
        # Ground truth unseen interaction
        test_set = test_df.loc[test_df['user_id'] == ep_user, 'item_id'].tolist()
        test_set = [x for x in test_set if x not in wild_items]

        if args.eval_coldstart:
            test_set.extend(additional_test)

        if not ckpt:
            # Skip users without test interactions
            print(rec_list, test_set)

        # Skip absolute cold-start user or N/A testing interactions
        if len(test_set) == 0 or len(rec_list) == 0:
            continue

        all_reclist.append(rec_list)
        all_testlist.append(test_set)
        all_testers.append(ep_user)

        # Evaluation stats
        match_ = len(set(rec_list) & set(test_set))

        if len(rec_list) == 0:
            precision.append(0.)
        else:
            precision.append(match_ / (len(rec_list)))

        recall.append(match_ / (len(test_set)))

        try:
            ndcg.append(ndcg_metric({ep_user: rec_list}, {ep_user: test_set}))
        except:
            ndcg.append(0.0)

        epc_numer, epc_denom = 0, 0
        for r in range(len(rec_list)):
            candidate = rec_list[r]
            if candidate in test_set:
                # As r is 0-based, np.log2(r + 2) meant to ensure positive-ness
                epc_numer += float((1 - freq[candidate] / max_freq) / np.log2(r + 2))
                epc_denom += float(1. / np.log2(r + 2))

        epc.append((epc_numer, epc_denom))
        coverage.extend(rec_list)

        # novelty.append(novelty_metric(rec_list, env.item_pop_dict))
        # ils.append(ils_metric(rec_list, env.item_sim_matrix))
        # interdiv.append(rec_list)

    final_epc = float(sum([x[0] for x in epc]) / (sum([x[1] for x in epc]) + 1e-5))
    final_coverage = float(len(set(coverage)) / item_emb.weight.shape[0])
    if ckpt:
        print('Evaluation complete!')
        print('####################')
        print(f"Precision@{args.topk}: ", np.round(np.mean(precision), 4), end= '     ')
        print(f"Recall@{args.topk}: ", np.round(np.mean(recall), 4), end= '     ')
        print(f"NDCG@{args.topk}: ", np.round(np.mean(ndcg), 4), end= '     ')
        print(f"EPC@{args.topk}: ", np.round(final_epc, 4), end= '     ')
        print(f"Coverage@{args.topk}: ", np.round(final_coverage, 4))

        valstr = f"Precision@{args.topk}: {np.round(np.mean(precision), 4)}     "
        valstr += f"Recall@{args.topk}: {np.round(np.mean(recall), 4)}     "
        valstr += f"NDCG@{args.topk}: {np.round(np.mean(ndcg), 4)}     "
        valstr += f"EPC@{args.topk}: {np.round(final_epc, 4)}     "
        valstr += f"Coverage@{args.topk}: {np.round(final_coverage, 4)}     "

        logs.append(valstr)
        print('####################')
    
    if export_list:
        return (float(np.mean(precision)), float(np.mean(recall)), float(np.mean(ndcg)), 
                final_epc, final_coverage, all_reclist, all_testlist, all_testers)
    else:
        return (float(np.mean(precision)), float(np.mean(recall)), float(np.mean(ndcg)), 
                final_epc, final_coverage)

def train_dqn(train_df, test_df, query_df, item_pop_dict,
              max_item_id, item_list, mask_list, 
              repr_user, item_emb, user_emb, 
              min_freq, max_freq, freq, wild_items, args):
    
    global precision, ndcg, novelty, coverage, ils, interdiv, epc, logs

    # {user1 : [item1, item3, item5, ...], user2 : [item1, item2], ...}
    print('RL Agent training starts...')

    train_dict = {}
    for index, row in tqdm(train_df.iterrows(), desc= f'Loading train_dict, nrows = {len(train_df)}'):
        train_dict.setdefault(int(row['user_id']), list())
        train_dict[int(row['user_id'])].append(int(row['item_id']))

    # Get max n_obs
    max_obs = 1
    for key, value in train_dict.items():
        max_obs = max(max_obs, len(value))
    args.max_obs = max_obs

    if args.eval_query:
        query_dict = {}
        query_episodes = []
        for index, row in tqdm(query_df.iterrows(), desc= f'Loading query_dict, nrows = {len(query_df)}'):
            query_dict.setdefault(int(row['user_id']), list())
            query_dict[int(row['user_id'])].append(int(row['item_id']))
            query_episodes.append(int(row['user_id']))

        query_episodes = list(set(query_episodes))
        print(f'Queried users, len = {len(query_episodes)}:', query_episodes)

    # Initialize DQN agent with provided environment
    if args.sim_mode == 'stats':
        agent = dqn.DQN(args.obswindow, max_item_id + 1,
                        args.memory, args.agent_lr, args.epsilon,
                        args.replace_freq, args.agent_batch, args.gamma, args.tau, args.topk, 
                        mode= args.dqn_mode, 
                        args= args,
                        item_pop_dict= item_pop_dict)
    elif args.sim_mode == 'item_embedding':
        try:
            assert item_emb is not None
        except:
            print('Item embedding must not be None!')
        agent = dqn.DQN(args.embed_size, max_item_id + 1,
                args.memory, args.agent_lr, args.epsilon,
                args.replace_freq, args.agent_batch, args.gamma, args.tau, args.topk, 
                embd= copy.deepcopy(item_emb), mode= args.dqn_mode,
                args= args,
                item_pop_dict= item_pop_dict)
    elif args.sim_mode == 'user_embedding':
        try:
            assert repr_user is not None
        except:
            print('Representative user must not be None!')
        agent = dqn.DQN(args.embed_size, max_item_id + 1,
                args.memory, args.agent_lr, args.epsilon,
                args.replace_freq, args.agent_batch, args.gamma, args.tau, args.topk, 
                embd= copy.deepcopy(repr_user), mode= args.dqn_mode,
                args= args,
                item_pop_dict= item_pop_dict) 
    else:
        raise NotImplementedError(f"Similarity mode {args.sim_mode} not found!")

    # futures = []
    # executor = ThreadPoolExecutor(max_workers=args.j)
    if not args.all_episodes:
        train_episodes = random.sample(list(train_dict.keys()), args.episode_max)
    else:
        train_episodes = list(train_dict.keys())
    # train_episodes = [125]
    episode_id = 0 # Each episode corresponds to 1 user interactive session

    # Generating initial memory
    print('Initializing memory ...')
    for ep_user in tqdm(train_episodes):
        env = environment.Env(ep_user, train_dict[ep_user], list(range(max_item_id + 1)),
                          item_pop_dict, mask_list, args.sim_mode, repr_user, item_emb, wild_items, args)

        # Generate transitions (s, a, r, s_) and store in agent replay memory
        _ = setInteraction(env, agent, ep_user, train_df, args, freq, augment= False)

        # Check if memory is full
        if agent.check_memory():
            break

    print('Memory initialization complete! Training started ...')
    
    # agent.align_memory()
    ckpt_precision, ckpt_recall, ckpt_ndcg, ckpt_epc, ckpt_coverage = [], [], [], [], []

    for t in range(args.epoch_max):
        episode_id = 0
        random.shuffle(train_episodes)
        if args.replace_freq < 0 and (t + 1) % (-args.replace_freq) == 0:
            agent.net_hard_update()
        for ep_user in tqdm(train_episodes, desc= f'Epoch {t}'):
            episode_id += 1
            # print(f'Episode {episode_id}: User : {ep_user}')
            # future = executor.submit(recommender,
            #                         agent, ep_user, train_df, test_df, train_dict,
            #                         item_sim_dict, item_quality_dict, item_pop_dict,
            #                         max_item_id, mask_list, repr_user, item_emb, args, min_freq, max_freq)
            _prec, _recall, _ndcg, _epc, _coverage = recommender(agent, train_episodes, ep_user, train_df, test_df, 
                                                train_dict, item_pop_dict,
                                                max_item_id, mask_list, repr_user, item_emb, episode_id, args,
                                                min_freq, max_freq, freq, wild_items)

        if args.eval_query:
            _prec, _recall, _ndcg, _epc, _coverage = evaluate(agent, query_episodes, query_df, test_df, query_dict, item_pop_dict,
                        max_item_id, mask_list, repr_user, item_emb, args, ckpt= True,
                        min_freq= min_freq, max_freq= max_freq, freq= freq, wild_items= wild_items)
            
        else:
            _prec, _recall, _ndcg, _epc, _coverage = evaluate(agent, train_episodes, train_df, test_df, train_dict, item_pop_dict,
                        max_item_id, mask_list, repr_user, item_emb, args, ckpt= True,
                        min_freq= min_freq, max_freq= max_freq, freq= freq, wild_items= wild_items)
        
        if _prec is not None:
            ckpt_precision.append(_prec)
        if _recall is not None:
            ckpt_recall.append(_recall)
        if _ndcg is not None:
            ckpt_ndcg.append(_ndcg)
        if _epc is not None:
            ckpt_epc.append(_epc)
        if _coverage is not None:
            ckpt_coverage.append(_coverage)
        #     futures.append(future)
        # wait(futures)

    print('RL agent training complete!')

    print('####################')
    print('Running evaluations on trained agent ...')
    if args.eval_query:
        _, _, _, epc, coverage, reclist, testlist, testers = evaluate(agent, query_episodes, query_df, test_df, query_dict, item_pop_dict,
                    max_item_id, mask_list, repr_user, item_emb, args, ckpt= True,
                    min_freq= min_freq, max_freq= max_freq, freq= freq, wild_items= wild_items, export_list= True)
        
    else:
        _, _, _, epc, coverage, reclist, testlist, testers = evaluate(agent, train_episodes, train_df, test_df, train_dict, item_pop_dict,
                    max_item_id, mask_list, repr_user, item_emb, args, ckpt= True,
                    min_freq= min_freq, max_freq= max_freq, freq= freq, wild_items= wild_items, export_list= True)

    print(f"Precision@{args.topk}: ", np.round(np.mean(precision), 4))
    print(f"Recall@{args.topk}: ", np.round(np.mean(recall), 4))
    print(f"NDCG@{args.topk}: ", np.round(np.mean(ndcg), 4))
    print(f"EPC@{args.topk}: ", np.round(epc, 4))
    print(f"Coverage@{args.topk}: ", np.round(coverage, 4))

    print('####################')
    print('Plotting train curve ...')
    resdict = {
        'freq' : freq,
        'precision' : ckpt_precision, 
        'recall' : ckpt_recall,
        'ndcg' : ckpt_ndcg, 
        'epc' : ckpt_epc,
        'coverage' : ckpt_coverage,
        'reclist' : reclist,
        'testlist' : testlist,
        'testers' : testers
    }
    exps_log = agent.stats_plot(args, resdict)
    print('Train curve finished!')

    print('####################')
    with open(os.path.join(exps_log, 'agent.txt'), 'w') as file:
        file.write('########### RL AGENT EVALUATIONS ###########\n')
        file.write(f"Precision@{args.topk}: {np.round(np.mean(precision), 4)}\n")
        file.write(f"Recall@{args.topk}: {np.round(np.mean(recall), 4)}\n")
        file.write(f"NDCG@{args.topk}: {np.round(np.mean(ndcg), 4)}\n")
        file.write(f"EPC@{args.topk}: {np.round(epc, 4)}\n")
        file.write(f"Coverage@{args.topk}: {np.round(coverage, 4)}\n")
        file.close()

    print('Exporting logs ...')
    logstr = [log + '\n' for log in logs]
    with open(os.path.join(exps_log, 'logs.txt'), 'w') as file:
        file.writelines(logstr)
        file.close()

    if args.eval_graph:
        print('####################')
        print('Running evaluations on trained encoder ...')
        if args.eval_query:
            _, _, _, epc, coverage = evaluate(agent, query_episodes, query_df, test_df, query_dict, item_pop_dict,
                                            max_item_id, mask_list, repr_user, item_emb, args, encoder= True,
                                            min_freq= min_freq, max_freq= max_freq, freq= freq, 
                                            user_emb= user_emb, wild_items= wild_items, ckpt= True)
        else:
            _, _, _, epc, coverage = evaluate(agent, train_episodes, train_df, test_df, train_dict, item_pop_dict,
                                            max_item_id, mask_list, repr_user, item_emb, args, encoder= True,
                                            min_freq= min_freq, max_freq= max_freq, freq= freq, 
                                            user_emb= user_emb, wild_items= wild_items, ckpt= True)

        print(f"Precision@{args.topk}: ", np.round(np.mean(precision), 4))
        print(f"Recall@{args.topk}: ", np.round(np.mean(recall), 4))
        print(f"NDCG@{args.topk}: ", np.round(np.mean(ndcg), 4))
        print(f"EPC@{args.topk}: ", np.round(epc, 4))
        print(f"Coverage@{args.topk}: ", np.round(coverage, 4))

        with open(os.path.join(exps_log, 'graphenc.txt'), 'w') as file:
            file.write('########### ENCODER EVALUATIONS ###########\n')
            file.write(f"Precision@{args.topk}: {np.round(np.mean(precision), 4)}\n")
            file.write(f"Recall@{args.topk}: {np.round(np.mean(recall), 4)}\n")
            file.write(f"NDCG@{args.topk}: {np.round(np.mean(ndcg), 4)}\n")
            file.write(f"EPC@{args.topk}: {np.round(epc, 4)}\n")
            file.write(f"Coverage@{args.topk}: {np.round(coverage, 4)}\n")
            file.close()

    return agent

if __name__ == "__main__":
    pass