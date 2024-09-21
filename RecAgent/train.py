import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait
from .model import environment, dqn
from .util.metrics_util import ndcg_metric, novelty_metric, ils_metric, interdiv_metric

from tqdm import tqdm
import copy

user_num = 0
precision, ndcg, novelty, coverage, ils, interdiv, recall = [], [], [], [], [], [], []

def stateAugment(observations, history_size, n_augment):
    n_obs = len(observations)
    try:
        assert n_obs >= history_size + 1
    except:
        raise ValueError('Sampling history size exceed known observations size!')
    aug_obsevations = []
    aug_actions = []
    for i in range(n_augment):
        idx = random.choices(list(range(n_obs)), k= history_size + 1)
        history = idx[:-1]
        action = idx[-1]

        aug_actions.append(action)
        aug_obsevations.append(history.copy())
    
    return aug_obsevations, aug_actions

def setInteraction(env, agent, ep_user, train_df, obswindow, augment= True, ckpt= False):
    user_df = train_df[train_df['user_id'] == ep_user]
    observations = user_df['item_id']

    interaction_num = 0
    args = agent.args
    for history_size in range(1, len(observations)):
        if augment:
            aug_obsevations, aug_actions = stateAugment(observations, history_size, args.n_augment)
        else:
            if not ckpt:
                aug_obsevations, aug_actions = stateAugment(observations, history_size, int(args.n_augment / 5))
            else:
                aug_obsevations, aug_actions = stateAugment(observations, history_size, 1)

        for i in range(len(aug_actions)):
            s = np.array(env.reset(aug_obsevations[i]))
            a = int(aug_actions[i])
            s_, r, done = env.step(a)
            agent.store_transition(env.build_state(s).reshape((-1)), a, r, s_.reshape((-1)))
        
        interaction_num += 1
    
    return interaction_num


def recommend_offpolicy(env, agent, last_obs):
    state = np.array(last_obs)
    so = env.reset(state)

    return agent.choose_action(so, env, mode= 'infer')


def trainAgent(agent, step_max):
    for step in tqdm(range(step_max)):
        agent.learn()
    # for step in range(step_max):
    #     agent.learn()

def recommender(agent, train_episodes, ep_user, train_df, test_df, train_dict, item_pop_dict,
                max_item_id, mask_list, repr_user, item_emb, episode_id, args):
    # Newest interaction made by [ep_user]
    last_obs = train_dict[ep_user][-args.obswindow:]
    new_mask_list = copy.copy(mask_list)
    new_mask_list.extend(train_dict[ep_user][:-1])
    # Simulate the enviroment, regarding [ep_user] preferences
    env = environment.Env(ep_user, train_dict[ep_user][-args.obswindow:], list(range(max_item_id + 1)),
                          item_pop_dict, new_mask_list, args.sim_mode, repr_user, item_emb, args)

    # Generate transitions (s, a, r, s_) and store in agent replay memory
    interaction_num = setInteraction(env, agent, ep_user, train_df, args.obswindow)
    if interaction_num <= args.min_obs:
        return None, None, None
    else:
        global user_num
        user_num += 1

    trainAgent(agent, args.step_max)
    if episode_id % args.eval_freq != 0:
        return None, None, None
    prec, recall, ndcg = evaluate(agent, train_episodes, train_df, test_df, train_dict, item_pop_dict,
                        max_item_id, mask_list, repr_user, item_emb, args, ckpt= True)
    return prec, recall, ndcg

def evaluate(agent, ep_users, train_df, test_df, train_dict, item_pop_dict,
            max_item_id, mask_list, repr_user, item_emb, args, ckpt= False):

    global precision, ndcg, novelty, coverage, ils, interdiv, recall
    if ckpt:
        precision, ndcg, novelty, coverage, ils, interdiv, recall = [], [], [], [], [], [], []
        print('Evaluating checkpoint ...')

    for ep_user in ep_users:
        if not ckpt:
            print('Evaluating user', ep_user)

        last_obs = train_dict[ep_user][-args.obswindow:]
        ep_mask_list = copy.copy(mask_list)
        ep_mask_list.extend(train_dict[ep_user][:-1])

        # Simulate the enviroment, regarding [ep_user] preferences
        env = environment.Env(ep_user, train_dict[ep_user][-args.obswindow:], list(range(max_item_id + 1)),
                          item_pop_dict, ep_mask_list, args.sim_mode, repr_user, item_emb, args)
        interaction_num = setInteraction(env, agent, ep_user, train_df, args.obswindow, augment= False, ckpt= ckpt)
        if interaction_num <= args.min_obs:
            continue
        
        # Generated unseen interaction using learned policy
        rec_list = recommend_offpolicy(env, agent, last_obs)
        # Ground truth unseen interaction
        test_set = test_df.loc[test_df['user_id'] == ep_user, 'item_id'].tolist()

        if not ckpt:
            print(rec_list, test_set)

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
        novelty.append(novelty_metric(rec_list, env.item_pop_dict))
        coverage.extend(rec_list)
        # ils.append(ils_metric(rec_list, env.item_sim_matrix))
        interdiv.append(rec_list)

    if ckpt:
        print('Evaluation complete!')
        print('####################')
        print(f"Precision@{args.topk}: ", np.round(np.mean(precision), 4))
        print(f"Recall@{args.topk}: ", np.round(np.mean(recall), 4))
        print(f"NDCG@{args.topk}: ", np.round(np.mean(ndcg), 4))
        print('####################')
    return float(np.mean(precision)), float(np.mean(recall)), float(np.mean(ndcg))

def train_dqn(train_df, test_df, item_pop_dict,
              max_item_id, item_list, mask_list, repr_user, item_emb, args):
    
    global precision, ndcg, novelty, coverage, ils, interdiv

    # {user1 : [item1, item3, item5, ...], user2 : [item1, item2], ...}
    train_dict = {}
    for index, row in train_df.iterrows():
        train_dict.setdefault(int(row['user_id']), list())
        train_dict[int(row['user_id'])].append(int(row['item_id']))

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
    train_episodes = random.sample(list(train_dict.keys()), args.episode_max)
    # train_episodes = [125]
    episode_id = 0 # Each episode corresponds to 1 user interactive session

    # Generating initial memory
    print('Initializing memory ...')
    for ep_user in train_episodes:
        env = environment.Env(ep_user, train_dict[ep_user][-args.obswindow:], list(range(max_item_id + 1)),
                          item_pop_dict, mask_list, args.sim_mode, repr_user, item_emb, args)

        # Generate transitions (s, a, r, s_) and store in agent replay memory
        interaction_num = setInteraction(env, agent, ep_user, train_df, args.obswindow, augment= False)
    
    agent.align_memory()
    ckpt_precision, ckpt_recall, ckpt_ndcg = [], [], []

    for ep_user in train_episodes:
        episode_id += 1
        print(f'Episode {episode_id}: User : {ep_user}')
        # future = executor.submit(recommender,
        #                         agent, ep_user, train_df, test_df, train_dict,
        #                         item_sim_dict, item_quality_dict, item_pop_dict,
        #                         max_item_id, mask_list, repr_user, item_emb, args)
        _prec, _recall, _ndcg = recommender(agent, train_episodes, ep_user, train_df, test_df, 
                                            train_dict, item_pop_dict,
                                            max_item_id, mask_list, repr_user, item_emb, episode_id, args)
        if _prec is not None:
            ckpt_precision.append(_prec)
        if _recall is not None:
            ckpt_recall.append(_recall)
        if _ndcg is not None:
            ckpt_ndcg.append(_ndcg)
    #     futures.append(future)
    # wait(futures)

    print('RL agent training complete!')
    print('####################')
    print('Plotting train curve ...')
    agent.stats_plot(args, ckpt_precision, ckpt_recall, ckpt_ndcg)
    print('Train curve finished!')
    print('####################')
    print('Running evaluations on trained agent ...')
    evaluate(agent, train_episodes, train_df, test_df, train_dict, item_pop_dict,
            max_item_id, mask_list, repr_user, item_emb, args)

    print(f"Precision@{args.topk}: ", np.round(np.mean(precision), 4))
    print(f"Recall@{args.topk}: ", np.round(np.mean(recall), 4))
    print(f"NDCG@{args.topk}: ", np.round(np.mean(ndcg), 4))
    # print("Novelty: ", 1 - np.mean(novelty))
    # print("ILS: ", np.mean(ils))

    return agent

if __name__ == "__main__":
    pass
