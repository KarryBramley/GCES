import sys
import math
import numpy as np
import torch
import torch.autograd as autograd
import argparse
import random
import jsonlines
import heapq
import pandas as pd
from tqdm import tqdm, trange
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset

from jax import jacrev, jit, vmap
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map

from if_util import *
import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def calculate_coefficient(new_data, replay_data):


    vectorizer = CountVectorizer(min_df=10, ngram_range=(1,1))
    transformer = TfidfTransformer()
    train_corpus = [i.origin_source1+i.origin_source2 for i in new_data+replay_data]
    train_tfidf = transformer.fit_transform(vectorizer.fit_transform(train_corpus)).toarray().tolist()
    train_tfidf = np.array(train_tfidf)
    
    replay_feature = train_tfidf[:len(new_data)].mean(axis=0)
    new_feature = train_tfidf[len(new_data):].mean(axis=0)

    a = replay_feature.dot(new_feature)
    b = np.linalg.norm(replay_feature)
    c = np.linalg.norm(new_feature)

    return a/(b*c)

class TextDataset(Dataset):
    def __init__(self, data):
        self.examples = data

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)

def get_l2_norm(preds, labels):
    labels = labels.cpu().item()
    preds = preds.cpu().item()
    
    labels = np.array([1 - labels, labels])
    preds = np.array([1 - preds, preds])
    loss = labels - preds
    # print('type: {0}, loss:{1}'.format(type(loss), loss))
    score = torch.norm(torch.from_numpy(loss), p=2, dim=0)
    return score.cpu().item()

    labels = labels.numpy().tolist()
    preds = preds.numpy().tolist()
    labels = np.array([1 - labels], [labels])
    preds = np.array([1 - preds], [preds])
    loss = labels - preds
    all_score = []
    
    for logit1, logit2 in zip(loss[0], loss[1]):
        one_loss = [logit1, logit2]
        score = torch.norm(torch.from_numpy(one_loss), p=2, dim=0)
        print(score)
        all_score.append(score.cpu().item())
    return all_score
    
def calculate_e2ln_score(data_loader, model, args):
    total_batch = len(data_loader)
    scores = []
    bar = tqdm(data_loader,total=len(data_loader))
    for step, batch in enumerate(bar):
        inputs = batch[0].to(args.device)        
        label = batch[1].to(args.device) 
        if step % 1000 == 0:
            logger.info(f'score batch {step + 1} of {total_batch}')
        scores.append(get_l2_norm(model(inputs), label))
    # print(scores)
    # scores = np.concatenate(scores)
    return scores

# def calculate_grad_sim(data_loader, model, args, param_influence, whole_grads):
#     total_batch = len(data_loader)
#     scores = []
#     bar = tqdm(data_loader,total=len(data_loader))
#     for step, batch in enumerate(bar):
#         model.zero_grad()

#         inputs = batch[0].to(args.device)        
#         label = batch[1].to(args.device) 

#         loss, logits = model(inputs, label)
#         grad = autograd.grad(loss, param_influence)
#         import torch.nn.functional as F
#         # 计算余弦相似度
#         # similarity = F.cosine_similarity(gather_flat_grad(grad), gather_flat_grad(whole_grads), dim=0)
#         diff = gather_flat_grad(grad) - gather_flat_grad(whole_grads)
#         similarity = torch.norm(diff, p=2, dim=0)
#         similarity = similarity.cpu().numpy().item()
#         if step % 1000 == 0:
#             logger.info(f'score batch {step + 1} of {total_batch}')
#         scores.append(similarity)
#     # print(scores)
#     # scores = np.concatenate(scores)
#     return scores


def construct_exemplars(model, args, train_exemplar, eval_exemplar, mode):
    task_id = args.task_id
    train_replay_size = args.train_replay_size
    train_replay_path = args.train_examplar_path
    eval_replay_size = args.eval_replay_size
    eval_replay_path = args.eval_examplar_path

    if task_id>0:
        old_train_replay_size = train_replay_size//(task_id+1)
        old_eval_replay_size = eval_replay_size//(task_id+1)
        new_train_replay_size = train_replay_size-task_id*old_train_replay_size
        new_eval_replay_size = eval_replay_size-task_id*old_eval_replay_size
    else:
        old_train_replay_size = 0
        old_eval_replay_size = 0
        new_train_replay_size = train_replay_size
        new_eval_replay_size = eval_replay_size

    old_replay_train_data = {}
    old_replay_valid_data = {}
    train_exemplars = []
    eval_exemplars = []
    if task_id>0:
        with jsonlines.open(train_replay_path) as reader:
            for obj in reader:
                if obj['task_id'] not in old_replay_train_data:
                    old_replay_train_data[obj['task_id']]=[obj]
                else:
                    old_replay_train_data[obj['task_id']].append(obj)
        with jsonlines.open(eval_replay_path) as reader:
            for obj in reader:
                if obj['task_id'] not in old_replay_valid_data:
                    old_replay_valid_data[obj['task_id']]=[obj]
                else:
                    old_replay_valid_data[obj['task_id']].append(obj)
                    
                    
    if mode == 'random':
        random.shuffle(train_exemplar)
        new_train_exemplars = []
        for idx in range(new_train_replay_size):
            new_train_exemplars.append({'code1':train_exemplar[idx].origin_source1, 'code2':train_exemplar[idx].origin_source2, 'label':train_exemplar[idx].origin_target, 'task_id':str(task_id), 'score':0})
        for idx in old_replay_train_data:
            train_exemplars.extend(old_replay_train_data[idx][:old_train_replay_size])
        train_exemplars.extend(new_train_exemplars)

        random.shuffle(eval_exemplar)
        new_eval_exemplars = []
        for idx in range(new_eval_replay_size):
            new_eval_exemplars.append({'code1':eval_exemplar[idx].origin_source1, 'code2':eval_exemplar[idx].origin_source2, 'label':eval_exemplar[idx].origin_target, 'task_id':str(task_id), 'score':0})
        for idx in old_replay_valid_data:
            eval_exemplars.extend(old_replay_valid_data[idx][:old_eval_replay_size])
        eval_exemplars.extend(new_eval_exemplars)

    with jsonlines.open(train_replay_path, mode='w') as f:
        f.write_all(train_exemplars)
    with jsonlines.open(eval_replay_path, mode='w') as f:
        f.write_all(eval_exemplars)


def construct_exemplars_if(model, args, train_exemplar, eval_exemplar, tokenizer, device, param_influence, mode):
    task_id = args.task_id
    train_replay_size = args.train_replay_size
    train_replay_path = args.train_examplar_path
    eval_replay_size = args.eval_replay_size
    eval_replay_path = args.eval_examplar_path

    # 当前task下replay size的大小
    if task_id>0:
        old_train_replay_size = train_replay_size//(task_id+1)
        old_eval_replay_size = eval_replay_size//(task_id+1)
        new_train_replay_size = train_replay_size-task_id*old_train_replay_size
        new_eval_replay_size = eval_replay_size-task_id*old_eval_replay_size
    else:
        old_train_replay_size = 0
        old_eval_replay_size = 0
        new_train_replay_size = train_replay_size
        new_eval_replay_size = eval_replay_size

    old_replay_train_data = {}
    old_replay_valid_data = {}
    train_exemplars = []
    eval_exemplars = []
    if task_id>0:
        with jsonlines.open(train_replay_path) as reader:
            for obj in reader:
                if obj['task_id'] not in old_replay_train_data:
                    old_replay_train_data[obj['task_id']]=[obj]
                else:
                    old_replay_train_data[obj['task_id']].append(obj)
        for key in old_replay_train_data:
            old_replay_train_data[key].sort(key = lambda x:x["el2n"], reverse=True)
            # old_replay_train_data[key].sort(key = lambda x:x["el2n"])

        with jsonlines.open(eval_replay_path) as reader:
            for obj in reader:
                if obj['task_id'] not in old_replay_valid_data:
                    old_replay_valid_data[obj['task_id']]=[obj]
                else:
                    old_replay_valid_data[obj['task_id']].append(obj)
        for key in old_replay_valid_data:
            old_replay_valid_data[key].sort(key = lambda x:x["el2n"], reverse=True)
            # old_replay_valid_data[key].sort(key = lambda x:x["el2n"])

    
    if mode == 'random':
        # train
        random.shuffle(train_exemplar)
        new_train_exemplars = []
        for idx in range(new_train_replay_size):
            new_train_exemplars.append({'code1':train_exemplar[idx].origin_source1, 'code2':train_exemplar[idx].origin_source2, 'label':train_exemplar[idx].origin_target, \
                                        'task_id':str(task_id), 'score':0})
        for idx in old_replay_train_data:
            train_exemplars.extend(old_replay_train_data[idx][:old_train_replay_size])
        train_exemplars.extend(new_train_exemplars)


        # eval
        random.shuffle(eval_exemplar)
        new_eval_exemplars = []
        for idx in range(new_eval_replay_size):
            new_eval_exemplars.append({'code1':eval_exemplar[idx].origin_source1, 'code2':eval_exemplar[idx].origin_source2, 'label':eval_exemplar[idx].origin_target, \
                                       'task_id':str(task_id), 'score':0})
        for idx in old_replay_valid_data:
            eval_exemplars.extend(old_replay_valid_data[idx][:old_eval_replay_size])
        eval_exemplars.extend(new_eval_exemplars)
    elif mode == 'ours':
        #train
        train_exemplar_class = {}
        eval_exemplar_class = {}
        train_sum=len(train_exemplar)
        eval_sum=len(eval_exemplar)
        new_train_replay_size_copy = new_train_replay_size
        new_eval_replay_size_copy = new_eval_replay_size
        new_eval_exemplars = []
        new_train_exemplars = []

        for i in train_exemplar:
            if i.origin_target not in train_exemplar_class:
                train_exemplar_class[i.origin_target] = [i]
            else:
                train_exemplar_class[i.origin_target].append(i)
        for i in eval_exemplar:
            if i.origin_target not in eval_exemplar_class:
                eval_exemplar_class[i.origin_target] = [i]
            else:
                eval_exemplar_class[i.origin_target].append(i)

        all_train_if_score = []

        for clas in train_exemplar_class:
            print("当前class为 {0}".format(clas))
            # train
            train_exemplar = train_exemplar_class[clas]
            new_train_replay_size = math.ceil(new_train_replay_size_copy*len(train_exemplar)//train_sum)

            vectorizer = CountVectorizer(min_df=10, ngram_range=(1,1))
            transformer = TfidfTransformer()
            train_corpus = [i.origin_source1+i.origin_source2 for i in train_exemplar]
            train_tfidf = transformer.fit_transform(vectorizer.fit_transform(train_corpus)).toarray().tolist()
            cluster_number = args.k
            clf = KMeans(n_clusters=cluster_number, init='k-means++') 
            train_label = clf.fit_predict(train_tfidf)
    
            
            train_dataset = TextDataset(train_exemplar)
            train_sampler = SequentialSampler(train_dataset)

            print('dataset size is:')
            print(len(train_dataset))

            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1, num_workers=8,pin_memory=True)
    
            # import pickle
            # with open('tensor_data.pkl', 'wb') as file:
            #     pickle.dump(whole_grads, file)
            
            # import pickle
            # with open('tensor_data.pkl', 'rb') as file:
            #     whole_grads = pickle.load(file)
            # print("get whole gradient!")

            influence_score = calculate_e2ln_score(train_dataloader, model, args)

            cluster_replay_number = []
            for i in range(cluster_number):
                cluster_replay_number.append(math.ceil(train_label.tolist().count(i)*new_train_replay_size//len(train_label)))

            class_score = {}
            class_abs_score = {}
            class_pos = {}

            for idx in range(len(train_label)):
                i = train_label[idx]
                j = influence_score[idx]
                if i not in class_score:
                    class_score[i] = [j]
                    class_pos[i] = [idx]
                else:
                    class_score[i].append(j)
                    class_pos[i].append(idx)

            train_topk = []
            for i in range(cluster_number):
                sorted_idx = list(np.argsort(class_score[i]))
                sorted_idx.reverse()
                print(len(sorted_idx))
                class_score_size = len(class_score[i])
                start_idx = int(0.5 * class_score_size)
                # start_idx = 0
                select_idx = sorted_idx[start_idx - int(0.5 * cluster_replay_number[i]): start_idx + int(0.5 * cluster_replay_number[i])]

                train_topk.extend([class_pos[i][j] for j in select_idx])
    
            for idx in train_topk:
                new_train_exemplars.append({'code1':train_exemplar[idx].origin_source1, 'code2':train_exemplar[idx].origin_source2, 'label':train_exemplar[idx].origin_target, \
                                            'task_id':str(task_id), 'el2n':influence_score[idx]})
                
            logger.info("######## FINISHED TRAIN EXAMPLAR SELECTION ########")

            #eval
            eval_exemplar = eval_exemplar_class[clas]
            new_eval_replay_size = math.ceil(new_train_replay_size_copy*len(eval_exemplar)//eval_sum)

            vectorizer = CountVectorizer(min_df=10, ngram_range=(1,1))
            transformer = TfidfTransformer()
            eval_corpus = [i.origin_source1+i.origin_source2 for i in eval_exemplar]
            eval_tfidf = transformer.fit_transform(vectorizer.fit_transform(eval_corpus)).toarray().tolist()
            clf = KMeans(n_clusters=cluster_number, init='k-means++') 
            eval_label = clf.fit_predict(eval_tfidf)
    
            eval_dataset = TextDataset(eval_exemplar)
            eval_sampler = SequentialSampler(eval_dataset)

            #这里要把batch size调整为1
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1,num_workers=8,pin_memory=True)
            influence_score = calculate_e2ln_score(eval_dataloader, model, args)
            print(influence_score[:20])

            cluster_replay_number = []
            for i in range(cluster_number):
                cluster_replay_number.append(math.ceil(eval_label.tolist().count(i)*new_eval_replay_size//len(eval_label)))

            class_score = {}
            class_abs_score = {}
            class_pos = {}
            for idx in range(len(eval_label)):
                i = eval_label[idx]
                j = influence_score[idx]
                if i not in class_score:
                    class_score[i] = [j]
                    class_pos[i] = [idx]
                else:
                    class_score[i].append(j)
                    class_pos[i].append(idx)
            eval_topk = []
            for i in range(cluster_number):
                sorted_idx = list(np.argsort(class_score[i]))
                sorted_idx.reverse()
                class_score_size = len(class_score[i])

                start_idx = int(0.5 * class_score_size)
                # start_idx = 0
                select_idx = sorted_idx[start_idx - int(0.5 * cluster_replay_number[i]): start_idx + int(0.5 * cluster_replay_number[i])]

                # class_topk = heapq.nsmallest(min(args.mu * cluster_replay_number[i],len(class_score[i])), range(len(class_score[i])), class_score[i].__getitem__)
                eval_topk.extend([class_pos[i][j] for j in select_idx])
    
            for idx in eval_topk:
                new_eval_exemplars.append({'code1':eval_exemplar[idx].origin_source1, 'code2':eval_exemplar[idx].origin_source2, 'label':eval_exemplar[idx].origin_target, \
                                           'task_id':str(task_id), 'el2n':influence_score[idx]})
            logger.info("######## FINISHED EVAL EXAMPLAR SELECTION ########")
                                    
        for idx in old_replay_train_data:
            train_exemplars.extend(old_replay_train_data[idx][:old_train_replay_size])
        train_exemplars.extend(new_train_exemplars)
        for idx in old_replay_valid_data:
            eval_exemplars.extend(old_replay_valid_data[idx][:old_eval_replay_size])
        eval_exemplars.extend(new_eval_exemplars)
    

    # with jsonlines.open("./saved_models/emr_ours_adaewc/all_train_if_score.jsonl", mode='w') as f:
    #     f.write_all(all_train_if_score)

    with jsonlines.open(train_replay_path, mode='w') as f:
        f.write_all(train_exemplars)
    with jsonlines.open(eval_replay_path, mode='w') as f:
        f.write_all(eval_exemplars)
    
