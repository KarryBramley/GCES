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

def calculate_grad_sim(data_loader, model, args, param_influence, whole_grads):
    total_batch = len(data_loader)
    scores = []
    bar = tqdm(data_loader,total=len(data_loader))
    for step, batch in enumerate(bar):
        model.zero_grad()

        inputs = batch[0].to(args.device)        
        label = batch[1].to(args.device) 

        loss, logits = model(inputs, label)
        grad = autograd.grad(loss, param_influence)
        import torch.nn.functional as F
        # 计算余弦相似度
        # similarity = F.cosine_similarity(gather_flat_grad(grad), gather_flat_grad(whole_grads), dim=0)
        diff = gather_flat_grad(grad) - gather_flat_grad(whole_grads)
        similarity = torch.norm(diff, p=2, dim=0)
        similarity = similarity.cpu().numpy().item()
        if step % 1000 == 0:
            logger.info(f'score batch {step + 1} of {total_batch}')
        scores.append(similarity)
    # print(scores)
    # scores = np.concatenate(scores)
    return scores

def cross_entropy_loss(prob, labels):
    from jax import numpy as jnp
    labels=labels.float()
    loss=jnp.log(prob[:,0]+1e-10)*labels+jnp.log((1-prob)[:,0]+1e-10)*(1-labels)
    loss=-jnp.mean(loss)
    return loss

def flatten_jacobian(J):
  """Jacobian pytree -> Jacobian matrix"""
  return jnp.concatenate(tree_flatten(tree_map(vmap(jnp.ravel), J))[0], axis=1)

def calculate_grad(model, inputs, label):
    per_sample_loss_fn = lambda x, y: vmap(cross_entropy_loss)(model(x), y)

    inputs = inputs.cpu().numpy()
    inputs = inputs.astype(np.float32)
    label = label.cpu().numpy()
    label = label.astype(np.float32)
    loss_grads = flatten_jacobian(jacrev(per_sample_loss_fn)(inputs, label))
    return loss_grads

def getInstanceGradient(data_loader, model, param_influence, args):
    cnt = 0
    lst = []
    total_batch = len(data_loader)
    scores = []
    bar = tqdm(data_loader,total=len(data_loader))
    for step, batch in enumerate(bar):
        inputs = batch[0].to(args.device)        
        label = batch[1].to(args.device)

        model.zero_grad()
        loss, logit = model(inputs, label)
        # grad = calculate_grad(model, inputs, label)
        import time
        time_start = time.time()
        grad = autograd.grad(loss, param_influence)
        grad = gather_flat_grad(grad)
        
        lst.append(grad)
        cnt += 1
        if cnt == 40:
            time_start = time.time()
            import pickle
            with open('tensor_data.pkl', 'ab') as file:
                pickle.dump(lst, file)
            print(f'{time.time() - time_start}s')
            del grad
            lst = []
        if step % 1000 == 0:
            logger.info(f'gradient calculation batch {step + 1} of {total_batch}')
        
    return np.array(scores)

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
        # 每个任务的train data放到old replay train data字典里面
        with jsonlines.open(train_replay_path) as reader:
            for obj in reader:
                if obj['task_id'] not in old_replay_train_data:
                    old_replay_train_data[obj['task_id']]=[obj]
                else:
                    old_replay_train_data[obj['task_id']].append(obj)
        # 根据if score进行从大到小排序
        for key in old_replay_train_data:
            # old_replay_train_data[key].sort(key = lambda x:x["el2n"], reverse=True)
            old_replay_train_data[key].sort(key = lambda x:x["el2n"])

        # 同理对于验证集来说
        with jsonlines.open(eval_replay_path) as reader:
            for obj in reader:
                if obj['task_id'] not in old_replay_valid_data:
                    old_replay_valid_data[obj['task_id']]=[obj]
                else:
                    old_replay_valid_data[obj['task_id']].append(obj)
        for key in old_replay_valid_data:
            # old_replay_valid_data[key].sort(key = lambda x:x["el2n"], reverse=True)
            old_replay_valid_data[key].sort(key = lambda x:x["el2n"])

    
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
        # 将train data每一个类型的分在一起, clone的话就是0 1两类
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
            # 计算每个类别的replay size
            new_train_replay_size = math.ceil(new_train_replay_size_copy*len(train_exemplar)//train_sum)

            # 0 1标签的样本直接对半分
            # if clas == 0:
            #     new_train_replay_size = math.ceil(new_train_replay_size_copy * 0.9)
            # else:
            #     new_train_replay_size = math.ceil(new_train_replay_size_copy * 0.1)
            

            # 将每个class的train data组成语料库，并且进行向量化，然后KMeans聚类成k个
            vectorizer = CountVectorizer(min_df=10, ngram_range=(1,1))
            transformer = TfidfTransformer()
            train_corpus = [i.origin_source1+i.origin_source2 for i in train_exemplar]
            train_tfidf = transformer.fit_transform(vectorizer.fit_transform(train_corpus)).toarray().tolist()
            cluster_number = args.k
            clf = KMeans(n_clusters=cluster_number, init='k-means++') 
            # train_label是聚类得到的一个个样本的标签
            train_label = clf.fit_predict(train_tfidf)
    
            
            train_dataset = TextDataset(train_exemplar)
            train_sampler = SequentialSampler(train_dataset)

            print('dataset size is:')
            print(len(train_dataset))

            #这里要把batch size调整为1
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1, num_workers=8,pin_memory=True)

            
            # getInstanceGradient(train_dataloader, model, param_influence, args)

            whole_train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=8)
            bar = tqdm(whole_train_dataloader, total=len(whole_train_dataloader))

            whole_grads = None
            for step, batch in enumerate(bar):
                model.eval()
                model.zero_grad()
                inputs = batch[0].to(args.device)        
                label=batch[1].to(args.device)

                loss, logits = model(inputs, label)
                if whole_grads is None:
                    whole_grads = autograd.grad(loss, param_influence)
                else:
                    tmp = autograd.grad(loss, param_influence)
                    model.zero_grad()
                    with torch.no_grad():
                        whole_grads = tuple(x + y for x, y in zip(whole_grads, tmp))
                        del tmp
            whole_grads = tuple(ele / len(bar) for ele in whole_grads)
            # import pickle
            # with open('tensor_data.pkl', 'wb') as file:
            #     pickle.dump(whole_grads, file)
            
            # import pickle
            # with open('tensor_data.pkl', 'rb') as file:
            #     whole_grads = pickle.load(file)
            # print("get whole gradient!")

            # influence_score = calculate_e2ln_score(train_dataloader, model, args)
            influence_score = calculate_grad_sim(train_dataloader, model, args, param_influence, whole_grads)
            for score in influence_score:
                print(type(score))
                break

            # influence_score = np.zeros(len(train_dataloader.dataset))
            # bar = tqdm(train_dataloader,total=len(train_dataloader))
            # grad_one = ()
            # for step, batch in enumerate(bar):
            #     inputs = batch[0].to(args.device)        
            #     label=batch[1].to(args.device) 

            #     lm_loss,logit = model(inputs,label,relay=True)
            #     test_grads = autograd.grad(lm_loss, param_influence)
            #     # 训练梯度维度 flatten后:
            #     # torch.Size([43118593])
            #     # test_grads长度68
            #     for grad in test_grads:   
            #         # print(grad.shape)
            #         tensor_ones_like = torch.ones_like(grad)
            #         grad_one =  grad_one + (tensor_ones_like,)
            #     break
            # # print("$$$$$$$$$$$")
            # # for grad in grad_one:
            # #     print(grad.shape)
            # # print("##########")

            # model.eval()
            # logger.info("######## START COMPUTING IHVP ########")
            # # 这里lisa_dataloader的bs设置为原本的32
            # train_dataloader_lissa = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
            # damping = args.damping
            # logger.info("train number is: ########")
            # logger.info(len(train_exemplar))
            # # 参数train_sum替换了原先代码的len(train_example)
            # if len(train_exemplar) < 10000:
            #     inverse_hvp = get_inverse_hvp_lissa(grad_one, model, args.device, param_influence, train_dataloader_lissa, damping=damping, num_samples=args.lissa_repeat, recursion_depth=int(len(train_exemplar)*args.lissa_depth))
            # else:
            #     inverse_hvp = get_inverse_hvp_lissa(grad_one, model, args.device, param_influence, train_dataloader_lissa, damping=damping, num_samples=args.lissa_repeat, recursion_depth=int(10000*args.lissa_depth))
            # logger.info("######## FINISHED COMPUTING IHVP ########")
            
            # for step, batch in enumerate(bar):
            #     inputs = batch[0].to(args.device)        
            #     label=batch[1].to(args.device) 

            #     model.zero_grad()
            #     lm_loss,logit = model(inputs,label,relay=True)
            #     test_grads = autograd.grad(lm_loss, param_influence)

            #     # model.zero_grad()
            #     # train_grads = autograd.grad(lm_loss, param_influence)

            #     with torch.no_grad():
            #         s_train = [x * y for x, y in zip(test_grads, inverse_hvp)]
                    
            #     ####### L_TRAIN GRADIENT ########
            #     train_grads = test_grads
            #     influence_score[step] = torch.dot(gather_flat_grad(s_train), gather_flat_grad(train_grads)).item()

            #     # logger.info(step)
            #     # logger.info(influence_score[step])
            #     # if step == 3:
            #     #     break
            #     ###############
            # logger.info("######## FINISHED TRAIN IF COMPUTATION ########")  

            # 查看所有样本的if值
            # assert len(influence_score) == len(train_dataloader.dataset)
            # for idx in range(len(train_dataloader.dataset)):
            #     all_train_if_score.append({'code1':train_exemplar[idx].origin_source1, 'code2':train_exemplar[idx].origin_source2, 'label':train_exemplar[idx].origin_target, \
            #                                 'task_id':str(task_id), 'el2n':influence_score[idx]})

            cluster_replay_number = []
            for i in range(cluster_number):
                cluster_replay_number.append(math.ceil(train_label.tolist().count(i)*new_train_replay_size//len(train_label)))

            class_score = {}
            class_abs_score = {}
            class_pos = {}

            # 把每个聚类的if score分到一块
            for idx in range(len(train_label)):
                i = train_label[idx]
                j = influence_score[idx]
                if i not in class_score:
                    class_score[i] = [j]
                    # class_abs_score[i] = [abs(j)]
                    class_pos[i] = [idx]
                else:
                    class_score[i].append(j)
                    # class_abs_score[i].append(abs(j))
                    class_pos[i].append(idx)

            train_topk = []
            for i in range(cluster_number):
                # 从大到小排序
                sorted_idx = list(np.argsort(class_score[i]))
                sorted_idx.reverse()
                print(len(sorted_idx))
                class_score_size = len(class_score[i])
                start_idx = int(0.5 * class_score_size)
                # start_idx = 0
                select_idx = sorted_idx[start_idx - int(0.5 * cluster_replay_number[i]): start_idx + int(0.5 * cluster_replay_number[i])]

                # class_topk = heapq.nsmallest(min(args.mu * cluster_replay_number[i],len(class_score[i])), range(len(class_score[i])), class_score[i].__getitem__)
                train_topk.extend([class_pos[i][j] for j in select_idx])

                # if clas == 0:
                #     # 挑10% IF值最大的
                #     class_topk = heapq.nlargest(min(cluster_replay_number[i],len(class_score[i])), range(len(class_score[i])), class_score[i].__getitem__)
                #     class_topk = np.random.choice(class_topk, int(0.1 * cluster_replay_number[i]), replace=False)
                #     train_topk.extend([class_pos[i][j] for j in class_topk])

                #     # 挑90% IF值绝对值最低的
                #     class_abs_lastk = heapq.nsmallest(min(cluster_replay_number[i],len(class_abs_score[i])), range(len(class_abs_score[i])), class_abs_score[i].__getitem__)
                #     class_abs_lastk = np.random.choice(class_abs_lastk, int(0.9 * cluster_replay_number[i]), replace=False)
                #     train_topk.extend([class_pos[i][j] for j in class_abs_lastk])
                # else:
                #     class_topk = heapq.nsmallest(min(args.mu * cluster_replay_number[i],len(class_score[i])), range(len(class_score[i])), class_score[i].__getitem__)
                #     class_topk = np.random.choice(class_topk, cluster_replay_number[i], replace=False)
                #     train_topk.extend([class_pos[i][j] for j in class_topk])
    
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


            whole_eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=8)
            bar = tqdm(whole_eval_dataloader, total=len(whole_eval_dataloader))

            whole_grads = None
            for step, batch in enumerate(bar):
                model.eval()
                model.zero_grad()
                inputs = batch[0].to(args.device)        
                label=batch[1].to(args.device)

                loss, logits = model(inputs, label)
                if whole_grads is None:
                    whole_grads = autograd.grad(loss, param_influence)
                else:
                    tmp = autograd.grad(loss, param_influence)
                    model.zero_grad()
                    with torch.no_grad():
                        whole_grads = tuple(x + y for x, y in zip(whole_grads, tmp))
                        del tmp
            whole_grads = tuple(ele / len(bar) for ele in whole_grads)
            model.eval()
            # influence_score = calculate_e2ln_score(eval_dataloader, model, args)
            influence_score = calculate_grad_sim(eval_dataloader, model, args, param_influence, whole_grads)
            print(influence_score[:20])

            # bar = tqdm(eval_dataloader,total=len(eval_dataloader))
            # grad_one = ()
            # for step, batch in enumerate(bar):
            #     inputs = batch[0].to(args.device)        
            #     label=batch[1].to(args.device) 

            #     model.zero_grad()
            #     lm_loss,logit = model(inputs,label,relay=True)
            #     test_grads = autograd.grad(lm_loss, param_influence)

            #     # test_grads长度68
            #     for grad in test_grads:
            #         tensor_ones_like = torch.ones_like(grad)
            #         grad_one =  grad_one + (tensor_ones_like,)
            #     break
            

            # model.eval()
            # logger.info("######## START COMPUTING IHVP ########")
            # # 这里lisa_dataloader的bs设置为原本的32
            # eval_dataloader_lissa = DataLoader(eval_dataset, batch_size=8, shuffle=True, drop_last=True)
            # damping = args.damping
            # logger.info("eval number is: ########")
            # logger.info(len(eval_exemplar))
            # inverse_hvp = get_inverse_hvp_lissa(grad_one, model, args.device, param_influence, eval_dataloader_lissa, damping=damping, num_samples=args.lissa_repeat, recursion_depth=int(len(eval_exemplar)*args.lissa_depth))
            # logger.info("######## FINISHED COMPUTING IHVP ########")

            # model.eval()
            # influence_score = np.zeros(len(eval_dataloader.dataset))
            # for step, batch in enumerate(bar):
            #     inputs = batch[0].to(args.device)        
            #     label=batch[1].to(args.device) 
                
            #     model.zero_grad()
            #     lm_loss,logit = model(inputs,label,relay=True)
            #     test_grads = autograd.grad(lm_loss, param_influence)

            #     with torch.no_grad():
            #         s_eval = [x * y for x, y in zip(test_grads, inverse_hvp)]
                
            #     ######## L_TRAIN GRADIENT ########
            #     train_grads = test_grads
            #     influence_score[step] = torch.dot(gather_flat_grad(s_eval), gather_flat_grad(train_grads)).item()
            #     ################
            # logger.info("######## FINISHED EVAL IF COMPUTATION ########")



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
                    # class_abs_score[i] = [abs(j)]
                    class_pos[i] = [idx]
                else:
                    class_score[i].append(j)
                    # class_abs_score[i].append(abs(j))
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

                # if clas == 0:
                #     class_topk = heapq.nlargest(min(cluster_replay_number[i],len(class_score[i])), range(len(class_score[i])), class_score[i].__getitem__)
                #     class_topk = np.random.choice(class_topk, int(0.1 * cluster_replay_number[i]), replace=False)
                #     eval_topk.extend([class_pos[i][j] for j in class_topk])

                #     # 挑75% IF值绝对值最低的
                #     class_abs_lastk = heapq.nsmallest(min(cluster_replay_number[i],len(class_abs_score[i])), range(len(class_abs_score[i])), class_abs_score[i].__getitem__)
                #     class_abs_lastk = np.random.choice(class_abs_lastk, int(0.9 * cluster_replay_number[i]), replace=False)
                #     eval_topk.extend([class_pos[i][j] for j in class_abs_lastk])
                # else:
                #     class_topk = heapq.nsmallest(min(args.mu * cluster_replay_number[i],len(class_score[i])), range(len(class_score[i])), class_score[i].__getitem__)
                #     class_topk = np.random.choice(class_topk, cluster_replay_number[i], replace=False)
                #     eval_topk.extend([class_pos[i][j] for j in class_topk])
    
            for idx in eval_topk:
                new_eval_exemplars.append({'code1':eval_exemplar[idx].origin_source1, 'code2':eval_exemplar[idx].origin_source2, 'label':eval_exemplar[idx].origin_target, \
                                           'task_id':str(task_id), 'el2n':influence_score[idx]})
            logger.info("######## FINISHED EVAL EXAMPLAR SELECTION ########")
                                    
        for idx in old_replay_train_data:
            # if task_id > 0:
            #     old_large_size = int(train_replay_size // task_id * 0.15)
            #     train_exemplars.extend(old_replay_train_data[idx][: int(old_train_replay_size * 0.15)]) # 把IF较大的按比例抽出来
            #     train_exemplars.extend(old_replay_train_data[idx][old_large_size: old_large_size + int(old_train_replay_size * 0.85)]) # 把IF较小的按比例抽出来
            # else:
            train_exemplars.extend(old_replay_train_data[idx][:old_train_replay_size])
        train_exemplars.extend(new_train_exemplars)
        for idx in old_replay_valid_data:
            # if task_id > 0:
            #     old_large_size = int(eval_replay_size // task_id * 0.15)
            #     eval_exemplars.extend(old_replay_valid_data[idx][: int(old_eval_replay_size * 0.15)])
            #     eval_exemplars.extend(old_replay_valid_data[idx][old_large_size: old_large_size + int(old_eval_replay_size * 0.85)])
            # else:
            eval_exemplars.extend(old_replay_valid_data[idx][:old_eval_replay_size])
        eval_exemplars.extend(new_eval_exemplars)
    

    # with jsonlines.open("./saved_models/emr_ours_adaewc/all_train_if_score.jsonl", mode='w') as f:
    #     f.write_all(all_train_if_score)

    with jsonlines.open(train_replay_path, mode='w') as f:
        f.write_all(train_exemplars)
    with jsonlines.open(eval_replay_path, mode='w') as f:
        f.write_all(eval_exemplars)
    
