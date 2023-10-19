from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

from if_util import *

import numpy as np
import torch
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import Seq2Seq
from copy import deepcopy
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig,RobertaTokenizer,RobertaModel,)
from construct_exemplar_if import construct_exemplars, calculate_coefficient, construct_exemplars_if
from ewc import *

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
}

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 origin_target,
                 origin_source,
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.origin_target = origin_target
        self.origin_source = origin_source
        self.code=''

def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            code=' '.join(js['code_tokens']).replace('\n',' ')
            code=' '.join(code.strip().split())
            nl=' '.join(js['docstring_tokens']).replace('\n','')
            nl=' '.join(nl.strip().split())            
            examples.append(
                Example(
                        idx = idx,
                        source=code,
                        target = nl,
                        origin_source=js['code_tokens'],
                        origin_target=js['docstring_tokens'],
                        )
            )
    return examples

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask  

def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   
   
        if example_index < 5 and False:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
                
                logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )
    return features

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None,extra=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js,tokenizer,args))
        self.origin_data = deepcopy(self.examples)

        if extra is not None:
            self.replay_examples = []
            if extra == 'train':
                examplar_path = args.train_examplar_path
            else:
                examplar_path = args.eval_examplar_path
            with open(examplar_path) as f:
                for line in f:
                    js=json.loads(line.strip())
                    self.replay_examples.append(convert_examples_to_features(js,tokenizer,args))
                if 'emr' in args.mode or args.mode in ['hybrid']:
                    self.examples.extend(self.replay_examples)

        if False and 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)
            
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer, param_influence):
    """ Train the model """ 
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_features = convert_examples_to_features(train_dataset, tokenizer, args, stage='train')
    all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)        
    all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)    
    train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)

    if args.local_rank == -1:
        train_sampler = SequentialSampler(train_data)  
    else:
       train_sampler =  DistributedSampler(train_data)
    
    train_dataloader = DataLoader(train_data, sampler=train_sampler, 
                                  batch_size=1, num_workers=4, pin_memory=True)
    
    # args.max_steps=args.epoch * len( train_dataloader)
    args.save_steps=len( train_dataloader)
    args.warmup_steps=len( train_dataloader)
    args.logging_steps=len( train_dataloader)
    # args.num_train_epochs=args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    # no_decay = ['bias', 'LayerNorm.weight']

    frozen = [
            'embeddings.',
            'encoder.layer.0.',
            'encoder.layer.1.',
            'encoder.layer.2.',
            'encoder.layer.3.',
            'encoder.layer.4.',
            'encoder.layer.5.',
            'encoder.layer.6.',
            'encoder.layer.7.',
        ]
    # frozen = [
    #     ]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in frozen)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in frozen)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    

    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr=0.0
    best_acc=0.0
    # model.resize_token_embeddings(len(tokenizer))

    adaptive_weight = 1
    logger.info("  Adaptive weight = %f", adaptive_weight)

    eval_examples = read_examples(args.dev_filename)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
    
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)        
    all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)   
    
    eval_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)   
    
    # Note that DistributedSampler samples randomly
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data) 
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size = 1, num_workers=4, pin_memory=True)
    
    start_test_idx = args.start_test_idx
    end_test_idx = args.end_test_idx

    bar = tqdm(eval_dataloader,total=len(eval_dataloader))
    # 测试数据的bs为1
    for test_idx, batch in enumerate(bar):
        print("当前测试idx为{0}".format(test_idx))
        if test_idx < start_test_idx:
            continue
        if test_idx > end_test_idx:
            break
        
        source_ids = batch[0].to(args.device)        
        source_mask = batch[1].to(args.device)
        target_ids = batch[2].to(args.device) 
        target_mask = batch[3].to(args.device) 

        model.eval()
        model.zero_grad()  
        # 先用test的ground truth label
        test_loss, _, num  = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask)

        test_loss = test_loss / num
        test_grads = autograd.grad(test_loss, param_influence)

        # print('test grad is ')
        # for grad in test_grads:
        #     print(grad.shape)

        logger.info("######## START COMPUTING IHVP ########")
        # 这里lisa_dataloader的bs设置为原本的32
        train_dataloader_lissa = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
        damping = args.damping
        logger.info("train number is: ########")
        logger.info(len(train_data))
        # 参数train_sum替换了原先代码的len(train_example)
        inverse_hvp = get_inverse_hvp_lissa(test_grads, model, args.device, param_influence, train_dataloader_lissa, damping=damping, num_samples=args.lissa_repeat, recursion_depth=int(10000*args.lissa_depth))
        logger.info("######## FINISHED COMPUTING IHVP ########")

        influence_score = np.zeros(len(train_dataloader.dataset))
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        for train_idx, batch in enumerate(bar):
            model.eval()
            source_ids = batch[0].to(args.device)        
            source_mask = batch[1].to(args.device)
            target_ids = batch[2].to(args.device) 
            target_mask = batch[3].to(args.device) 

            ######## L_TRAIN GRADIENT ########
            model.zero_grad()
            train_loss, _, num  = model(source_ids, source_mask, target_ids, target_mask)
            train_loss = train_loss / num
            train_grads = autograd.grad(train_loss, param_influence)
            influence_score[train_idx] = torch.dot(inverse_hvp, gather_flat_grad(train_grads)).item()
            ################
        
        if_score_path = args.output_dir + '/if_score/'
        if not os.path.exists(if_score_path):
            os.makedirs(if_score_path)   
        pickle.dump(influence_score, open(os.path.join(if_score_path, "influences_test_" + str(test_idx) + ".pkl"), "wb"))


                        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--mode", default="", type=str, help="Mode")
    parser.add_argument("--task_id", default=0, type=int, help="The id of current task.")
    parser.add_argument("--ewc_weight", default=1, type=int, help="The weight of EWC penalty.")
    parser.add_argument("--train_examplar_path", default="", type=str, help="Path of training examplars.")
    parser.add_argument("--eval_examplar_path", default="", type=str, help="Path of valid examplars.")
    parser.add_argument("--train_replay_size", default=100, type=int, help="The size of replayed training examplars.")
    parser.add_argument("--eval_replay_size", default=25, type=int, help="The size of replayed valid examplars.")
    parser.add_argument("--k", default=5, type=int, help="Hyperparameter")
    parser.add_argument("--mu", default=5, type=int, help="Hyperparameter")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # IF计算
    parser.add_argument("--lissa_repeat", default=1, type=int)
    parser.add_argument("--lissa_depth", default=1.0, type=float)
    parser.add_argument('--damping', type=float, default=0.0, help="probably need damping for deep models")
    parser.add_argument('--start_test_idx',
                        type=int,
                        default=-1,
                        help="when not -1, --test_idx will be disabled")
    parser.add_argument('--end_test_idx',
                        type=int,
                        default=-1,
                        help="when not -1, --test_idx will be disabled")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        # 第3张显卡空的 这里指定显卡2的话, bash脚本里的export cuda visible要改
        print('local rank is {0}'.format(args.local_rank))
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.eval_replay_size = args.train_replay_size//8
    args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))


    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    # 如果存在保存的检查点
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)
    
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
    
    # cnt = 0
    # all_cnt = 0
    # for n, p in model.named_parameters():
    #     all_cnt += 1
    #     if p.requires_grad:
    #         cnt+=1
    # print('可训练参数{0}'.format(cnt))
    # print('总参数{0}'.format(all_cnt))

    model.to(args.device)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    

    # Training
    if args.do_train:
        #########  冻结大部分层  #########
        param_optimizer = list(model.named_parameters())

        frozen = [
            'embeddings.',
            'encoder.layer.0.',
            'encoder.layer.1.',
            'encoder.layer.2.',
            'encoder.layer.3.',
            'encoder.layer.4.',
            'encoder.layer.5.',
            'encoder.layer.6.',
            'encoder.layer.7.',
        ]
        # frozen = [
        # ]
        param_influence = []
        for n, p in param_optimizer:
            if (not any(fr in n for fr in frozen)):
                param_influence.append(p)
            elif 'embeddings.word_embeddings.' in n:
                pass 
            else:
                p.requires_grad = False
        ###################

        param_size = 0
        for p in param_influence:
            tmp_p = p.clone().detach()
            param_size += torch.numel(tmp_p)
        logger.info("  Parameter size = %d", param_size)

        train_dataset = read_examples(args.train_filename)
        train(args, train_dataset, model, tokenizer, param_influence)


if __name__ == "__main__":
    main()


