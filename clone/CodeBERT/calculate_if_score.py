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
from model import Model
from copy import deepcopy
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig,RobertaTokenizer,RobertaForSequenceClassification,)
from construct_exemplar_if import construct_exemplars, calculate_coefficient, construct_exemplars_if
from ewc import *

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
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
                 input_tokens,
                 input_ids,
                 label,
                 origin_source1,
                 origin_source2,
                 origin_target

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        self.origin_source1=origin_source1
        self.origin_source2=origin_source2
        self.origin_target=origin_target

def convert_examples_to_features(js,tokenizer,args):
    #source
    code1=' '.join(js['code1'].split())
    code1_tokens=tokenizer.tokenize(code1)
    code2=' '.join(js['code2'].split())
    code2_tokens=tokenizer.tokenize(code2)
    label=int(js['label'])

    code1_tokens=code1_tokens[:args.block_size-2]
    code1_tokens =[tokenizer.cls_token]+code1_tokens+[tokenizer.sep_token]
    code2_tokens=code2_tokens[:args.block_size-2]
    code2_tokens =[tokenizer.cls_token]+code2_tokens+[tokenizer.sep_token]  
    
    code1_ids=tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = args.block_size - len(code1_ids)
    code1_ids+=[tokenizer.pad_token_id]*padding_length
    
    code2_ids=tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = args.block_size - len(code2_ids)
    code2_ids+=[tokenizer.pad_token_id]*padding_length
    
    source_tokens=code1_tokens+code2_tokens
    source_ids=code1_ids+code2_ids
    return InputFeatures(source_tokens,source_ids,label,js['code1'],js['code2'],js['label'])

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

    if args.local_rank == -1:
        train_sampler = SequentialSampler(train_dataset)  
    else:
       train_sampler =  DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
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


    eval_dataset=TextDataset(tokenizer, args, args.eval_data_file)

    # Note that DistributedSampler samples randomly
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_dataset) 
    else:
        eval_sampler = DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size = 1, num_workers=4, pin_memory=True)
    
    start_test_idx = args.start_test_idx
    end_test_idx = args.end_test_idx

    dev_bar = tqdm(eval_dataloader,total=len(eval_dataloader))
    # 测试数据的bs为1
    for test_idx, batch in enumerate(dev_bar):
        print("当前测试idx为{0}".format(test_idx))
        if test_idx < start_test_idx:
            continue
        if test_idx > end_test_idx:
            break
        
        inputs = batch[0].to(args.device)        
        labels = batch[1].to(args.device)

        model.eval()
        model.zero_grad()  
        # 先用test的ground truth label
        loss, logits  = model(inputs, labels)

        test_grads = autograd.grad(loss, param_influence)

        # print('test grad is ')
        # for grad in test_grads:
        #     print(grad.shape)

        logger.info("######## START COMPUTING IHVP ########")
        # 这里lisa_dataloader的bs设置为原本的32
        train_dataloader_lissa = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
        damping = args.damping
        logger.info("train number is: ########")
        logger.info(len(train_dataset))
        # 参数train_sum替换了原先代码的len(train_example)
        inverse_hvp = get_inverse_hvp_lissa(test_grads, model, args.device, param_influence, train_dataloader_lissa, damping=damping, num_samples=args.lissa_repeat, recursion_depth=int(20000*args.lissa_depth))
        logger.info("######## FINISHED COMPUTING IHVP ########")

        influence_score = np.zeros(len(train_dataloader.dataset))
        train_bar = tqdm(train_dataloader,total=len(train_dataloader))
        for train_idx, batch in enumerate(train_bar):
            model.eval()
            inputs = batch[0].to(args.device)        
            labels = batch[1].to(args.device)

            ######## L_TRAIN GRADIENT ########
            model.zero_grad()
            train_loss, logits  = model(inputs, labels)
            train_grads = autograd.grad(train_loss, param_influence)
            influence_score[train_idx] = torch.dot(inverse_hvp, gather_flat_grad(train_grads)).item()
            ################
        
        if_score_path = args.output_dir + '/if_score/'
        if not os.path.exists(if_score_path):
            os.makedirs(if_score_path)   
        pickle.dump(influence_score, open(os.path.join(if_score_path, "influences_test_" + str(test_idx) + ".pkl"), "wb"))


                        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--mode", default="", type=str, help="Mode")
    parser.add_argument("--task_id", default=0, type=int, help="The id of current task.")
    parser.add_argument("--ewc_weight", default=1, type=int, help="The weight of EWC penalty.")
    parser.add_argument("--train_examplar_path", default="", type=str, help="Path of training examplars.")
    parser.add_argument("--eval_examplar_path", default="", type=str, help="Path of valid examplars.")
    parser.add_argument("--train_replay_size", default=100, type=int, help="The size of replayed training examplars.")
    parser.add_argument("--eval_replay_size", default=25, type=int, help="The size of replayed valid examplars.")
    parser.add_argument("--k", default=5, type=int, help="Hyperparameter")
    parser.add_argument("--mu", default=5, type=int, help="Hyperparameter")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")


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
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    model=Model(model,config,tokenizer,args)
    
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

        train_dataset=TextDataset(tokenizer, args, args.train_data_file)
        train(args, train_dataset, model, tokenizer, param_influence)


if __name__ == "__main__":
    main()


