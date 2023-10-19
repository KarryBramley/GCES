from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
from copy import deepcopy
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Model
from construct_exemplar import construct_exemplars, construct_exemplars_ours, calculate_coefficient
from ewc import *
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

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
    

    parser.add_argument('--test_idx',
                        type=int,
                        default=None,
                        required=True)
    parser.add_argument('--influence_file_dir',
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument('--loo_percentage',
                        type=float, default=0.1)
    
    # print arguments
    args = parser.parse_args()
    logger.info(args)


    # 加载influence result
    influences = pickle.load(open(os.path.join(args.influence_file_dir, "influences_test_" + str(args.test_idx) + ".pkl"), "rb"))
    
    # 条数据
    # influences = np.sort(influences, axis=0)

    train_idx_sorted = list(np.argsort(influences))
    train_idx_sorted.reverse()
    train_idx_abs_sorted = list(np.argsort(np.absolute(influences)))
    
    most_influential_idx = train_idx_sorted[:int(len(influences) * args.loo_percentage)]
    least_influential_idx = train_idx_sorted[-int(len(influences) * args.loo_percentage):]
    zero_influential_idx = train_idx_abs_sorted[:int(len(influences) * args.loo_percentage)]
    random_influential_idx = random.sample(train_idx_sorted, int(len(influences) * args.loo_percentage))
    loo_influential_idx_list = [least_influential_idx, most_influential_idx, [],  zero_influential_idx, random_influential_idx] # 0: remove most influential, 1: remove least influential, 2: remove zero influential, 3: remove random, 4: unchanged


    if args.do_train:
        for loo_i, loo_inf_idx in enumerate(loo_influential_idx_list):
            # Setup CUDA, GPU & distributed training
            if args.local_rank == -1 or args.no_cuda:
                device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
                args.n_gpu = torch.cuda.device_count()
            else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
                torch.cuda.set_device(args.local_rank)
                device = torch.device("cuda", args.local_rank)
                torch.distributed.init_process_group(backend='nccl')
                args.n_gpu = 1
            
            args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
            args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
            logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                            args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
            args.device = device
            # Set seed
            set_seed(args.seed)

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
                
            model.to(device)

            # Prepare training data loader
            train_dataset = TextDataset(tokenizer, args, args.train_data_file)
            
            train_examples = [te for tei, te in enumerate(train_dataset) if tei not in loo_inf_idx]
            assert len(train_examples) == len(train_dataset) - len(loo_influential_idx_list[loo_i])

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

            args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_examples)
            else:
                train_sampler = DistributedSampler(train_examples)
            train_dataloader = DataLoader(train_examples, sampler=train_sampler, batch_size=args.train_batch_size)

            # make dir if output_dir not exist
            if os.path.exists(args.output_dir) is False:
                os.makedirs(args.output_dir)
                
            if args.local_rank != -1:
                # Distributed training
                try:
                    from apex.parallel import DistributedDataParallel as DDP
                except ImportError:
                    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

                model = DDP(model)
            elif args.n_gpu > 1:
                # multi-gpu training
                model = torch.nn.DataParallel(model)

            # Prepare optimizer and schedule (linear warmup and decay)
            param_optimizer = list(model.named_parameters())

            frozen = [
                'roberta.embeddings.',
                'roberta.encoder.layer.0.',
                'roberta.encoder.layer.1.',
                'roberta.encoder.layer.2.',
                'roberta.encoder.layer.3.',
            ]
            # frozen = [

            # ]

            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if(not any(fr in n for fr in frozen)) and (not any(nd in n for nd in no_decay))],
                'weight_decay': 0.01},
                {'params': [p for n, p in model.named_parameters() if (not any(fr in n for fr in frozen)) and (any(nd in n for nd in no_decay))], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=len(train_dataloader)*args.num_train_epochs)

            # 准备
            #Start training
            logger.info("***** Running training idx = %d *****", loo_i)
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num epoch = %d", args.num_train_epochs)

            model.train()
            
            args.start_step = 0
            global_step = args.start_step
            tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
            model.zero_grad()
            
            adaptive_weight=1
            logger.info("  Adaptive weight = %f", adaptive_weight)

            for epoch in range(int(args.num_train_epochs)):
                tr_num=0
                train_loss=0

                bar = tqdm(train_dataloader,total=len(train_dataloader))
                for step, batch in enumerate(bar):
                    inputs = batch[0].to(args.device)        
                    labels=batch[1].to(args.device) 
                    model.train()
                    loss, logits = model(inputs, labels)

                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    tr_loss += loss.item()
                    tr_num+=1
                    train_loss+=loss.item()
                    if avg_loss==0:
                        avg_loss=tr_loss
                    avg_loss = round(train_loss / tr_num, 5)
                    bar.set_description("epoch {} loss {}".format(epoch, avg_loss))
        
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        #Update parameters
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        global_step += 1

                        avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                        if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                            logging_loss = tr_loss
                            tr_nb = global_step
                
                
            if args.do_test:
                eval_dataset = TextDataset(tokenizer, args,args.eval_data_file)

                eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
                eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1, num_workers=4, pin_memory=True)
                
                eval_bar = tqdm(eval_dataloader,total=len(eval_dataloader))
                # Eval!
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_dataset))
                logger.info("  Batch size = %d", args.eval_batch_size)

                model.eval()
                logits=[] 
                labels=[]
                for step, batch in enumerate(eval_bar):
                    if step != args.test_idx:
                        continue
                    # if step != 1:
                    #     continue
                    inputs = batch[0].to(args.device)        
                    label=batch[1].to(args.device) 
                    logger.info("label is: {}".format(label))
                    with torch.no_grad():
                        lm_loss, logit = model(inputs,label)
                        logger.info("loss is: {}".format(lm_loss.mean().item()))
                        logger.info("prob is: {}".format(logit))
                        logits.append(logit.cpu().numpy())
                        labels.append(label.cpu().numpy())
                    break

                logits=np.concatenate(logits,0)
                labels=np.concatenate(labels,0)
                preds=logits[:,0]>0.5

                logger.info(preds)

                eval_acc=np.mean(labels==preds)
                
                logger.info("eval acc: {}".format(eval_acc))

if __name__ == "__main__":
    main()


