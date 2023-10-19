from __future__ import absolute_import
import os
import sys
import bleu
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
from model import Seq2Seq
from construct_exemplar import construct_exemplars, construct_exemplars_ours, calculate_coefficient
from ewc import *
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

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



def convert_examples_to_features(examples, tokenizer, args,stage=None):
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



def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
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
    
    # 27519条数据
    influences = np.sort(influences, axis=0)

    train_idx_sorted = list(np.argsort(influences))
    train_idx_sorted.reverse()
    train_idx_abs_sorted = list(np.argsort(np.absolute(influences)))
    
    most_influential_idx = train_idx_sorted[:int(len(influences) * args.loo_percentage)]
    least_influential_idx = train_idx_sorted[-int(len(influences) * args.loo_percentage):]
    zero_influential_idx = train_idx_abs_sorted[:int(len(influences) * args.loo_percentage)]
    random_influential_idx = random.sample(train_idx_sorted, int(len(influences) * args.loo_percentage))
    loo_influential_idx_list = [most_influential_idx, least_influential_idx, zero_influential_idx, random_influential_idx] # 0: remove most influential, 1: remove least influential, 2: remove zero influential, 3: remove random, 4: unchanged


    if args.do_train:
        # Prepare training data loader
        full_train_examples = read_examples(args.train_filename)
        origin_train_examples = deepcopy(full_train_examples)

        # 检查if的长度是否和训练数据一样长
        assert len(full_train_examples) == len(influences)

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
            logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                            args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
            args.device = device
            # Set seed
            set_seed(args)
            
            train_examples = [te for tei, te in enumerate(full_train_examples) if tei not in loo_inf_idx]

            assert len(train_examples) == len(full_train_examples) - len(loo_influential_idx_list[loo_i])

            config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
            config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)
            
            #budild model
            encoder = model_class.from_pretrained(args.model_name_or_path,config=config)
            decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
            decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
            model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                        beam_size=args.beam_size,max_length=args.max_target_length,
                        sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
            if args.load_model_path is not None:
                logger.info("reload model from {}".format(args.load_model_path))
                model.load_state_dict(torch.load(args.load_model_path))
                
            model.to(device)

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

            # 准备数据
            train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
            all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
            all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
            all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)    
            train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)
            
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps)

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
                'roberta.encoder.layer.4.',
                'roberta.encoder.layer.5.',
                'roberta.encoder.layer.6.',
                'roberta.encoder.layer.7.',
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
            dev_dataset={}
            nb_tr_examples, nb_tr_steps, tr_loss, \
                global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6

            adaptive_weight=1
            logger.info("  Adaptive weight = %f", adaptive_weight)

            for epoch in range(args.num_train_epochs):
                bar = tqdm(train_dataloader,total=len(train_dataloader))
                for batch in bar:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask = batch
                    loss, _, _ = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids, target_mask=target_mask)
                    
                    if args.n_gpu > 1:
                        loss = loss.mean() # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    tr_loss += loss.item()
                    train_loss=round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("loss {}".format(train_loss))
                    nb_tr_examples += source_ids.size(0)
                    nb_tr_steps += 1
                    loss.backward()
        
                    if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                        #Update parameters
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        global_step += 1
                
                
            if args.do_test:
                files=[]
                if args.test_filename is not None:
                    files.append(args.test_filename)
                for idx, file in enumerate(files):   
                    logger.info("Test file: {}".format(file))
                    eval_examples = read_examples(file)

                    eval_number = len(eval_examples)

                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
                    eval_data = TensorDataset(all_source_ids, all_source_mask)   

                    eval_sampler = SequentialSampler(eval_data)
                    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                    model.eval() 
                    p=[]
                    for step, batch in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader))):
                        
                        batch = tuple(t.to(device) for t in batch)
                        source_ids, source_mask= batch                  
                        with torch.no_grad():
                            preds = model(source_ids=source_ids,source_mask=source_mask)  
                            for pred in preds:
                                t=pred[0].cpu().numpy()
                                t=list(t)
                                if 0 in t:
                                    t=t[:t.index(0)]
                                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                                p.append(text)
                    model.train()
                    predictions=[]
                    with open(os.path.join(args.output_dir,"test_{}.output".format(str(args.test_idx))),'w') as f, open(os.path.join(args.output_dir,"test_{}.gold".format(str(args.test_idx))),'w') as f1:
                        
                        for ref, gold in zip(p, eval_examples):
                            if gold.idx != args.test_idx:
                                continue
                            # if gold.idx > 11:
                            #     continue
                            predictions.append(str(gold.idx)+'\t'+ref)
                            f.write(str(gold.idx)+'\t'+ref+'\n')
                            f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

                    (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "test_{}.gold".format(args.test_idx))) 
                    dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                    logger.info("TYPE %d:  %s = %s "%(loo_i, "bleu-4", str(dev_bleu)))
                    logger.info("  "+"*"*20)

                    # 先只选第一个test file
                    logger.info("test file {} finished!".format(file))
                    break
                
                
if __name__ == "__main__":
    main()


