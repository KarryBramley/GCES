export CUDA_VISIBLE_DEVICES=2
lang=java #programming language
name=emr_ours_adaewc
weight=2000
train_replay_size=1448 #java:1448, python:2245, go:1462. php:2146, javascript:822, ruby:221, clone:2000, svd:1509
k=5
mu=5
lr=5e-5

train_batch_size=32
eval_batch_size=128

beam_size=10
source_length=256
target_length=128
data_dir=../CodeSearchNet
num_train_epochs=15

train_file=$data_dir/$lang/cl/train_1.jsonl
dev_file=$data_dir/$lang/cl/dev_1.jsonl

pretrained_model=../../clone/CodeBERT/microsoft/codebert-base
output_dir=model/$lang/$name/task_0

load_model_path=model/$lang/finetune/task_0/checkpoint-best-bleu/pytorch_model.bin

influence_file_dir=$output_dir/if_score

test_file=$data_dir/$lang/cl/dev_1.jsonl
python run_loo_tagger.py --do_train --do_test --model_type roberta --model_name_or_path $pretrained_model --load_model_path $load_model_path \
--train_filename $train_file --test_filename $test_file --output_dir $output_dir \
--max_source_length $source_length --max_target_length $target_length --beam_size $beam_size \
--num_train_epochs $num_train_epochs \
--train_batch_size $train_batch_size --eval_batch_size $eval_batch_size \
--test_idx 1 --influence_file_dir $influence_file_dir --loo_percentage 0.2