export CUDA_VISIBLE_DEVICES=0,1
lang=java #programming language
name=emr_ours_adaewc
weight=2000
train_replay_size=1448 #java:1448, python:2245, go:1462. php:2146, javascript:822, ruby:221, clone:2000, svd:1509
k=5
mu=5
lr=5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
data_dir=../CodeSearchNet
num_train_epochs=15
train_file=$data_dir/$lang/cl/train_1.jsonl
dev_file=$data_dir/$lang/cl/dev_1.jsonl
pretrained_model=../../clone/CodeBERT/microsoft/codebert-base
output_dir=model/$lang/$name/task_0
train_examplar_path=model/$lang/$name/train_examplar.jsonl
dev_examplar_path=model/$lang/$name/dev_examplar.jsonl
load_model_path=model/$lang/$name/task_1/checkpoint-best-bleu/pytorch_model.bin

python calculate_if_score.py \
 --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --load_model_path $load_model_path --mode $name \
 --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size 1 --learning_rate $lr  --num_train_epochs $num_train_epochs \
 --damping=3e-3 --lissa_repeat=1 --lissa_depth=0.25 \
 --start_test_idx=1 --end_test_idx=10 
