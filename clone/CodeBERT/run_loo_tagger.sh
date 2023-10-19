export CUDA_VISIBLE_DEVICES=1
name=emr_ours_adaewc
train_replay_size=1000 #java:1448, python:2245, go:1462. php:2146, javascript:822, ruby:221, clone:2000, svd:1509
weight=2000
k=5
mu=5
pretrained_model="./microsoft/codebert-base"

train_data_file=../POJ_clone/binary/train_1.jsonl
eval_data_file=../POJ_clone/binary/dev_1.jsonl
load_model_path=./saved_models/finetune/task_0/checkpoint-best-acc/model.bin
output_dir=./saved_models/$name/task_0
influence_file_dir=$output_dir/if_score

test_file=$data_dir/$lang/cl/dev_1.jsonl

train_batch_size=16
eval_batch_size=64

num_train_epochs=5

python run_loo_tagger.py --do_train --do_test --model_type roberta --model_name_or_path $pretrained_model --load_model_path $load_model_path \
--tokenizer_name $pretrained_model \
--train_data_file $train_data_file --eval_data_file $eval_data_file --output_dir $output_dir \
--num_train_epochs $num_train_epochs \
--train_batch_size $train_batch_size --eval_batch_size $eval_batch_size \
--block_size 256 --learning_rate 2e-5 --max_grad_norm 1.0 \
--test_idx 8 --influence_file_dir $influence_file_dir --loo_percentage 0.1