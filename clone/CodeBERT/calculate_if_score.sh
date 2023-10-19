export CUDA_VISIBLE_DEVICES=2
name=emr_ours_adaewc
train_replay_size=1000 #java:1448, python:2245, go:1462. php:2146, javascript:822, ruby:221, clone:2000, svd:1509
weight=2000
k=5
mu=5
train_examplar_path=./saved_models/$name/train_examplar_if.jsonl
dev_examplar_path=./saved_models/$name/dev_examplar_if.jsonl
pretrained_model="./microsoft/codebert-base"

train_data_file=../POJ_clone/binary/train_1.jsonl
eval_data_file=../POJ_clone/binary/dev_1.jsonl
load_model_path=./saved_models/$name/task_1/checkpoint-best-acc/model.bin
output_dir=./saved_models/$name/task_0

python calculate_if_score.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --load_model_path $load_model_path\
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --mode $name \
    --epoch 5 \
    --block_size 256 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --damping=3e-3 \
    --lissa_repeat=1 \
    --lissa_depth=0.25 \
    --start_test_idx=8 --end_test_idx=10 
