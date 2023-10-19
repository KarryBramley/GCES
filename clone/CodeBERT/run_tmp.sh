export CUDA_VISIBLE_DEVICES=0,1,2
name=emr_ours_adaewc
train_replay_size=2000 #java:1448, python:2245, go:1462. php:2146, javascript:822, ruby:221, clone:2000, svd:1509
weight=2000
k=5
mu=5
train_examplar_path=./saved_models/$name/train_examplar.jsonl
dev_examplar_path=./saved_models/$name/dev_examplar.jsonl
pretrained_model="./microsoft/codebert-base"

#train-finetune
train_data_file=../POJ_clone/binary/train_0.jsonl
eval_data_file=../POJ_clone/binary/dev_0.jsonl
output_dir=./saved_models/$name/task_0
python run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --epoch 5 \
    --block_size 256 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
train_data_file=../POJ_clone/binary/train_1.jsonl
eval_data_file=../POJ_clone/binary/dev_1.jsonl
load_model_path=$output_dir/checkpoint-best-acc/model.bin
output_dir=./saved_models/$name/task_1
python run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --load_model_path=$load_model_path \
    --epoch 5 \
    --block_size 256 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log

#generate-finetune
data_dir=../POJ_clone/binary
test_data_file=$data_dir/test_0.jsonl,$data_dir/test_1.jsonl
output_dir=./saved_models/$name/task_0,./saved_models/$name/task_1
python run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --do_test \
    --train_data_file=$test_data_file \
    --test_data_file=$test_data_file \
    --epoch 5 \
    --block_size 256 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee test_single.log

#train-el2n
train_replay_size=1000 #java:1448, python:2245, go:1462. php:2146, javascript:822, ruby:221, clone:2000, svd:1509
weight=2000
k=5
mu=5
train_examplar_path=./saved_models/$name/train_examplar_if.jsonl
dev_examplar_path=./saved_models/$name/dev_examplar_if.jsonl
pretrained_model="./microsoft/codebert-base"

train_data_file=../POJ_clone/binary/train_0.jsonl
eval_data_file=../POJ_clone/binary/dev_0.jsonl
load_model_path=./saved_models/finetune/task_0/checkpoint-best-acc/model.bin
output_dir=./saved_models/$name/task_0
python run_cl_if.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --load_model_path $load_model_path\
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --mode $name \
    --task_id 0 \
    --k $k \
    --mu $mu \
    --ewc_weight $weight \
    --train_replay_size=$train_replay_size \
    --train_examplar_path=$train_examplar_path \
    --eval_examplar_path=$dev_examplar_path \
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
    --seed 123456  2>&1 | tee train.log 
train_data_file=../POJ_clone/binary/train_1.jsonl
eval_data_file=../POJ_clone/binary/dev_1.jsonl
load_model_path=$output_dir/checkpoint-best-acc/model.bin
output_dir=./saved_models/$name/task_1
python run_cl_if.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --load_model_path $load_model_path\
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --mode $name \
    --task_id 1 \
    --k $k \
    --mu $mu \
    --ewc_weight $weight \
    --train_replay_size=$train_replay_size \
    --train_examplar_path=$train_examplar_path \
    --eval_examplar_path=$dev_examplar_path \
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
    --seed 123456  2>&1 | tee train.log

#generate-el2n
cp -r ./saved_models/finetune/task_0 ./saved_models/multi_task
data_dir=../POJ_clone/binary
test_data_file=$data_dir/test_0.jsonl,$data_dir/test_1.jsonl
output_dir=./saved_models/$name/task_0,./saved_models/$name/task_1
python run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --do_test \
    --train_data_file=$test_data_file \
    --test_data_file=$test_data_file \
    --epoch 5 \
    --block_size 256 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee test-el2n.log

train_replay_size=2000 #java:1448, python:2245, go:1462. php:2146, javascript:822, ruby:221, clone:2000, svd:1509
weight=2000
k=5
mu=5
train_examplar_path=./saved_models/$name/train_examplar.jsonl
dev_examplar_path=./saved_models/$name/dev_examplar.jsonl
pretrained_model="./microsoft/codebert-base"

#train-loss
train_data_file=../POJ_clone/binary/train_0.jsonl
eval_data_file=../POJ_clone/binary/dev_0.jsonl
load_model_path=./saved_models/finetune/task_0/checkpoint-best-acc/model.bin
output_dir=./saved_models/$name/task_0
python run_cl.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --load_model_path $load_model_path\
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --mode $name \
    --task_id 0 \
    --k $k \
    --mu $mu \
    --ewc_weight $weight \
    --train_replay_size=$train_replay_size \
    --train_examplar_path=$train_examplar_path \
    --eval_examplar_path=$dev_examplar_path \
    --epoch 5 \
    --block_size 256 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
train_data_file=../POJ_clone/binary/train_1.jsonl
eval_data_file=../POJ_clone/binary/dev_1.jsonl
load_model_path=$output_dir/checkpoint-best-acc/model.bin
output_dir=./saved_models/$name/task_1
python run_cl.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --load_model_path $load_model_path\
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --mode $name \
    --task_id 1 \
    --k $k \
    --mu $mu \
    --ewc_weight $weight \
    --train_replay_size=$train_replay_size \
    --train_examplar_path=$train_examplar_path \
    --eval_examplar_path=$dev_examplar_path \
    --epoch 5 \
    --block_size 256 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log

# generate-loss
cp -r ./saved_models/finetune/task_0 ./saved_models/multi_task
data_dir=../POJ_clone/binary
test_data_file=$data_dir/test_0.jsonl,$data_dir/test_1.jsonl
output_dir=./saved_models/$name/task_0,./saved_models/$name/task_1
python run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --do_test \
    --train_data_file=$test_data_file \
    --test_data_file=$test_data_file \
    --epoch 5 \
    --block_size 256 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee test-loss.log