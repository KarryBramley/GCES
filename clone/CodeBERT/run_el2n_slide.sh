export CUDA_VISIBLE_DEVICES=0
name=finetune
pretrained_model="microsoft/codebert-base"

#train
# 这里task_0的block_size改成了256
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
    --seed 123456

#generate
data_dir=../POJ_clone/binary
test_data_file=$data_dir/test_0.jsonl
output_dir=./saved_models/$name/task_0
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
    --seed 123456
