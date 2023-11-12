export CUDA_VISIBLE_DEVICES=2
name=finetune
pretrained_model=../../clone/CodeBERT/microsoft/codebert-base

#train
train_data_file=../CodeSearchNet/defect/train_0.jsonl
eval_data_file=../CodeSearchNet/defect/dev_0.jsonl
output_dir=./saved_models/$name/task_0
python run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
#generate
data_dir=../CodeSearchNet/defect
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
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee test.log
