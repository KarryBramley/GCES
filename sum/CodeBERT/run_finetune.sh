export CUDA_VISIBLE_DEVICES=0,1,2
lang=java #programming language
name=finetune
lr=5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
data_dir=../CodeSearchNet
num_train_epochs=15
train_file=$data_dir/$lang/cl/train_0.jsonl
dev_file=$data_dir/$lang/cl/dev_0.jsonl
pretrained_model=../../clone/CodeBERT/microsoft/codebert-base
output_dir=model/$lang/$name/task_0


output_dir=model/$lang/$name/task_0
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
test_file=$data_dir/$lang/cl/test_0.jsonl
python run.py --do_test --model_type roberta --model_name_or_path $pretrained_model --load_model_path $test_model --dev_filename $test_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size

output_dir=model/$lang/$name/task_1
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl
python run.py --do_test --model_type roberta --model_name_or_path /microsoft/codebert-base --load_model_path $test_model --dev_filename $test_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size

output_dir=model/$lang/$name/task_2
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl,$data_dir/$lang/cl/test_2.jsonl
python run.py --do_test --model_type roberta --model_name_or_path /microsoft/codebert-base --load_model_path $test_model --dev_filename $test_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size

output_dir=model/$lang/$name/task_3
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl,$data_dir/$lang/cl/test_2.jsonl,$data_dir/$lang/cl/test_3.jsonl
python run.py --do_test --model_type roberta --model_name_or_path /microsoft/codebert-base --load_model_path $test_model --dev_filename $test_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size

output_dir=model/$lang/$name/task_4
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl,$data_dir/$lang/cl/test_2.jsonl,$data_dir/$lang/cl/test_3.jsonl,$data_dir/$lang/cl/test_4.jsonl
python run.py --do_test --model_type roberta --model_name_or_path /microsoft/codebert-base --load_model_path $test_model --dev_filename $test_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size


#Evaluate
python evaluate.py --mode $name --lang $lang