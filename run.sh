# EXP_DATA_SELECT
exp_name="11-03"
dataset_en_path=/workspace/yjy/lm-training/raw_data_en.jsonl # Processed Data path

# CODE_FILE_SELECT
sampling=/workspace/yjy/lm-training/get_train_dataset_1b_split.py
train_sh=/workspace/yjy/lm-training/train_scripts/deepspeed_train_1b.sh
eval_sh=/workspace/yjy/lm-evaluation-harness/examples/challenge-1B-stage1.sh

# DEFAULT_PATHS
train_data=/workspace/yjy/lm-training/1b_$exp_name.jsonl
sft_model_save_path=/workspace/yjy/models/Train_$exp_name
result_dir=/workspace/yjy/submission/results_$exp_name

pretrained_model_path=/workspace/data/models/falcon-rw-1b
challenge_data_dir=/workspace/data/challenge-data

device=2

## Sample
# 从raw_data_en.jsonl中筛选数据，分别将不含每个数据来源的数据各保留一份
python  $sampling\
    --en_data_dir $dataset_en_path \
    --output_file $train_data

## Train & Eval
# 数据集搜索时的循环头
# for key in HC3 ConvAI2 instinwild alpacaGPT4 ShareGPT finance dolly instruct prosocial-dialog GPTeacher GPT4all FastChat COIG MOSS; do
# 验证结果时的循环头
for key in GPTeacher ; do
    echo no_$key
    split_train_data=${train_data}_no_${key}.jsonl
    split_sft_model_save_path=${sft_model_save_path}_no_${key}
    split_result_dir=${result_dir}_no_${key}
    bash $train_sh\
        $pretrained_model_path \
        $split_train_data \
        $split_sft_model_save_path \
        $device

    for mode in dev board; do
        bash $eval_sh\
            $mode \
            $split_sft_model_save_path \
            $challenge_data_dir \
            $split_result_dir \
            $device
    done
done
