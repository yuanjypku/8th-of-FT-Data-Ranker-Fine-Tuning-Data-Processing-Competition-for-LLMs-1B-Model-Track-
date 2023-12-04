
## 思路
处于精选数据、去除污染数据的思路，我们对所有数据来源（HC3 ConvAI2 instinwild alpacaGPT4 ShareGPT finance dolly instruct prosocial-dialog GPTeacher GPT4all FastChat COIG MOSS）进行排查，以剔除无用或有害的数据源。

具体做法是在30000tokens精选数据集前，分别删除对应来源的数据。最终经过全面比对，训练结果最优的数据删除了***GPTeacher***这一来源。

## 实验代码
1. 设置run.sh中的dataset_en_path为对应的raw_data_en.jsonl（比赛中sh prepare_data_and_models.sh后得到的英文数据集路径）
2. 运行run.sh

## 数据文件
压缩包内的/lm-training/1b_11-03.jsonl_no_GPTeacher.jsonl

## 数据处理代码
run.sh中 Sample 节

## 模型文件
压缩包内的/models/Trian_11-03_no_GPTeacher

## 模型训练代码
run.sh中的Train & Eval节