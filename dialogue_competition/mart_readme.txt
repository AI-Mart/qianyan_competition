PaddleNLP目前为UnifiedTransformer提供了三个中文预训练模型：

unified_transformer-12L-cn 该预训练模型是在大规模中文会话数据集上训练得到的
unified_transformer-12L-cn-luge 该预训练模型是unified_transformer-12L-cn在千言对话数据集上进行微调得到的。
plato-mini 该模型使用了十亿级别的中文闲聊对话数据进行预训练。

一、数据准备
切换到lic2021_baseline目录，按住shift右键运行powershell,复制下面指令运行，即可在datasets文件夹生成训练数据、验证数据、测试数据，convert_data_to_numerical.py函数可配置采样多少数据，哪些数据作为训练集测试集和验证集。
python ./tools/convert_data_to_numerical.py ./tools/spm.model

本地内部调试
python convert_data_to_numerical.py
spm.model


二、模型训练
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" --log_dir ./log finetune.py \
    --model_name_or_path=unified_transformer-12L-cn-luge \
    --train_data_path=./datasets/train.txt \
    --valid_data_path=./datasets/valid.txt \
    --save_dir=./checkpoints \
    --logging_steps=500 \
    --save_steps=8000 \
    --seed=2021 \
    --epochs=10 \
    --batch_size=8192 \
    --lr=1e-5 \
    --weight_decay=0.01 \
    --warmup_steps=4000 \
    --max_grad_norm=0.1 \
    --sort_pool_size=65536 \
    --device=gpu


二、模型预测 直接预训练模型
export CUDA_VISIBLE_DEVICES=0
# GPU启动，预测仅支持单卡
python infer.py \
    --model_name_or_path=unified_transformer-12L-cn-luge \
    --test_data_path=./datasets/test.txt \
    --output_path=./predict.txt \
    --logging_steps=500 \
    --seed=2021 \
    --batch_size=4 \
    --min_dec_len=1 \
    --max_dec_len=64 \
    --num_samples=20 \
    --decode_strategy=sampling \
    --top_k=5 \
    --device=gpu

二、模型预测 训练好的模型
export CUDA_VISIBLE_DEVICES=0
# GPU启动，预测仅支持单卡
python infer.py \
    --model_name_or_path=unified_transformer-12L-cn-luge \
    --save_dir=./checkpoints/model_40000/model_state.pdparams \
    --test_data_path=./datasets/test.txt \
    --output_path=./predict.txt \
    --logging_steps=500 \
    --seed=2021 \
    --batch_size=8\
    --min_dec_len=1 \
    --max_dec_len=64 \
    --num_samples=20 \
    --decode_strategy=sampling \
    --top_k=5 \
    --device=gpu


# cpu
--model_name_or_path=unified_transformer-12L-cn-luge
--test_data_path=./datasets/test.txt
--output_path=./predict.txt
--logging_steps=500
--seed=2021
--batch_size=4
--min_dec_len=1
--max_dec_len=64
--num_samples=20
--decode_strategy=sampling
--top_k=5
--device=cpu


