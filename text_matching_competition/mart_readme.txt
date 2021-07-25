用vim 打开脚本 ，发现脚本格式为doc ，则需要将脚本格式转为unix
:set fileformat=unix  修改格式
：wq  保存退出
# 训练 所以数据
# 模型训练big_all

python train_pointwise.py --device gpu --save_dir ./checkpoints_big_all --batch_size 128 --learning_rate 5E-5  --input_file_train ./datas/big_all/train.tsv --input_file_dev ./datas/big_all/dev.tsv  --epochs 10 --warmup_proportion 0.1 --init_from_ckpt ./checkpoints_big_all/model_30310/model_state.pdparams --init_from_opt ./checkpoints_big_all/model_30310/optimizer.pdopt

--device cpu
--save_dir ./checkpoints_big_all
--batch_size 32
--learning_rate 2E-5
--input_file_train ./datas/big_all/train.tsv
--input_file_dev ./datas/big_all/dev.tsv
--init_from_ckpt ./checkpoints_lcqmc/model_1400/model_state.pdparams


# 训练
# 模型训练lcqmc
python train_pointwise.py --device gpu --save_dir ./checkpoints_lcqmc --batch_size 128 --learning_rate 5E-5  --input_file_train ./datas/lcqmc/train.tsv --input_file_dev ./datas/lcqmc/dev.tsv  --epochs 10 --warmup_proportion 0.1 --init_from_ckpt ./checkpoints_lcqmc/model_1400/model_state.pdparams

--device cpu
--save_dir ./checkpoints_lcqmc
--batch_size 32
--learning_rate 2E-5
--input_file_train ./datas/lcqmc/train.tsv
--input_file_dev ./datas/lcqmc/dev.tsv
--init_from_ckpt ./checkpoints_lcqmc/model_1400/model_state.pdparams


# 模型训练bq_corpus
python train_pointwise.py --device gpu --save_dir ./checkpoints_bq_corpus --batch_size 128 --learning_rate 5E-5 --input_file_train ./datas/bq_corpus/train.tsv --input_file_dev ./datas/bq_corpus/dev.tsv --epochs 10 --warmup_proportion 0.1  --init_from_ckpt ./checkpoints_bq_corpus/model_12500/model_state.pdparams

--device cpu
--save_dir ./checkpoints_bq_corpus
--batch_size 32
--learning_rate 2E-5
--input_file_train ./datas/bq_corpus/train.tsv
--input_file_dev ./datas/bq_corpus/dev.tsv


# 模型训练paws-x-zh
python train_pointwise.py --device gpu --save_dir ./checkpoints_paws-x-zh --batch_size 32 --learning_rate 2E-6  --input_file_train ./datas/paws-x-zh/train.tsv --input_file_dev ./datas/paws-x-zh/dev.tsv --epochs 20 --warmup_proportion 0.1 --init_from_ckpt ./checkpoints_paws-x-zh/model_3840/model_state.pdparams

--device cpu
--save_dir ./checkpoints_paws-x-zh
--batch_size 32
--learning_rate 2E-5
--input_file_train ./datas/paws-x-zh/train.tsv
--input_file_dev ./datas/paws-x-zh/dev.tsv




# 预测
# 模型预测lcqmc
python predict_pointwise.py --device gpu --params_path './checkpoints_lcqmc/model_18660/model_state.pdparams' --input_file './datas/lcqmc/test.tsv' --output_file './submit/lcqmc.tsv' --batch_size 128

--device gpu
--params_path './checkpoints_lcqmc/model_4400/model_state.pdparams'
--input_file './datas/lcqmc/test.tsv'
--output_file './submit/lcqmc.tsv'
--batch_size 128
--max_seq_length 64

# 模型预测bq_corpus
python predict_pointwise.py --device gpu --params_path './checkpoints_bq_corpus/model_7820/model_state.pdparams' --input_file './datas/bq_corpus/test.tsv' --output_file './submit/bq_corpus.tsv'
--device gpu
--params_path './checkpoints_bq_corpus/model_4400/model_state.pdparams'
--input_file './datas/bq_corpus/test.tsv'
--output_file './submit/bq_corpus.tsv'


# 模型预测paws-x-zh
python predict_pointwise.py --device gpu --params_path './checkpoints_paws-x-zh/model_23040/model_state.pdparams' --input_file './datas/paws-x-zh/test.tsv' --output_file './submit/paws-x.tsv'
--device gpu
--params_path './checkpoints_paws-x-zh/model_4400/model_state.pdparams'
--input_file './datas/paws-x-zh/test.tsv'
--output_file './submit/paws-x.tsv'


####################本地
####################本地
####################本地
# 预测
# 模型预测lcqmc
python predict_pointwise.py --device gpu --params_path './checkpoints_lcqmc/model_18660/model_state.pdparams' --input_file './datas/lcqmc/test.tsv' --output_file './submit/lcqmc.tsv' --batch_size 128

--device cpu
--params_path ./batch_neg_v1.0/model_state.pdparams
--input_file ./datas/lcqmc/test.tsv
--output_file ./submit/lcqmc.tsv

# 模型预测bq_corpus
python predict_pointwise.py --device gpu --params_path './checkpoints_bq_corpus/model_7820/model_state.pdparams' --input_file './datas/bq_corpus/test.tsv' --output_file './submit/bq_corpus.tsv'

--device cpu
--params_path ./batch_neg_v1.0/model_state.pdparams
--input_file ./datas/bq_corpus/test.tsv
--output_file ./submit/bq_corpus.tsv


# 模型预测paws-x-zh
python predict_pointwise.py --device gpu --params_path './checkpoints_paws-x-zh/model_23040/model_state.pdparams' --input_file './datas/paws-x-zh/test.tsv' --output_file './submit/paws-x.tsv'

--device cpu
--params_path ./batch_neg_v1.0/model_state.pdparams
--input_file ./datas/paws-x-zh/test.tsv
--output_file ./submit/paws-x.tsv


##############big_all#################
##############big_all#################
##############big_all#################
# 预测
# 模型预测lcqmc
python predict_pointwise.py --device gpu --params_path './checkpoints_big_all/model_60620/model_state.pdparams' --input_file './datas/lcqmc/test.tsv' --output_file './submit/lcqmc.tsv' --batch_size 128


# 模型预测bq_corpus
python predict_pointwise.py --device gpu --params_path './checkpoints_big_all/model_60620/model_state.pdparams' --input_file './datas/bq_corpus/test.tsv' --output_file './submit/bq_corpus.tsv'

# 模型预测paws-x-zh
python predict_pointwise.py --device gpu --params_path './checkpoints_big_all/model_60620/model_state.pdparams' --input_file './datas/paws-x-zh/test.tsv' --output_file './submit/paws-x.tsv'
