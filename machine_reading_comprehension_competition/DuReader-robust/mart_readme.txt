一、训练预测
python mart_run_du.py --task_name dureader_robust --model_type ernie_gram --model_name_or_path ernie-gram-zh --max_seq_length 384 --batch_size 12 --learning_rate 3e-5 --num_train_epochs 1 --logging_steps 10 --save_steps 1000 --warmup_proportion 0.1 --weight_decay 0.01 --output_dir ./tmp/dureader-robust/ --do_train --do_predict --device gpu

二、只预测
python mart_run_du.py --task_name dureader_robust --model_type ernie_gram --model_name_or_path=./tmp/dureader-robust/model_1709/ --max_seq_length 384 --batch_size 12 --learning_rate 3e-5 --num_train_epochs 1 --logging_steps 10 --save_steps 1000 --warmup_proportion 0.1 --weight_decay 0.01 --output_dir ./tmp/dureader-robust/ --do_predict --device gpu

--task_name dureader_robust
--model_type ernie_gram
--model_name_or_path ernie-gram-zh
--max_seq_length 384
--batch_size 12
--learning_rate 3e-5
--num_train_epochs 1
--logging_steps 10
--save_steps 1000
--warmup_proportion 0.1
--weight_decay 0.01
--output_dir ./tmp/dureader-robust/
--do_train
--do_predict
--device cpu

task_name: 数据集的名称，不区分大小写，如dureader_robust，cmrc2018, drcd。
model_type: 预训练模型的种类。如bert，ernie，roberta等。
model_name_or_path: 预训练模型的具体名称。如bert-base-chinese，roberta-wwm-ext等。或者是模型文件的本地路径。
output_dir: 保存模型checkpoint的路径。
do_train: 是否进行训练。
do_predict: 是否进行预测。
训练结束后模型会自动对结果进行评估，得到类似如下的输出：

{
  "exact": 72.90049400141143,
  "f1": 86.95957173352133,
  "total": 1417,
  "HasAns_exact": 72.90049400141143,
  "HasAns_f1": 86.95957173352133,
  "HasAns_total": 1417
}
评估结束后模型会自动对测试集进行预测，并将可提交的结果生成在prediction.json中。

NOTE: 如需恢复模型训练，则model_name_or_path只需指定到文件夹名即可。如--model_name_or_path=./tmp/dureader-robust/model_19000/，程序会自动加载模型参数/model_state.pdparams，也会自动加载词表，模型config和tokenizer的config。