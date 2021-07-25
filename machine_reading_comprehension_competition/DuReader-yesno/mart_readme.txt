一、训练
python run_du.py --model_type ernie_gram --model_name_or_path=./tmp/dureader-yesno/model_1/ --max_seq_length 384 --batch_size 12 --learning_rate 3e-5 --num_train_epochs 2 --logging_steps 200 --save_steps 1000 --warmup_proportion 0.1 --weight_decay 0.01 --output_dir ./tmp/dureader-yesno/ --device gpu


--model_type ernie_gram
--model_name_or_path ernie-gram-zh
--max_seq_length 384
--batch_size 12
--learning_rate 3e-5
--num_train_epochs 2
--logging_steps 200
--save_steps 1000
--warmup_proportion 0.1
--weight_decay 0.01
--output_dir ./tmp/dureader-yesno/
--device cpu

model_type: 预训练模型的种类。如bert，ernie，roberta等。
model_name_or_path: 预训练模型的具体名称。如bert-base-uncased，bert-large-cased等。或者是模型文件的本地路径。
output_dir: 保存模型checkpoint的路径。
训练结束后模型会自动对结果进行评估，得到类似如下的输出：
accu: 0.874954
评估结束后模型会自动对测试集进行预测，并将可提交的结果生成在prediction.json中。
NOTE: 如需恢复模型训练，则model_name_or_path只需指定到文件夹名即可。如--model_name_or_path=./tmp/dureader-yesno/model_19000/，程序会自动加载模型参数/model_state.pdparams，也会自动加载词表，模型config和tokenizer的config。