
该命令完成之后，数据集会被保存到dataset/文件夹中。此外，基于[ERNIE-1.0](https://arxiv.org/abs/1904.09223)微调后的基线模型参数也会被保存在\`finetuned_model/ \`文件夹中，可供直接预测使用。
预测结果会被保存在output/件夹中。
sh predict.sh --model_name_or_path finetuned_model --predict_file dataset/test.json

sh run_eval.sh dataset/dev.json output/dev_predictions.json