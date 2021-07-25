.sh文件不能直接用github上的格式，用官方案例notebook的，https://aistudio.baidu.com/aistudio/projectdetail/1639964
解决方案：
方案一
sed -i ‘s/\r//’ 脚本名
方案二
yum -y install dos2unix
dos2unix 脚本名


#################################篇章级事件抽取基线
数据预处理
bash run_duee_fin.sh data_prepare

# 触发词识别模型训练
bash run_duee_fin.sh trigger_train

# 触发词识别预测
bash run_duee_fin.sh trigger_predict
更改：pred_data=${data_dir}/duee_fin_test1.json  # 换其他数据，需要修改它

# 论元识别模型训练
bash run_duee_fin.sh role_train

# 论元识别预测
bash run_duee_fin.sh role_predict
更改：pred_data=${data_dir}/duee_fin_test1.json # 换其他数据，需要修改它

# 枚举分类模型训练
bash run_duee_fin.sh enum_train

# 枚举分类预测
bash run_duee_fin.sh enum_predict
更改：pred_data=${data_dir}/duee_fin_test1.json # 换其他数据，需要修改它

# 数据后处理，提交预测结果
# 结果存放于submit/test_duee_fin.json`
bash run_duee_fin.sh pred_2_submit


#################################句子级事件抽取基线
数据预处理
bash run_duee_1.sh data_prepare

# 训练触发词识别模型
bash run_duee_1.sh trigger_train

# 触发词识别预测
bash run_duee_1.sh trigger_predict
更改：pred_data=${data_dir}/duee_test1.json   # 换其他数据，需要修改它   # 换其他数据，需要修改它

# 论元识别模型训练
bash run_duee_1.sh role_train

# 论元识别预测
bash run_duee_1.sh role_predict
更改：pred_data=${data_dir}/duee_test1.json   # 换其他数据，需要修改它

# 数据后处理，提交预测结果
# 结果存放于submit/test_duee_1.json`
bash run_duee_1.sh pred_2_submit