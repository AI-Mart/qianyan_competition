o、
安装必要的包
pip install paddlenlp --upgrade
pip install -r requirements.txt

一、
第一步都是打开.sh文件，把doc变成unix格式，
:set fileformat=unix
:wq

一、1.1
下载ERNIE预训练模型
# 执行完成后会得到 data/ernie/ernie_1.0_base_ch 目录，其中包含ERNIE模型相关的配置、词表和模型参数
bash data/download_ernie1.0.sh

二、下载数据集和官方比赛数据集一致
bash data/download_model_data.sh
# 下载模型训练、测试数据
# 得到的数据包括 DuSQL, NL2SQL 和 CSpider 三个数据集（同[千言-语义解析](https://aistudio.baidu.com/aistudio/competition/detail/47)任务的三个数据集）

三、下载训练好的 Text2SQL 模型
bash data/download_trained_model.sh
# 下载训练好的 Text2SQL 模型
# 得到的数据包括：
#   data
#   ├── trained_model
#   │   ├── dusql.pdparams
#   │   ├── nl2sql.pdparams
#   │   ├── cspider.pdparams

四、数据预处理
对原始数据进行格式转换、依赖信息补充等，以适配模型的输入。下面以DuSQL数据集为例进行说明。
获取 Schema Linking 结果
将 schema linking 独立出来，以便于针对这一步进行特定优化，可有效提升模型最终的效果。
#################
训练集 DuSQL
bash run.sh ./script/schema_linking.py \
        -s data/DuSQL/db_schema.json \
        -c data/DuSQL/db_content.json \
        -o data/DuSQL/match_values_train.json \
        data/DuSQL/train.json --is-train

# 开发集
bash run.sh ./script/schema_linking.py \
        -s data/DuSQL/db_schema.json \
        -c data/DuSQL/db_content.json \
        -o data/DuSQL/match_values_dev.json \
        data/DuSQL/dev.json

# 测试集
bash run.sh ./script/schema_linking.py \
        -s data/DuSQL/db_schema.json \
        -c data/DuSQL/db_content.json \
        -o data/DuSQL/match_values_test.json \
        data/DuSQL/test.json

#################
训练集 CSpider
bash run.sh ./script/schema_linking.py \
        -s data/CSpider/db_schema.json \
        -c data/CSpider/db_content.json \
        -o data/CSpider/match_values_train.json \
        data/CSpider/train.json --is-train

# 开发集
bash run.sh ./script/schema_linking.py \
        -s data/CSpider/db_schema.json \
        -c data/CSpider/db_content.json \
        -o data/CSpider/match_values_dev.json \
        data/CSpider/dev.json

# 测试集
bash run.sh ./script/schema_linking.py \
        -s data/CSpider/db_schema.json \
        -c data/CSpider/db_content.json \
        -o data/CSpider/match_values_test.json \
        data/CSpider/test.json

#################
训练集 NL2SQL
bash run.sh ./script/schema_linking.py \
        -s data/NL2SQL/db_schema.json \
        -c data/NL2SQL/db_content.json \
        -o data/NL2SQL/match_values_train.json \
        data/NL2SQL/train.json --is-train \
        --sql-format nl2sql

# 开发集
bash run.sh ./script/schema_linking.py \
        -s data/NL2SQL/db_schema.json \
        -c data/NL2SQL/db_content.json \
        -o data/NL2SQL/match_values_dev.json \
        data/NL2SQL/dev.json \
        --sql-format nl2sql

# 测试集 NL2SQL
bash run.sh ./script/schema_linking.py \
        -s data/NL2SQL/db_schema.json \
        -c data/NL2SQL/db_content.json \
        -o data/NL2SQL/match_values_test.json \
        data/NL2SQL/test.json \
        --sql-format nl2sql

五、获得模型输入
对 DuSQL 原始数据和Schema Linking的结果做处理，得到模型的输入，位于 data/DuSQL/preproc 目录下：
DuSQL
bash run.sh ./script/text2sql_main.py \
        --mode preproc \
        --config conf/text2sql_dusql.jsonnet \
        --data-root data/DuSQL/ \
        --is-cached false \
        --output data/DuSQL/preproc

CSpider
bash run.sh ./script/text2sql_main.py \
        --mode preproc \
        --config conf/text2sql_cspider.jsonnet \
        --data-root data/CSpider/ \
        --is-cached false \
        --output data/CSpider/preproc

NL2SQL
bash run.sh ./script/text2sql_main.py \
        --mode preproc \
        --config conf/text2sql_nl2sql.jsonnet \
        --data-root data/NL2SQL/ \
        --is-cached false \
        --output data/NL2SQL/preproc


七、预测
以预测 DuSQL 开发集为例，结果保存到 output/dusql_dev_infer_result.json。其中的 --init-model-param 参数请修改为真实的模型路径。
DuSQL
bash run.sh ./script/text2sql_main.py --mode infer \
         --config conf/text2sql_dusql.jsonnet \
         --data-root data/DuSQL/preproc \
         --test-set data/DuSQL/preproc/test.pkl \
         --init-model-param data/trained_model/dusql.pdparams \
         --output output/dusql.sql

CSpider /模型出问题，改成和DU一样的
bash run.sh ./script/text2sql_main.py --mode infer \
         --config conf/text2sql_cspider.jsonnet \
         --data-root data/CSpider/preproc \
         --test-set data/CSpider/preproc/test.pkl \
         --init-model-param data/trained_model/dusql.pdparams \
         --output output/cspider.sql

NL2SQL
bash run.sh ./script/text2sql_main.py --mode infer \
         --config conf/text2sql_nl2sql.jsonnet \
         --data-root data/NL2SQL/preproc \
         --test-set data/NL2SQL/preproc/test.pkl \
         --init-model-param data/trained_model/nl2sql.pdparams \
         --output output/nl2sql.sql