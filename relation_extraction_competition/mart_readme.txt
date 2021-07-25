python run_duie.py --device gpu --seed 42 --do_train --data_path ./data --max_seq_length 128 --batch_size 8 --num_train_epochs 12 --learning_rate 2e-5 --output_dir ./checkpoints

--device cpu
--seed 42
--do_train
--data_path ./data
--max_seq_length 128
--batch_size 8
--num_train_epochs 12
--learning_rate 2e-5
--warmup_ratio 0.06
--output_dir ./checkpoints


python re_official_evaluation.py --golden_file=dev_data.json  --predict_file=predicitons.json.zip [--alias_file alias_dict]

python re_official_evaluation.py --golden_file ./data/dev_data.json --predict_file /data/predicitons.json.zip [--alias_file alias_dict]

python re_official_evaluation.py --golden_file=dev_data.json  --predict_file=predicitons.json.zip