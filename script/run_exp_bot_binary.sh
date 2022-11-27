conda activate iot
python ../src/tuning.py --name nf_bot_binary >> ../log/nf_bot_binary.txt 2>&1
python ../src/train.py --name nf_bot_binary # >> ../log/nf_bot_binary.txt 2>&1
# python ../src/predict.py --name nf_bot_binary --restore_model_name ...
