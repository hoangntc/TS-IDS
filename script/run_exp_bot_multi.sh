conda activate iot
python ../src/tuning.py --name nf_bot_multi >> ../log/nf_bot_multi.txt 2>&1
python ../src/train.py --name nf_bot_multi # >> ../log/nf_bot_multi.txt 2>&1
# python ../src/predict.py --name nf_bot_multi --restore_model_name model=nf_bot_multi-epoch=000-val_loss=1.1295-val_acc=0.6256-val_f1=0.2502.ckpt