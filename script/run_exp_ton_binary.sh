conda activate iot
python ../src/tuning.py --name nf_ton_binary >> ../log/nf_ton_binary.txt 2>&1
python ../src/train.py --name nf_ton_binary # >> ../log/nf_ton_binary.txt 2>&1
# python ../src/predict.py --name nf_ton_binary --restore_model_name model=nf_ton_binary-epoch=000-val_loss=1.1295-val_acc=0.6256-val_f1=0.2502.ckpt