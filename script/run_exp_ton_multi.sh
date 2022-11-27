conda activate iot
python ../src/tuning.py --name nf_ton_multi >> ../log/nf_ton_multi.txt 2>&1
python ../src/train.py --name nf_ton_multi # >> ../log/nf_ton_multi.txt 2>&1
python ../src/predict.py --name nf_ton_multi --restore_model_name model=gat_ton_multi-epoch=000-val_loss=1.1295-val_acc=0.6256-val_f1=0.2502.ckpt