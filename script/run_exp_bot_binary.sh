conda activate iot
python ../src/tuning.py --name nf_bot_multi >> ../log/nf_bot_binary.txt 2>&1
python ../src/train.py --name nf_bot_multi # >> ../log/nf_bot_binary.txt 2>&1
python ../src/predict.py --name nf_bot_multi --restore_model_name model=gat_nf_bot_multi-epoch=104-train_loss=0.2461-val_loss=0.2955-val_acc=0.8403-val_macro_f1=0.5689-val_micro_f1=0.8403.ckpt
