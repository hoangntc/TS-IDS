# conda activate iot

# python ../src/ablation.py --name nf_bot_binary
# python ../src/ablation.py --name nf_bot_multi
# python ../src/ablation.py --name nf_ton_binary
# python ../src/ablation.py --name nf_ton_multi

python ../src/main.py --name nf_bot_binary --stage all --n_folds 5
python ../src/main.py --name nf_ton_binary --stage all --n_folds 5

python ../src/main.py --name nf_bot_multi --stage all --n_folds 5
python ../src/main.py --name nf_ton_multi --stage all --n_folds 5

python ../src/main.py --name nf_unsw_binary --stage all --n_folds 5
python ../src/main.py --name nf_unsw_multi --stage all --n_folds 5

python ../src/main.py --name nf_cse_binary --stage all --n_folds 5
python ../src/main.py --name nf_cse_multi --stage all --n_folds 5