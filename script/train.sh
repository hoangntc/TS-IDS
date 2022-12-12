conda activate iot
# python ../src/train.py --name nf_bot_binary >> ../log/nf_bot_binary.txt 2>&1
# python ../src/train.py --name nf_bot_multi >> ../log/nf_bot_multi.txt 2>&1
# python ../src/train.py --name nf_ton_binary >> ../log/nf_ton_binary.txt 2>&1
# python ../src/train.py --name nf_ton_multi >> ../log/nf_ton_multi.txt 2>&1

python ../src/main.py --name nf_bot_binary
python ../src/main.py --name nf_bot_multi
# python ../src/main.py --name nf_ton_binary
# python ../src/main.py --name nf_ton_multi