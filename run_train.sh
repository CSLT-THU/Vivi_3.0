#! /bin/bash
python train.py --dataset 'resource/dataset/poem_1031k_4x.txt' --epochs 5 --ckpt_path '' --val_rate 0.1 --batch_size 80 --teacher_forcing_ratio 0.8 --model_name 'Seq2seq_7' --note ''