#! /bin/bash
python predict.py --model_name 'Seq2seq_5' --ckpt_path 'ckpt/06-05_Seq2seq_5_epoch=1_loss=146.1.pkl' --eval_set 'resource/dataset/test_1031k.txt' 