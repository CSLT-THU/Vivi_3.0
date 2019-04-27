# Vivi_3.0

## Quick Start

### Train a model
```
python train.py
```
Modify training parameters in the section [train] in the file config.ini. 
Modify model parameters in the model section in the file config.ini.
The models will be saved in 'ckpt/'.

### Generate a poem
```
python predict.py
```
Modify parameters in the section [predict] of the file config.ini. 
It supports 3 different types of input:
* Hidden head (Cangtou).
* Keywords.
* Test set file.

The priority is as the list, if you want to use keywords as a input, for example, leave the value of 'cangtou' blank.
```
cangtou = 
keywords = 夕阳-高峰-清泉-松叶-蝉噪
test_set = resource/dataset/testset.txt
```

### Tips
Since this is a developing version, we recommend you only use Seq2seq model.


