# Vivi_3.0
## Intro
This is the implementation for ["A Rhythm Model for Chinese Poetry Generation"](http://cslt.riit.tsinghua.edu.cn/mediawiki/images/f/f4/Technical_report_-_poetry_generation.pdf).
Some examples: (keywords ==== poem)
枯藤-僧寺-霜-雨后 ==== 藤根枯树老参开/寺里僧钟午夜来/露滴悬崖挂竹杖/苔阶盘柏古梧栽
冷-江风-烟雨-香 ==== 江水流沙草自长/风烟吹散落花黄/雨声夜里听云起/不觉清寒入梦乡
梦醒-素-短-玉骨 ==== 醒来梦里醉红尘/枕短长空夜色深/不见玉人花底处/魂销骨瘦有香痕
烟雨-平-荆山-越 ==== 雨洗烟花江北岸/青山绿水柳条平/楚腰翠袖荆宫里/锦绣流连蜀苑中
林-笛声-雨帘-墨 ==== 高堂小阁上层林/角声吹笛动人心/窗轩不掩疏帘后/弄笔书生写墨痕
蓬莱-墨-八骏-万里 ==== 蓬莱丈地无人羡/墨笔挥毫画半端/大道龙宫争作剑/金河赤浪战天关
千载-焦-一盏-月 ==== 秋风几度江边载/此地凄凉不可忧/漫饮杯壶金玉酒/更将月下对天楼

## Quick Start

### Train a model

#### 1. Obtain dataset
Due to the file size limit, the dataset we used cannot be uploaded. 
If you are a CSLT user, obtain the dataset from the server: `/work4/liuyibo/pycharm/Poetry Generation/poem_vivi_3.6/resource/dataset/poem_1031k_theme.txt`
Before training, you can split the dataset into train set and test set: first modify the file dirs in `resource/dataset/split_datase.py`, then run it.

#### 2. Run training
```
./run_train.sh
```
#### 3. Checkpoints
Checkpoints will be saved to `ckpt/` every epoch.
#### 4. Losses
Losses of all trainings are recorded in `loss/loss_log`.
Losses of the last traning are saved to `loss/loss.npy`. 
Run `plot_loss.py` to visualize the losses of the last training. It will be saved as a jpg file to `loss/`.

### Generate a poem
#### 1. Set prediction parameters in the config file
In the section [predict] of the file `config/config.ini`. 
It supports 4 different input types:
* Hidden head (Cangtou): get one poem with hidden head. Must be 4 characters.
* Keywords: get one poem with the keywords. It can be a set of keywords or a sentence. If using multiple keywords, seperate the words with '-', 
e.g. `夕阳-高峰-清泉-松叶-蝉噪`  
* Test set: get many poems with the keywords of every line.
* Evaluation set: get many poems and compare them with the target poems. The form is the same as training set. Each line looks like:
```
雨 - 江 南 - 水 - 荷 花==十 年 一 觉 江 南 雨	谁 是 江 南 意 中 人	竹 筏 清 歌 山 映 水	荷 花 香 远 亦 天 真
```
Among the parameters,  
- `cangtou`, `keywords`, `test_set`, `eval_set` are mutually exclusive, leave other parameters blank when using one of them. 
- `model` and  `ckpt_path` are required. 
- `use_planning` is related to planning mechanism, which extracts/expands 4 keywords from the input query. 
- When using evaluation set as input, setting `bleu_eval` to `True` can give a bleu score. 
- `poem_type` can be set as either `poem7` or `poem5`, which represents the sentence length.

#### 2. Run prediction
```
./run_predict.sh
```
#### 3. Results
Results will be saved to `result/`. The results generated by the same model are saved to the same file.


## Add a new model
All models are defined in `models/`, add your new model to this dir, the folder name should be the same as the model name.
Also add a config file to the dir `config/config_Model_name.ini` for model parameters.
The model folder should at least contain the following files:  

Model_name                                        
├── Model_name.py  
├── PoetryData.py  
└── Optim.py  

#### Model_name.py
File name should be the same as model name. The following class and methods are required.
```
class Model_name(nn.Module):
    def __init__(self, model_param):
        super(Model_name, self).__init__()
        ...
    
    def forward(self, batch_size, data, criterion,
                teacher_forcing_ratio):
        ...
        return loss
    
    def predict(self, data, cangtou, predict_param):
        ...
        return decoded_words
```  
- `model_param` is the the dictionary of parameters to initiate the model, which are definded in the model config file mentioned above.   
Note that all values are in `String` type, convert to other types when needed.
- `data` is one batch data of traning data generated from DataLoader.
- `criterion` is defined as `nn.CrossEntropyLoss()` in `train.py`. Principlly it's unnecessary to change this.
- `predict_param` represents the parameters to initiate the model, which are definded in the [predict] section of config file `config/config.ini`.
- `decoded words` should be the list of all characters in the poem, without any special sign between sentences. 

#### PoetryData.py
The following class and methods are required.
```
class PoetryData(Dataset):
    def __init__(self, data, src_max_len, tgt_max_len, test=False):
        ...
        
    def __len__(self):
        ...

    def __getitem__(self, index):
        ...
```
- `src_max_len` and `tgt_max_len` are required parameters in model config file.
- `test` default value is set to False. This parameter allowes the return of \_\_getitem\_\_ method to be different in traning and prediction.

#### Optim.py
```
def get_optimizer(model, model_param):
    ...
```
Return an optimizer for training.


## Use Planning
### Description
Detailed desciption of planning can be found in this repository https://github.com/CSLT-THU/Poetry-Planning-for-ViVi_3.0.   
General procedure of planning can be describe as:  
#### 1. Train the planner  
Segment the sentences in the corpus (use sxhy(诗学含英) as lexicon), then apply TextRank to the segmented words and get the ranking of all words.
Select keywords for each poem (one keyword per sentence) according to the word ranking. 
Train a word vector with the keywords. This word vector is the planning model. 
Note that this word vector is different from the word vector used in poem generation, which is __chatacter-based__, while this is __word-based__. 
#### 2. Use the planner  
First extract keywords from the query. When keywords are not enough, expand keywords by randomly picking a word which has small word vector distance with an existing keyword.  
   
In this project, planning package is utilized in 2 ways:   
1. Generate keywords from input query for prediction    
2. Extract keywords from a poem for creating dataset   

### Train the Planner
In order to use the planner, you have to train the planner according to the following steps:
1. Add corpus file(s) to `planning/raw/`. Note that only the poems with 7 characters per sentence count. One line in the corpus file should be like:
```
海 滨 清 洗 碧 天 空	地 近 扶 桑 东 复 东	金 镜 曜 辉 云 气 散	茅 檐 先 被 一 轮 红
```
2. Add corpus file names to `char_dict.py` and `poems.py`:
```
_corpus_list = ['poem_1031k.txt']
```
3. Download modern word2vec model from https://github.com/Embedding/Chinese-Word-Vectors 
(We recommend SGNS Baidu Encyclopedia Word + Character + Ngram 300d. Download link is [here](https://pan.baidu.com/s/1Gndr0fReIq_oJ3R34CxlPg))
Save it to dir `planning/save/`. Add the file name to `plan.py`:
```
_modern_model_path = os.path.join(save_dir, 'sgns.baidubaike.bigram-char')
```
4. Run `plan.py`
5. Intermediate files:
- `data/poem.txt` dataset after preprocessing from the raw daraset.
- `data/char_dict.txt` character dictionary. All characters in corpus.
- `data/plan_data.txt` keywords extracted from corpus. (4 keywords per poem)
- `data/plan_history.txt` keywords and poems in the corpus. __This can be used as training dataset for poem generation model.__
- `data/wordrank.txt` Ranking of words extracted from corpus.
- `save/ancient_model_5.bin` Ancient word vector, which is the __essence of the planner__.   
All these intermediate files inclueded in this repository are created with the corpus `poem_1031k`. If you want to train on your own corpus, delete these imtermediate files first.

## Evaluation
### Automatic evaluation
- `yun_rate`
- `lv_rate`
- `lm`
1. Combine the results of different models by running `combine_result.py` and get their `yun_rate`, `lv_rate`, and `lm`.
2. Plot by `plot_2d.py`, `plot_3d.py`, `plot_loss.py`.
### Human evaluation
Gnerate questionnaires by `scoring/scoring.py`. Analyse the questionnaire results by `scoring/scoring_result.py`.

## Tips
### For CSLT users
You can run this project directly on the server without any preparation at this dir: `/work4/liuyibo/pycharm/Poetry Generation/poem_vivi_3.6/`
### For All
Since this is an ongoing project, the model `Transformer` and `Seq2seq_new` provided do not work yet. Bleu score is not available. Poem type only supports `poem7`. 
## File Structure  
├── ckpt                                        
│   ├── 04-27_Seq2seq_epoch=7_loss=113.6.pkl  
│   ├── 05-05_Seq2seq_epoch=5_loss=143.6.pkl  
│   ├── 05-14_Seq2seq_epoch=4_loss=130.7.pkl  
│   └── 05-14_Seq2seq_epoch=6_loss=130.8.pkl  
├── config  
│   ├── config.ini  
│   ├── config_Seq2seq.ini  
│   ├── config_Seq2seq_new.ini  
│   └── config_Transformer.ini  
├── constrains.py  
├── data_utils.py  
├── get_feature.py  
├── loss  
│   ├── 58k_lr=1_batchsize=80_epoch=7.jpg  
│   ├── loss_log  
│   ├── loss_logs.py  
│   ├── loss.npy  
│   └── plot_loss.py  
├── models  
│   ├── Seq2seq  
│   │   ├── Optim.py  
│   │   ├── PoetryData.py  
│   │   ├── RNN.py  
│   │   └── Seq2seq.py  
│   ├── Seq2seq_bak  
│   │   ├── Optim.py  
│   │   ├── PoetryData.py  
│   │   ├── RNN.py  
│   │   └── Seq2seq.py  
│   ├── Seq2seq_new  
│   │   ├── Optim.py  
│   │   ├── PoetryData.py  
│   │   ├── RNN.py  
│   │   └── Seq2seq_new.py  
│   └── Transformer  
│       ├── Beam.py  
│       ├── Constants.py  
│       ├── \_\_init\_\_.py  
│       ├── Layers.py  
│       ├── Models.py  
│       ├── Modules.py  
│       ├── Optim.py  
│       ├── PoetryData.py  
│       ├── SubLayers.py  
│       ├── Transformer.py  
│       └── Translator.py  
├── planning  
│   ├── char_dict.py  
│   ├── data  
│   │   ├── char_dict.txt  
│   │   ├── plan_data.txt  
│   │   ├── plan_history.txt  
│   │   ├── poem.txt  
│   │   ├── sxhy_dict.txt  
│   │   └── wordrank.txt  
│   ├── data_utils.py  
│   ├── \_\_init\_\_.py  
│   ├── paths.py  
│   ├── plan.py  
│   ├── poems.py  
│   ├── rank_words.py  
│   ├── raw   
│   │   ├── pinyin.txt  
│   │   ├── poem_1031k.txt (__not included__)      
│   │   ├── shixuehanying.txt  
│   │   ├── stopwords.txt  
│   ├── save  
│   │   ├── ancient_model_5.bin  
│   │   └── sgns.baidubaike.bigram-char (__not included__)    
│   └── segment.py  
├── predict.py  
├── resource  
│   ├── dataset  
│   │   ├── poem_1031k_theme.txt (__not included__)     
│   │   ├── split_dataset.py  
│   │   ├── test_1031k.txt  (__not included__)    
│   │   ├── train_1031k.txt (__not included__)     
│   │   └── testset.txt  
│   ├── word_dict.json  
│   └── word_emb.json  
├── result  
│   └── result_05-14_Seq2seq_epoch=6_loss=130.8.txt  
├── train.py  
└── word_emb.py  



