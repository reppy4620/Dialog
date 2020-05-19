# Dialog
**Dialog** is japanese chatbot project.  
Used architecture in this project is EncoderDecoder model that has BERT Encoder and Transformer Decoder.

Note that this encoder-decoder model cannot use information of conversation history.  

[Article](https://qiita.com/reppy4620/items/e4305f22cd8f6962e00a) written in Japanese.

# Contents
1. [Result](#result)
2. [PreTrained Model](#pretrained-model)
2. [Usage](#usage)
    1. [Install Packages](#install-packages)
    2. [Train](#train)
    3. [Evaluate](#evaluate)
3. [Architecture](#architecture)

# Result
2epochs

![Result](./result/result.png)

This model has still contain the problem about dull response.  
To solve this problem i'm researching now.  

Then I found the paper tackled this problem.

[Another Diversity-Promoting Objective Function for Neural Dialogue Generation](https://arxiv.org/abs/1811.08100)

Authors belong to the Nara Institute of Science and Technology a.k.a NAIST.  
They propose the new objective function of Neural dialogue generation.  
I hope that this method can help me to solve that problem.

# Pretrained Model
- Pretrained model : ckpt.pth
- Training data : training_data.txt or train_data.pkl

in [google drive](https://drive.google.com/open?id=1wYrUQHb4Wg2T8ZvCleIBcGu7PTaFw6VO).

# Usage
### Install packages.
Needed packages are

- pytorch
- transformers
- tqdm
- MeCab(To use transformers.tokenization_bert_japanese.BertJapaneseTokenizer)
- neologdn
- emoji

If occur errors because of the packages, please install missing packages.

Example if you use conda.

```bash
# create new environment
# version >= 3.6
$ conda create -n dialog python=3.7

# activate new environment
$ activate dialog

# install pytorch
$ conda install pytorch torchvision cudatoolkit={YOUR_VERSION} -c pytorch

# install rest of depending package except for MeCab
$ pip install transformers tqdm neologdn emoji

##### Installed MeCab #####
### Ubuntu ###
$ pip install mecab-python3

### Windows ###
# and added "path/to/MeCab/bin" to system envrionment variable
$ pip install mecab-python-windows

##### Not Installed MeCab #####
# install Mecab in accordance with your OS.
# method described in below is one of the way,
# so you can use your way if you'll be able to use transformers.BertJapaneseTokenizer.
### Ubuntu ###
# if you've not installed MeCab, please execute following comannds.
$ apt install aptitude
$ aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y
$ pip install mecab-python3

### Windows ###
# if you've already installed MeCab from https://github.com/ikegami-yukino/mecab/releases/tag/v0.996
# and add "path/to/Mecab/bin" to system environment variable, run the following command.
$ pip install mecab-python-windows
```

## Train

### Prepare conversation data.
1. Download training data from [google drive](https://drive.google.com/open?id=1wYrUQHb4Wg2T8ZvCleIBcGu7PTaFw6VO)  
- train_data.pkl

2. Change path in config.py
```python
# in config.py, line 24
# default value is './data'
data_dir = 'path/to/dir_contains_training_data'
```

### Excecute
if you're ready to start training, run the main script.
```bash
$ python main.py
```

## Evaluate
-  Download pretrained weight from [google drive](https://drive.google.com/open?id=1wYrUQHb4Wg2T8ZvCleIBcGu7PTaFw6VO)
-  Change a path of pre-trained model in config.py
```python
# in config.py, line 24
# default value is './data'
data_dir = 'path/to/dir_contains_pretrained'
```
- run eval.py
```shell script
$ python run_eval.py
```


## Usage of get_tweet.py
If you wanna get more conversation data, please use get_tweet.py

Note that you have to need to change consumer_key and access_token
in order to use this script.

And then, execute following commands.
```bash
# usage
$ python get_tweet.py "query" "Num of continuous utterances"

# Example
# This works until occurs errors 
# and makes a file named "tweet_data_私は_5.txt" in "./data"
$ python get_tweet.py 私は 5
```
If you execute the Example command, script start to collect consecutive 5 sentences if last sentence contains "私は".

However you set 3 or more number to "continuous utterances", make_training_data.py automatically create pair of utterances.

Then execute following command.
```bash
$ python make_training_data.py
```
This script makes training data using './data/tweet_data_*.txt', just like the name.


# Architecture
- Encoder: [BERT](https://arxiv.org/abs/1810.04805)  
- Decoder: [Vanilla Transformer's Decoder](https://arxiv.org/abs/1706.03762)

- Loss: CrossEntropy
- Optimizer: AdamW

- Tokenizer: BertJapaneseTokenizer


If you want more information about architecture of BERT or Transformer, please refer to the following article.

- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [transformers](https://github.com/huggingface/transformers)
