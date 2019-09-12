# Dialog
Dialog is japanese chatbot project.  
Seq2Seq model has BERT Encoder and Transformer Decoder.

Note that you have to prepare the conversation data,
download pretrained model from [this](https://github.com/yoheikikuta/bert-japanese)
and change path in config.py
if you wanna train using this code.  

## Usage
Install packages.  
```bash
$ pip install -r requirements.txt
```
requirements.txt includes redundant packages.  
Maybe, Needed packages are

- pytorch
- pytorch-transformers
- sentencepiece
- tqdm

Prepare conversation data.  
1.I used twitter data that is scraped using [this](https://qiita.com/gacky01/items/89c6c626848417391438)

2.Normalized sentence(e.g. remove punctuations...) and encode sequence to ids using SentencePiece.

3.Convert to pkl from sentence pair  
```
# architecture
# q is input sentence, a is target sentence.
[(q1, a1), (q2, a2), (q3, a3), ...]
```

4.Change path in config.py

5.start training

## Architecture
- Encoder: BERT  
- Decoder: Vanilla Transformer Decoder

- Loss: KLDivLoss with LabelSmoothing
- Optimizer: Adam with warm-up

- Tokenizer: SentencePiece(trained wiki-japanese)

If you wanna more information about architecture of BERT or Transformer, please refer to the following blog.

- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [PyTorch-Transformers/modeling_bert](https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/modeling_bert.py)
