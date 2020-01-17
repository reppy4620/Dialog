import torch
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer


class Tokenizer(BertJapaneseTokenizer):

    def convert(self, x):
        return self.convert_tokens_to_ids(self.tokenize(x))

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.view(-1).tolist()
        s = ''.join([self.ids_to_tokens[x] for x in token_ids])
        s = s.replace('#', '').replace(' ', '')
        return s
