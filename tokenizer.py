import torch
from transformers import AutoTokenizer, PreTrainedTokenizer


class Tokenizer(PreTrainedTokenizer):

    def __init__(self, model_name, **kwargs):
        super().__init__(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, x, **kwargs):
        x = self._tokenizer.encode(x)
        return x

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.view(-1).tolist()
        s = self._tokenizer.convert_ids_to_tokens(token_ids)
        s = self._tokenizer.convert_tokens_to_string(s)
        s = s.replace('#', '').replace(' ', '')
        return s

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def cls_token_id(self) -> int:
        return self._tokenizer.cls_token_id

    @property
    def sep_token_id(self) -> int:
        return self._tokenizer.sep_token_id
