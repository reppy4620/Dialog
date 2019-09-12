import logging
import os
from shutil import copyfile

import unicodedata
import sentencepiece as spm

from pytorch_transformers.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': 'sp.model'}

SPIECE_UNDERLINE = u'▁'


class Tokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(self, vocab_file,
                 do_lower_case=False, remove_space=True, keep_accents=False,
                 bos_token="<s>", eos_token="</s>", unk_token="<unk>", sep_token="[SEP]",
                 pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]", **kwargs):
        """
        :param vocab_file: path of pre-trained SentencePiece model.
                           e.g. "sp.model" etc...
        :param do_lower_case: Handling Japanese, this is not effective i think.
        :param remove_space: same above
        :param keep_accents: same above
        :param bos_token: Begin Of Sentence.
        :param eos_token: End Of Sentence
        :param unk_token: Unknown token.
        :param sep_token: Separate
        :param pad_token: Padding
        :param cls_token: <cls> token is bos in pre-training
        :param mask_token: <mask> token is predicted token in pre-training
        """
        super(Tokenizer, self).__init__(bos_token=bos_token, eos_token=eos_token,
                                        unk_token=unk_token, sep_token=sep_token,
                                        pad_token=pad_token, cls_token=cls_token,
                                        mask_token=mask_token, **kwargs)

        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 3  # take into account special tokens

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return len(self.sp_model)

    @property
    def vocab(self):
        return [i for i in list(range(len(self.sp_model)))[6:]]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use XLNetTokenizer: https://github.com/google/sentencepiece"
                "pip install sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = ' '.join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if not self.keep_accents:
            outputs = unicodedata.normalize('NFKD', outputs)
            outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def _tokenize(self, text, sample=False):
        """ Tokenize a string """

        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)

        return pieces

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.sp_model.PieceToId(str(token))

    def _convert_id_to_token(self, index, return_unicode=True):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = ''.join(tokens).replace(SPIECE_UNDERLINE, '').strip()
        return out_string

    def add_special_tokens_single_sentence(self, token_ids):
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        """
        return [self.cls_token_id] + token_ids + [self.sep_token_id]

    def add_special_tokens_sentences_pair(self, token_ids_0, token_ids_1):
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A BERT sequence pair has the following format: [CLS] A [SEP] B [SEP]
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def save_vocabulary(self, save_directory):
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        out_vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES['vocab_file'])

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
