import glob
import re

from tqdm import tqdm
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer


def preprocess(s):
    # remove words with brackets
    s = re.sub(r'\(.+?\)', '', s)
    s = re.sub(r'「.+?」', '', s)
    s = re.sub(r'\[.+?\]', '', s)
    s = re.sub(r'【.+?】', '', s)
    return s


if __name__ == '__main__':
    min_size = 2
    max_size = 22
    num_use_uttr = 2
    use = 0
    not_use = 0
    tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese-whole-word-masking')
    file_name = f'training_data'
    with open(f'./data/{file_name}.txt', 'a', encoding='utf-8') as ff:
        files = glob.glob('./data/tweet_data_*.txt')
        num_files = len(files)
        for f_num, fn in enumerate(files, start=1):
            num_uttr = int(fn.split('/')[-1].split('.')[0].split('_')[-1]) + 1
            with open(fn, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i in tqdm(range(0, len(lines) - (num_uttr - 1), num_uttr + 1),
                              desc=f'File({fn.split("/")[-1]}): {f_num}/{num_files}'):
                    sentences = list(lines[i:i + num_uttr])
                    # Hard-Coding Filter
                    if any(map(lambda x: len(x) <= min_size, sentences)):
                        continue
                    if any(map(lambda x: 'ニュース' in x, sentences)):
                        continue
                    if any(map(lambda x: x.startswith('。'), sentences)):
                        continue
                    utterances = list(map(tokenizer.encode, sentences))
                    if all(map(lambda x: len(x) <= max_size, utterances)):
                        if num_uttr > num_use_uttr:
                            for j in range(num_uttr - (num_use_uttr - 1)):
                                for s in sentences[j:j + num_use_uttr]:
                                    ff.write(f'{s}')
                                ff.write('\n')
                                use += 1
                        else:
                            for s in sentences:
                                ff.write(f'{s}')
                            ff.write('\n')
                            use += 1
                    else:
                        not_use += 1
    print(f'Adapted: {use}/{use + not_use} ({use / (use + not_use) * 100}%)')
