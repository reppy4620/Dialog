import argparse
import pathlib
import re

from tqdm import tqdm
from transformers import AutoTokenizer

from utils import get_config


def preprocess(s):
    # remove words with brackets
    s = re.sub(r'\(.+?\)', '', s)
    s = re.sub(r'「.+?」', '', s)
    s = re.sub(r'\[.+?\]', '', s)
    s = re.sub(r'【.+?】', '', s)
    return s


def sentences_filter(sentences, config):
    # Hard-Coding Filter
    return any([len(x) <= config.min_size for x in sentences]) or \
           any(['ニュース' in x for x in sentences]) or \
           any([x.startswith('。') for x in sentences])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str)
    parser.add_argument('-i', '--data_dir', type=str)
    parser.add_argument('-o', '--output_dir', type=str, default='data')
    parser.add_argument('-f', '--file_name', type=str, default='training_data')
    args = parser.parse_args()

    config = get_config(args.config_file)

    use = 0
    not_use = 0

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    data_dir = pathlib.Path(config.data_dir)
    output_dir = pathlib.Path(config.output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    train_f = open(f'{config.output_dir}/{config.file_name}.txt', 'a', encoding='utf-8')
    files = list(data_dir.glob('tweet_data_*.txt'))
    num_files = len(files)
    for f_num, fn in enumerate(files, start=1):
        num_uttr = int(fn.split('/')[-1].split('.')[0].split('_')[-1]) + 1
        with open(fn, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i in tqdm(range(0, len(lines) - (num_uttr - 1), num_uttr + 1),
                      desc=f'File({fn.split("/")[-1]}): {f_num}/{num_files}'):
            sentences = list(lines[i:i + num_uttr])
            if sentences_filter(sentences, config):
                continue
            ids = [tokenizer.encode(x, ) for x in sentences]
            if all([len(x) <= config.max_size for x in ids]):
                if num_uttr > config.num_use_uttr:
                    for j in range(num_uttr - (config.num_use_uttr - 1)):
                        for s in sentences[j:j + config.num_use_uttr]:
                            train_f.write(f'{s}')
                        train_f.write('\n')
                        use += 1
                else:
                    for s in sentences:
                        train_f.write(f'{s}')
                    train_f.write('\n')
                    use += 1
            else:
                not_use += 1
    train_f.close()
    print(f'Adapted: {use}/{use + not_use} ({use / (use + not_use) * 100}%)')
