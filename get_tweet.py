import datetime
import json
import os
import re
import sys
import time
from socket import error as SocketError

import emoji
import neologdn
from requests_oauthlib import OAuth1Session

DIALOG_CNT = 3


class TweetData:
    def __init__(self, id, text, status_id):
        self.cnt = 0
        self.ids = [id]
        self.texts = [text]
        self.status_ids = [status_id]

    @property
    def last_status_id(self):
        return self.status_ids[-1]

    @property
    def last_text(self):
        return self.texts[-1]

    def add_data(self, id, text, status_id):
        self.ids.append(id)
        self.texts.append(text)
        self.status_ids.append(status_id)
        self.cnt += 1


def tweet_id2time(tweet_id):
    id_bin = bin(tweet_id >> 22)
    tweet_time = int(id_bin, 2)
    tweet_time += 1288834974657
    return tweet_time


def get_tweet(res, start_time):
    res_text = json.loads(res.text)
    url = 'https://api.twitter.com/1.1/statuses/lookup.json'

    max_tweet = start_time
    tweets = list()

    for tweet in res_text['statuses']:
        status_id = tweet['in_reply_to_status_id_str']
        tweet_id = tweet['id']
        if status_id is not None:
            tweet_time = tweet_id2time(tweet_id)
            if tweet_time <= start_time:
                continue

            if max_tweet < tweet_time:
                max_tweet = tweet_time

            res_sentence = tweet['text']
            if res_sentence[0:3] == 'RT ':
                continue
            res_sentence = screening(res_sentence)
            if res_sentence == '':
                continue
            tweets.append(TweetData(tweet_id, res_sentence, status_id))

    for cnt in range(DIALOG_CNT):
        id_list = [data.last_status_id for data in tweets]
        id_list = ','.join(id_list)
        unavailable_cnt = 0
        while True:
            try:
                req = session.get(url, params={'id': id_list, 'count': len(tweets)})
            except SocketError as e:
                print('ソケットエラー errno=', e.errno)
                if unavailable_cnt > 10:
                    raise

                wait_until_reset(time.mktime(datetime.datetime.now().timetuple()) + 30)
                unavailable_cnt += 1
                continue

            if req.status_code == 503:
                if unavailable_cnt > 10:
                    raise Exception('Twitter API error %d' % res.status_code)

                unavailable_cnt += 1
                print('Service Unavailable 503')
                wait_until_reset(time.mktime(datetime.datetime.now().timetuple()) + 30)
                continue

            if req.status_code == 200:
                req_text = json.loads(req.text)
                break
            else:
                raise Exception('Twitter API error %d' % res.status_code)

        idx_list = list()
        for i in range(0, len(tweets)):
            for j in range(0, len(req_text)):
                if req_text[j]['id_str'] == tweets[i].last_status_id:
                    idx_list.append((i, j))

        for i, j in idx_list:
            if req_text[j]['in_reply_to_status_id_str'] is not None:
                text = screening(req_text[j]['text'])
                if text == '' or text == '\n' or text == '\t' or text == '。':
                    continue
                tweets[i].add_data(req_text[j]['id_str'],
                                   screening(req_text[j]['text']),
                                   req_text[j]['in_reply_to_status_id_str'])
        tweets = [tweet for tweet in tweets if tweet.cnt == cnt + 1]
        if len(tweets) == 0:
            break

        max_tweet = max(max_tweet, start_time)
    return max_tweet, len(tweets), tweets


def screening(text):
    s = text

    if s[0:3] == "RT ":
        s = s.replace(s[0:3], "")
    while s.find("@") != -1:
        index_at = s.find("@")
        if s.find(" ") != -1:
            index_sp = s.find(" ", index_at)
            if index_sp != -1:
                s = s.replace(s[index_at:index_sp + 1], "")
            else:
                s = s.replace(s[index_at:], "")
        else:
            s = s.replace(s[index_at:], "")

    while s.find("\n") != -1:
        index_ret = s.find("\n")
        s = s.replace(s[index_ret], "")
    s = s.replace('\n', '')

    s = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", s)
    non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), '')
    s = s.translate(non_bmp_map)
    s = ''.join(c if c not in emoji.UNICODE_EMOJI else '' for c in s)

    s = re.sub('。+', '。', s)

    while s.find('#') != -1:
        index_hash = s.find('#')
        s = s[0:index_hash]

    s = neologdn.normalize(s, repeat=4)

    s = re.sub(r'[^、。!?ー〜1-9a-zA-Zぁ-んァ-ヶ亜-腕纊-黑一-鿕]', '', s)

    return s


def check_limit(session):
    unavailable_cnt = 0
    url = "https://api.twitter.com/1.1/application/rate_limit_status.json"

    while True:
        try:
            res = session.get(url)
        except SocketError as e:
            print('erron=', e.errno)
            print('ソケットエラー')
            if unavailable_cnt > 10:
                raise

            wait_until_reset(time.mktime(datetime.datetime.now().timetuple()) + 30)
            unavailable_cnt += 1
            continue

        if res.status_code == 503:
            # 503 : Service Unavailable
            if unavailable_cnt > 10:
                raise Exception('Twitter API error %d' % res.status_code)

            unavailable_cnt += 1
            print('Service Unavailable 503')
            wait_until_reset(time.mktime(datetime.datetime.now().timetuple()) + 30)
            continue

        unavailable_cnt = 0

        if res.status_code != 200:
            raise Exception('Twitter API error %d' % res.status_code)

        remaining_search, remaining_user, remaining_limit, reset = get_limit_context(json.loads(res.text))
        if remaining_search <= 1 or remaining_user <= 1 or remaining_limit <= 1:
            wait_until_reset(reset + 30)
        else:
            break

    sec = reset - time.mktime(datetime.datetime.now().timetuple())
    print(remaining_search, remaining_user, remaining_limit, sec)
    return reset


def wait_until_reset(reset):
    seconds = reset - time.mktime(datetime.datetime.now().timetuple())
    seconds = max(seconds, 0)
    print('\n     =====================')
    print('     == waiting %d sec ==' % seconds)
    print('     =====================')
    sys.stdout.flush()
    time.sleep(seconds + 10)


def get_limit_context(res_text):
    remaining_search = res_text['resources']['search']['/search/tweets']['remaining']
    reset1 = res_text['resources']['search']['/search/tweets']['reset']
    remaining_user = res_text['resources']['statuses']['/statuses/lookup']['remaining']
    reset2 = res_text['resources']['statuses']['/statuses/lookup']['reset']
    remaining_limit = res_text['resources']['application']['/application/rate_limit_status']['remaining']
    reset3 = res_text['resources']['application']['/application/rate_limit_status']['reset']

    return int(remaining_search), int(remaining_user), int(remaining_limit), max(int(reset1), int(reset2), int(reset3))


if __name__ == '__main__':

    CK = 'xxx'  # Consumer Key
    CS = 'xxx'  # Consumer Secret
    AT = 'xx-xx'  # Access Token
    AS = 'xxx'  # Accesss Token Secert

    args = sys.argv

    try:
        DIALOG_CNT = int(args[2]) - 1
    except:
        DIALOG_CNT = 1

    session = OAuth1Session(CK, CS, AT, AS)

    total = -1
    total_count = 0
    cnt = 0
    unavailableCnt = 0
    save_dir = './data'
    url = 'https://api.twitter.com/1.1/search/tweets.json'
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    start_time = 1288834974657
    while True:
        reset = check_limit(session)
        get_time = time.mktime(datetime.datetime.now().timetuple())
        try:
            res = session.get(url, params={'q': args[1], 'count': 100})
        except SocketError as e:
            print('ソケットエラー errno=', e.errno)
            if unavailableCnt > 10:
                raise

            wait_until_reset(time.mktime(datetime.datetime.now().timetuple()) + 30)
            unavailableCnt += 1
            continue

        if res.status_code == 503:
            if unavailableCnt > 10:
                raise Exception('Twitter API error %d' % res.status_code)

            unavailableCnt += 1
            print('Service Unavailable 503')
            wait_until_reset(time.mktime(datetime.datetime.now().timetuple()) + 30)
            continue

        unavailableCnt = 0

        if res.status_code != 200:
            raise Exception('Twitter API error %d' % res.status_code)

        start_time, count, tweets = get_tweet(res, start_time)

        fn = f'{save_dir}/tweet_data_{args[1]}_{DIALOG_CNT}.txt'

        with open(fn, 'a', encoding='utf-8') as f:
            for i in range(0, len(tweets)):
                for t in tweets[i].texts[::-1]:
                    f.write(str(t) + '\n')
                f.write('\n')
        f.close()

        total_count += count
        print('total_count=', total_count, 'start_time=', start_time)

        current_time = time.mktime(datetime.datetime.now().timetuple())
        if current_time - get_time < 2:
            wait_until_reset(time.mktime(datetime.datetime.now().timetuple()) + 2)
