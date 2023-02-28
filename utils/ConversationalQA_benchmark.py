import json
import os
import sys
sys.path.append(os.path.dirname(__file__))
from typing import Sequence, Tuple
import re
import six
import csv
import unicodedata
from typing import Iterator, Tuple, List

def ngrams(sentence: str, n: int) -> Iterator[Tuple]:
    """Yield ngrams.

    Example:
        >>> sentence = 'i love u'
        >>> list(ngrams(sentence, 2))
        >>> [('i', 'love'), ('love', 'u')]
    """
    sentence = tokenize(sentence)  # convert to list consisting of tokens
    sentence = iter(sentence)  # make sentence iterable
    history = []
    while n > 1:
        # PEP 479, prevent RuntimeError from being raised when StopIteration bubbles out of generator
        try:
            next_item = next(sentence)
        except StopIteration:
            # no more data, terminate the generator
            return
        history.append(next_item)
        n -= 1
    for item in sentence:
        history.append(item)
        yield tuple(history)
        del history[0]


def _normalize(text: str):
    text = text.lower()
    dirty_char = [
        u":", u"_", u"`", u"，", u"。", u"：", u"？", u"！", u"(", u")", u"“", u"”",
        u"；", u"’", u"《", u"》", u"……", u"·", u"、", u",", u"「", u"」", u"（",
        u"）", u"－", u"～", u"『", u"』", "|",
    ]
    out_segs = []
    for char in text:
        if char in dirty_char:
            continue
        out_segs.append(char)
    return "".join(out_segs)


def tokenize(text: str) -> List[str]:
    """Clean and Tokenize input text into a list of tokens.

    Args:
        text: A text blob to tokenize.

    Returns:
        A list of string tokens extracted from input text.
    """
    normalized_text = _normalize(text)
    output = ""
    for char in normalized_text:
        # exclude punctuations
        if is_punctuation(char):
            continue
        if is_chinese_char(
                ord(char)) or is_whitespace(char) or is_control(char):
            output += " " + char + " "
        else:
            # for non-chinese characters
            # only english characters and numbers are permitted
            char = re.sub(r"[^a-z0-9]+", "", six.ensure_str(char))
            if char == "":
                continue
            output += char
    return output.split()


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
        return True

    return False


def is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64)
            or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

class RougeScorer:
    """Calculate the `ROUGE_N` and `ROUGE_L`.

    Example:
        >>> scorer = RougeScorer(rouge_types=['rouge1', 'rougeL'])
        >>> candidates = ['I love eating apples.', 'eat apples I like to.']
        >>> reference = 'I love eating apples.'
        >>> for cand in candidates:
                scorer.add_instance(cand, reference)
        >>> print(scorer.score())
    """

    def __init__(self, rouge_types: Sequence[str]):
        """Initialize a new RougeScorer.

        The element of rouge_type should be the following:
        - rougen: rouge1, rouge2...
        - rougeL
        """
        self.rouge_types = rouge_types
        self.metric = {rouge_type: [] for rouge_type in rouge_types}

    def add_instance(self, candidate: str, reference: str):
        """Add a new instance and keep scores in record.

        Must process this before `score` function."""
        temp_rouge = {}
        for rouge_type in self.rouge_types:
            if rouge_type == "rougeL":
                score = _score_lcs(candidate, reference)
            elif rouge_type == "EM":
                score = _score_em(candidate, reference)
            else:
                n = int(rouge_type[5:])

                c_grams = list(ngrams(candidate, n))
                r_grams = list(ngrams(reference, n))

                score = _score_rouge_n(c_grams, r_grams)
            temp_rouge[rouge_type] = score

        for key in self.metric:
            self.metric[key].append(temp_rouge[key])
        return temp_rouge

    def score(self) -> float:
        """Return the final mean scores."""
        result = {}
        for key in self.metric:
            score_lis = self.metric[key]
            result[key] = sum(score_lis) / len(score_lis)

        return result


def _score_rouge_n(c_grams: Sequence[Tuple], r_grams: Sequence[Tuple]) -> float:
    """Calculate the `ROUGE_N whose beta is 1 (i.e. F1).`"""
    overlapping_count = 0
    for ngram in c_grams:
        if ngram in r_grams:
            overlapping_count += 1

    candidate_count = len(c_grams)
    reference_count = len(r_grams)

    precision = overlapping_count / candidate_count
    recall = overlapping_count / reference_count
    # in case precision = recall = 0
    f1 = 2 * (precision * recall / (precision + recall + 1e-8))
    return f1
    # return {"f1": f1, "p": precision, "r": recall}


def _score_lcs(candidate: str, reference: str) -> float:
    """Calculate the `ROUGE_L` whose `beta` is 1 (i.e. F1).

    Example:
        >>> from rouge import _score_lcs
        >>> x = 'i love u so much'
        >>> y = 'i hate u'
        >>> _score_lcs(x, y)
        >>> 0.5
    """
    # convert to list consisting of tokens
    candidate = tokenize(candidate)
    reference = tokenize(reference)
    table = _lcs(candidate, reference)
    m, n = len(candidate), len(reference)
    lcs = table[n, m]

    if len(candidate) == 0:
        return 0
    precision = lcs / len(candidate)
    recall = lcs / len(reference)
    # in case precision = recall = 0
    f1 = 2 * (precision * recall / (precision + recall + 1e-8))
    # return {"f1": f1, "p": precision, "r": recall}
    return f1


def _score_em(candidate: str, reference: str) -> float:
    """Calculate the `Exact Match Score`.

    Example:
        >>> from rouge import _score_em
        >>> x = 'i love u so much'
        >>> y = 'i hate u'
        >>> _score_em(x, y)
        >>> 0 
    """
    candidate = tokenize(candidate)
    reference = tokenize(reference)
    if len(candidate) != len(reference):
        return 0
    for (i, j) in zip(candidate, reference):
        if i != j:
            return 0
    return 1


def _lcs(candidate: Sequence[str], reference: Sequence[str]) -> dict:
    """Calculate the longest common subsequence table.

    Using the Dynamic Programming (DP) algorithm: O(mn)
    for `m = len(x); n = len(y)`

    Args:
        x: Sequence consisting of words or ngrams.
        y: Same as x.

    Returns:
        table: The longest common subsequence table.
    """
    m, n = len(candidate), len(reference)
    table = dict()

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif reference[i - 1] == candidate[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i, j - 1], table[i - 1, j])
    return table


########################################################

import sys
import math
from collections import Counter

def get_dict(tokens, ngram, gdict=None):
    """
    get_dict
    统计n-gram频率并用dict存储
    """
    token_dict = {}
    if gdict is not None:
        token_dict = gdict
    tlen = len(tokens)
    for i in range(0, tlen - ngram + 1):
        ngram_token = "".join(tokens[i:(i + ngram)])
        if token_dict.get(ngram_token) is not None: 
            token_dict[ngram_token] += 1
        else:
            token_dict[ngram_token] = 1
    return token_dict

def count(pred_tokens, gold_tokens, ngram, result):
    """
    计算BLEU中pn
    """
    cover_count, total_count = result
    pred_dict = get_dict(pred_tokens, ngram)
    gold_dict = get_dict(gold_tokens, ngram)
    cur_cover_count = 0
    cur_total_count = 0
    for token, freq in pred_dict.items():
        if gold_dict.get(token) is not None:
            gold_freq = gold_dict[token]
            cur_cover_count += min(freq, gold_freq)
        cur_total_count += freq
    result[0] += cur_cover_count
    result[1] += cur_total_count


def calc_bp(pair_list):
    """
    calc_bp
    """
    c_count = 0.0
    r_count = 0.0
    for pair in pair_list:
        pred_tokens, gold_tokens = pair
        c_count += len(pred_tokens)
        r_count += len(gold_tokens)
    bp = 1
    if c_count < r_count:
        bp = math.exp(1 - r_count / c_count)
    return bp 


def calc_cover_rate(pair_list, ngram):
    """
    calc_cover_rate
    """
    result = [0.0, 0.0] # [cover_count, total_count]
    for pair in pair_list:
        pred_tokens, gold_tokens = pair
        count(pred_tokens, gold_tokens, ngram, result)
    cover_rate = result[0] / result[1]
    return cover_rate 


def calc_bleu(pair_list):
    """
    calc_bleu
    """
    bp = calc_bp(pair_list)
    cover_rate1 = calc_cover_rate(pair_list, 1)
    cover_rate2 = calc_cover_rate(pair_list, 2)
    cover_rate3 = calc_cover_rate(pair_list, 3)
    bleu1 = 0
    bleu2 = 0
    bleu3 = 0
    if cover_rate1 > 0:
        bleu1 = bp * math.exp(math.log(cover_rate1))
    if cover_rate2 > 0:
        bleu2 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2)) / 2)
    if cover_rate3 > 0:
        bleu3 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2) + math.log(cover_rate3)) / 3)
    return [bleu1, bleu2]


def calc_distinct_ngram(pair_list, ngram):
    """
    calc_distinct_ngram
    """
    ngram_total = 0.0
    ngram_distinct_count = 0.0
    pred_dict = {}
    for predict_tokens, _ in pair_list:
        get_dict(predict_tokens, ngram, pred_dict)
    for key, freq in pred_dict.items():
        ngram_total += freq
        ngram_distinct_count += 1 
        #if freq == 1:
        #    ngram_distinct_count += freq
    return ngram_distinct_count / ngram_total


def calc_distinct(pair_list):
    """
    calc_distinct
    """
    # pair_list = 
    distinct1 = calc_distinct_ngram(pair_list, 1)
    distinct2 = calc_distinct_ngram(pair_list, 2)
    return [distinct1, distinct2]


def calc_f1(data):
    """
    calc_f1
    """
    golden_char_total = 0.0
    pred_char_total = 0.0
    hit_char_total = 0.0
    for response, golden_response in data:
        #golden_response = "".join(golden_response).decode("utf8")
        #response = "".join(response).decode("utf8")
        golden_response = "".join(golden_response)
        response = "".join(response)
        common = Counter(response) & Counter(golden_response)
        hit_char_total += sum(common.values())
        golden_char_total += len(golden_response)
        pred_char_total += len(response)
    p = hit_char_total / pred_char_total
    r = hit_char_total / golden_char_total
    f1 = 2 * p * r / (p + r)
    return f1


########################################################

def benchmark_sample(predict, label):
    scorer = RougeScorer(["rougeL", "EM"])
    for (cand, ref) in zip(predict, label):
        scorer.add_instance(cand, ref)  
    return scorer.score()

def benchmark_file():
    with open("/mnt/lihongguang/preprocess/CQA_benchmark/result/labels.txt", "r")as f1, open("/mnt/lihongguang/preprocess/CQA_benchmark/result/query_rewriter_generative_reader_responser.txt", "r") as f2:
        predict, label = [], []
        for item1, item2 in zip(f1.readlines(), f2.readlines()):
            item1 = item1.strip()
            item2 = item2.strip()
            label.append(item1)
            predict.append(item2)
    print(len(label))
    scorer = RougeScorer(["rougeL", "EM"])
    for (cand, ref) in zip(predict, label):
        scorer.add_instance(cand, ref)  
    return scorer.score()

label = ["中国领土面积有960万平方公里", "中国的首都是北京","960万平方公里", "北京啊", "我很开心"]
predict = ["960万平方公里", "北京", "960万平方公里", "北京", "开心"]
# print("conversational QA benchmark sample:")
# print(benchmark_sample(predict, label))
print(benchmark_file())

#######################################################

# with open('bart_large_generate_gpu.json', 'r') as f:
#     datas = f.readlines()
# one_stage = [eval(data)['one_stage'] for data in datas]
# one_stage_prompt = [eval(data)['one_stage_prompt'] for data in datas]
# two_stage_responser_prompt = [eval(data)['two_stage_responser_prompt'] for data in datas]
# two_stage_reader_prompt = [eval(data)['two_stage_reader_prompt'] for data in datas]
# two_stage_reader = [eval(data)['two_stage_reader'] for data in datas]
# two_stage_responser = [eval(data)['two_stage_responser'] for data in datas]
# response = [eval(data)['response'] for data in datas]
# answers = [eval(data)['answer'] for data in datas]

# print('***start testing ***')
# # store file
# f2 =  open("/mnt/chennuo/penguin/test_bart_large.csv", "w")
# csv_writer = csv.writer(f2)
# csv_writer.writerow(["Model", "BLEU-1", "BLEU-2", "Distinct-1", "Distinct-2", "Rouge-l", "EM"])

# # names = ['one_stage','two_stage_reader', 'two_stage_responser']
# names  = ['one_stage', 'one_stage_prompt', 'two_stage_responser_prompt', 'two_stage_reader_prompt', 'two_stage_reader', 'two_stage_responser']
# # for tokenize
# print('******start tokenzing*****')
# # all_predict =  [one_stage, two_stage_reader, two_stage_responser]
# all_predict = [one_stage, one_stage_prompt, two_stage_responser_prompt, two_stage_reader_prompt, two_stage_reader, two_stage_responser]
# for i in range(0,6):
#     sents = []
#     print('caclulating ....')
#     # for two-stage reader, only consider answer as reference. Otherwise, response is regarded as reference.
#     if i in [3,4]:
#         for pre, res in zip(all_predict[i], answers):
#             sents.append([tokenize(pre), tokenize(res)])
#     else:
#         for pre, res in zip(all_predict[i], response):
#             sents.append([tokenize(pre), tokenize(res)])
#     # calc bleu
#     bleu1, bleu2 = calc_bleu(sents)
#     # calc distinct
#     distinct1, distinct2 = calc_distinct(sents)
#     if i in [3,4]:
#         metrics  = benchmark_sample(all_predict[i], answers)
#     else:
#         metrics  = benchmark_sample(all_predict[i], response)
#     print('test_metrics of {}: '.format(names[i]), metrics)
#     csv_writer.writerow([names[i], bleu1, bleu2,distinct1, distinct2, metrics['rougeL'] , metrics["EM"]])

