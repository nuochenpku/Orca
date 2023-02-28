import json, jsonlines
import pandas as pd
import csv
from ConversationalQA_benchmark import benchmark_sample, calc_bleu, calc_distinct, tokenize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--predict_path', type=str, help='predict path')
parser.add_argument('--labels_path', type=str, help='label path for evaluate the model')
args = parser.parse_args()

## load ground truth resoponses
with open(args.labels_path, 'r', encoding='utf-8') as f:
    data = f.readlines()
labels = [line.strip() for line in data]


def load_t5_preds(phrase):
    if phrase == '0_shot':
        filepath = 'total_results/t5/zeroshot_prediction_sampling_300.json'
    elif phrase == '5_shot':
        filepath = 'total_results/t5/5shot_prediction_sampling_300.json'
    elif phrase == '10_shot':
        filepath = 'total_results/t5/10shot_prediction_sampling_300.json'
    elif phrase == '200_shot':
        filepath = 'total_results/t5/fewshot_prediction_sampling_300.json'
    
    preds, labels = [], []
    with jsonlines.open(filepath, 'r') as reader:
        for row in reader:
            t5_pred = row['prediction']
            gt_label = row['target']
            preds.append(t5_pred)
            labels.append(gt_label)

    return preds, labels


def load_gpt3_preds(phrase):
    if phrase == '0_shot':
        # filepath = 'total_results/gpt3/gpt3_zero_shot_results.txt'
        filepath = 'total_results/gpt3/gpt3_xiaoice_convqa_results.txt'
    elif phrase == '5_shot':
        filepath = 'total_results/gpt3/gpt3_5shot_results.txt'
        # filepath = 'total_results/gpt3/gpt3_5shot_results.tsv'
    elif phrase == '0_shot_wo_passage':
        filepath = 'total_results/gpt3/gpt3_orca_0shot_no_passage_results.txt'
    
    with open(filepath, 'r') as f:
        preds = f.readlines()
    gpt3_preds = [line.strip().split('\t')[1] for line in preds]
    labels = [line.strip().split('\t')[2] for line in preds]

    # gpt3_preds, labels = [], []
    # with open(filepath, 'r') as f:
    #     tsvreader = csv.reader(f, delimiter='\t')
    #     for line in tsvreader:
    #         gpt3_preds.append(line[1])
    #         labels.append(line[2])

    return gpt3_preds, labels


def evaluation(preds, phrase):
    print(phrase)

    ## rouge-L & EM
    print(benchmark_sample(preds, labels))

    ## BLEU & Dist
    sents = []
    for pred, label in zip(preds, labels):
        sents.append([tokenize(pred), tokenize(label)])

    bleu1, bleu2 = calc_bleu(sents)
    distinct1, distinct2 = calc_distinct(sents)

    print('bleu1: ', bleu1)
    print('bleu2: ', bleu2)
    print('distinct1: ', distinct1)
    print('distinct2: ', distinct2)
    print()


if __name__ == '__main__':

    ##load predictions of different models
    ## pipeline system (zero-shot)
    # filepath = 'total_results/pileline_system/query_rewriter_generative_reader_responser.txt'
    # with open(filepath, 'r') as f:
    #     pipeline_preds = f.readlines()
    # pipe_0_shot_preds = [line.strip() for line in pipeline_preds]

    # bart (zero-shot & few-shot)
    with open(agrs.predict_path, 'r') as f:
        bart_0_shot = f.readlines()
    bart_0_shot_preds = [line.strip() for line in bart_0_shot]
    # with open('total_results/bart/five_shot_2.txt', 'r') as f:
    #     bart_5_shot = f.readlines()
    # bart_5_shot_preds = [line.strip() for line in bart_5_shot]
    # with open('total_results/bart/ten_shot_2.txt', 'r') as f:
    #     bart_10_shot = f.readlines()
    # bart_10_shot_preds = [line.strip() for line in bart_10_shot]
    # with open('total_results/bart/few_shot_2.txt', 'r') as f:
    #     bart_200_shot = f.readlines()
    # bart_200_shot_preds = [line.strip() for line in bart_200_shot]


    ## t5 (zero-shot & few-shot)
    # t5_0_shot_preds, _ = load_t5_preds('0_shot')
    # t5_5_shot_preds, _ = load_t5_preds('5_shot')
    # t5_10_shot_preds, _ = load_t5_preds('10_shot')
    # t5_200_shot_preds, _ = load_t5_preds('200_shot')

    ## gpt3 (zero-shot & 5-shot)
    # gpt3_0_shot_preds, _ = load_gpt3_preds('0_shot')
    # gpt3_5_shot_preds, _ = load_gpt3_preds('5_shot')

    ## evaluation
    # evaluation(pipe_0_shot_preds, 'pipeline_0_shot')

    # evaluation(gpt3_0_shot_preds, 'gpt3_0_shot')
    # evaluation(gpt3_5_shot_preds, 'gpt3_5_shot')

    evaluation(bart_0_shot_preds, 'bart_0_shot')
    # evaluation(bart_5_shot_preds, 'bart_5_shot')
    # evaluation(bart_10_shot_preds, 'bart_10_shot')
    # evaluation(bart_200_shot_preds, 'bart_200_shot')

    # evaluation(t5_0_shot_preds, 't5_0_shot')
    # evaluation(t5_5_shot_preds, 't5_5_shot')
    # evaluation(t5_10_shot_preds, 't5_10_shot')
    # evaluation(t5_200_shot_preds, 't5_200_shot')


    ## gpt3 (zero-shot w/o passage)
    # gpt3_0_shot_wo_passage_preds, _ = load_gpt3_preds('0_shot_wo_passage')
    # evaluation(gpt3_0_shot_wo_passage_preds, 'gpt3_0_shot_wo_passage')

"""
gpt3_0_shot_wo_passage
{'rougeL': 0.34023869721871564, 'EM': 0.02040253653156879}
bleu1:  0.2541321303397353
bleu2:  0.18814354772725536
distinct1:  0.05267407823967931
distinct2:  0.44835980768060374
"""

"""
pipeline_0_shot
{'rougeL': 0.6206763015403293, 'EM': 0.1483319547835677}
bleu1:  0.6110043954922615
bleu2:  0.5821165018319964
distinct1:  0.044703126320746744
distinct2:  0.48514888995529515

------------------------------------
[tsv另存为txt文件, 跑出来的结果]
gpt3_0_shot
{'rougeL': 0.3676835491707568, 'EM': 0.03253377446925834}
bleu1:  0.3229383478073638
bleu2:  0.29051564228326665
distinct1:  0.03855953491710328
distinct2:  0.4476663853779315

gpt3_5_shot
{'rougeL': 0.5691637777514997, 'EM': 0.07085745795423215}
bleu1:  0.5017747791053486
bleu2:  0.4663286287916172
distinct1:  0.05012041560586252
distinct2:  0.485138619311104
-------------------------------------

bart_0_shot
{'rougeL': 0.14989445770397947, 'EM': 0.0}
bleu1:  0.11120726736445381
bleu2:  0.0922244437534092
distinct1:  0.011380860695192643
distinct2:  0.1821388790539126

bart_5_shot
{'rougeL': 0.26792119029834927, 'EM': 0.015439757375241246}
bleu1:  0.24719508437403712
bleu2:  0.2137059360109999
distinct1:  0.03463776259266644
distinct2:  0.3972356276062409

bart_10_shot
{'rougeL': 0.4087214198273776, 'EM': 0.05817480011028398}
bleu1:  0.3872977636744762
bleu2:  0.3529648908028048
distinct1:  0.04236395010828402
distinct2:  0.4451039651837524

bart_200_shot
{'rougeL': 0.6989569273472869, 'EM': 0.2522746071133168}
bleu1:  0.6158657425177291
bleu2:  0.590469700142345
distinct1:  0.05335234982968235
distinct2:  0.5156070038343223

t5_0_shot
{'rougeL': 0.2061257957301521, 'EM': 0.0035842293906810036}
bleu1:  0.16462158082317055
bleu2:  0.13550594472103028
distinct1:  0.01968669159604387
distinct2:  0.18090690594456016

t5_5_shot
{'rougeL': 0.47391803532343885, 'EM': 0.06892748828232699}
bleu1:  0.42410034207525654
bleu2:  0.39703938256146554
distinct1:  0.036488027366020526
distinct2:  0.41947017868261666

t5_10_shot
{'rougeL': 0.5779696470427008, 'EM': 0.12544802867383512}
bleu1:  0.4658861114415348
bleu2:  0.4397966970858333
distinct1:  0.035813210528998926
distinct2:  0.4309397108716807

t5_200_shot
{'rougeL': 0.7306089337929578, 'EM': 0.3140336366142818}
bleu1:  0.6824911454781442
bleu2:  0.6610775931777265
distinct1:  0.04502056888927998
distinct2:  0.4681567701948744
"""

"""
[tsv文件, 跑出来的结果]

gpt3_0_shot
{'rougeL': 0.38328257778080976, 'EM': 0.03451301550160866}
bleu1:  0.33705412464492024
bleu2:  0.30467408939087653
distinct1:  0.040016776924103485
distinct2:  0.45383152521997894

gpt3_5_shot
{'rougeL': 0.5621384725387714, 'EM': 0.06933879461673494}
bleu1:  0.4928234071839632
bleu2:  0.4577083545731402
distinct1:  0.051922036525273794
distinct2:  0.4908808274206209
"""