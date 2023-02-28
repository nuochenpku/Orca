# torch 1.7.1+cu110  transformers 4.4.1
import os, json
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from transformers import BertTokenizer, BartForConditionalGeneration
from tqdm import tqdm
from ConversationalQA_benchmark import benchmark_sample
import torch
import time
import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='few_shot', help='zero_shot/five_shot/ten_shot/few_shot')
parser.add_argument('--model_path', type=str, help='inference model path')
parser.add_argument('--output_path', type=str, help='output file path')
parser.add_argument('--bsz', type=int, default=30, help='batch size')
args = parser.parse_args()


# 指定GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("模型推理硬件设备:{}\t".format(device))


def main():
    predictions = []
    labels = []

    ## load model
    bart_model_path = args.model_path
    tokenizer=BertTokenizer.from_pretrained(bart_model_path)
    bart_model = BartForConditionalGeneration.from_pretrained(bart_model_path)
    bart_model.config.max_length=128
    bart_model.to(device)
    bart_model.eval()
    print("加载bart-" + args.type + "模型完成")

    def inference(prompt):
        #预处理
        inputs = prompt
        # 模型编码
        inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
        # 模型推理
        with torch.no_grad():
            generated_ids = bart_model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=128).to(device)
        # 模型解码
        outputs = tokenizer.batch_decode(generated_ids , skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs = [item.replace(" ", "") for item in outputs]
        return outputs

    ## save generated responses
    # output_file = open('result/bart_1121_new/' + args.type + '.txt', 'w+')
    output_file = open(args.output_path + args.type + '.txt', 'w+')
    labels_file = open(args.output_path + 'labels.txt', 'w+')

    with open("orca_test_reformat_1115.json", "r") as f1:
        item = json.load(f1)
    
    batch_prompt = []
    batch_labels = []
    for i in tqdm(range(len(item.keys()))):
        query = item[str(i)]["query"]
        label = item[str(i)]["response"]
        passage = item[str(i)]["knowledge"]
        sessions = item[str(i)]["context-cqa"]
        history_turn = len(list(sessions))

        if history_turn == 0:
            history = "无。"
        else:
            history = ''
            for t in range(history_turn):
                history += sessions[str(t)]['query'] + ","+ sessions[str(t)]['response'] +","
        
        prompt = "对话历史:"+ history + "问题：" + query + ","+ "段落：" +  passage
        # if len(prompt) > 510:
        #     # prompt = prompt[-512:]
        #     prompt = prompt[0:510]
        
        if len(prompt) > 510:
            tmp = "问题：" + query + ","+ "段落：" +  passage
            if len(tmp) > 510:
                prompt = tmp[0:510]
            else:
                rest_len = 510 - len(tmp) - 5
                prompt = "对话历史:"+ history[-rest_len:] + tmp
        
        batch_prompt.append(prompt)
        batch_labels.append(label)
        if i % args.bsz == 0:
            outputs = inference(batch_prompt)
            predictions += outputs
            labels += batch_labels
            for pred, label in zip(outputs, batch_labels):
                output_file.write(pred + "\n")
                labels_file.write(label+ "\n")
            batch_labels = []
            batch_prompt = []
        elif i == (len(item.keys())-1):
            outputs = inference(batch_prompt)
            predictions += outputs
            labels += batch_labels
            for pred, label in zip(outputs, batch_labels):
                output_file.write(pred + "\n")
                labels_file.write(label+ "\n")
            batch_labels = []
            batch_prompt = []

    print(args.type)
    print(benchmark_sample(predictions, labels))


if __name__ == "__main__":
    main()

"""
[0:510]

zero_shot
{'rougeL': 0.14914854588919793, 'EM': 0.0}

five_shot
{'rougeL': 0.2647925339601548, 'EM': 0.01488833746898263}

ten_shot
{'rougeL': 0.40446598782554405, 'EM': 0.05596912048524952}

few_shot
{'rougeL': 0.6893599774383141, 'EM': 0.24703611800385994}
"""

"""
if len(prompt) > 510:
            tmp = "问题：" + query + ","+ "段落：" +  passage
            if len(tmp) > 510:
                prompt = tmp[0:510]
            else:
                rest_len = 510 - len(tmp) - 5
                prompt = "对话历史:"+ history[-rest_len:] + tmp
====================================================================
[bart_1121_new/few_shot_2.txt]      [221121_bart_new_history_2.log & 221121_bart_new_history_few_shot_2.log]
====================================================================
few_shot
{'rougeL': 0.6989569273472869, 'EM': 0.2522746071133168}

zero_shot
{'rougeL': 0.14989445770397947, 'EM': 0.0}

five_shot
{'rougeL': 0.26792119029834927, 'EM': 0.015439757375241246}

ten_shot
{'rougeL': 0.4087214198273776, 'EM': 0.05817480011028398}
"""