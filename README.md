# Orca: A Few-shot Benchmark for Chinese Conversational Machine Reading Comprehension


[**Tasks**](#task-description) | [**Dataset**](#dataset) | [**Checkpoints**](#Checkpoints) |
[**Paper**](https://arxiv.org/abs/2010.04898) |
[**Citation**](#citation) | [**License**](#license)

This repository contains resources for accessing the official benchmarks, codes and checkpoints of the paper:  ***Orca: A Few-shot Benchmark for Chinese Conversational Machine Reading Comprehension***.

We collect $\textbf{Orca}$, the first few-shot Chinese CMRC benchmark. Concretely, we  annotate the following components in Orca: 1) $\textbf{Topic}$, which consists of several sentences to drive the whole conversation; 2) $\textbf{Domain}$, only one or two words indicate the specific field o which the content of the conversation belongs; 3) $\textbf{Conversations}$, where each turn is assigned a golden knowledgeable passage. Importantly, the question and response in each turn  are human-crafted, leading to more coherent conversations. And response-related passages are also manually selected from the search engine. In total, we collect manually annotated 831 conversations and 4742 turns in Orca. Even though the size of our dataset could be questioned, our goal is to build strong CMRC models under few-shot or even zero-shot settings, and without needing to collect data for each target domain.

The highlight of $\textbf{Orca}$ are three-folds: 1) Each turn of a conversation is assigned a ground-truth knowledgeable passage, fitting human cognition towards CMRC. 2) Responses are free-form text with demonstrative pronouns instead of extracted answer span from the passage, making the expression more natural. 3) As the conversations are collected based on hot-topics from social media across 33 domain, the domains of conversations are much more diverse than existing datasets.

## DataSet

Orca has the following salient features: 1) Data in Orca are collected from November 2021 to November 2022 on Weibo, one of the most popular social media platforms in China. This means that the collected data reflect real human interests, are quite new, and have never been included in earlier benchmarks, posing a challenge to existing language models.  Moreover, good results on
Orca are of practical interest. 2) We carefully  annotate conversations across 33 domains. In contrast, as the current commonly-used datasets, data in CoQA are only from 7 domains, DoQA contains data from 3 domains. The variety of data domains makes Orca closer to real scenarios and  better-evaluating the generalization of CMRC models. 3) Answers at each turn in a conversation are  natural and informative responses from human annotation rather than certain spans in the provided passage. This way, we can both evaluate models' comprehension ability and generation ability.

$\textbf{Orca}$ contains 831 conversations and 4,742 turns. For each conversation, there are 5.71 turns on average, and each turn is assigned a knowledgeable passage. We randomly split collected data into support (train) and test set. Concretely, the support set and test set contain 200 and 631 conversations, separately.

Concretely, we store our dataset in json files:


```
 "0": {
        "topic": "邓伦资本版图",
        "domain": "人物",
        "context": {
            "0": {
                "query": "邓论是谁？",
                "response": "邓伦，1992年10月21日出生于河北省石家庄市，中国内地影视男演员，毕业于上海戏剧学院表演系。",
                "query-type": "Causal",
                "passage": "邓伦，1992年10月21日出生于河北省石家庄市，中国内地影视男演员，毕业于上海戏剧学院表演系。天眼查App显示，邓伦名下有2家公司，分别为邓伦（上海）影视文化工作室和舟山邓伦影视文化工作室，2家工作室均为邓伦个人独资企业，分别成立于2016年和2018年。"
            },
...
            "4": {
                "query": "他有哪些代言",
                "response": "邓伦有宝格丽、雪花秀、欧莱雅、联合利华等代言",
                "query-type": "List",
                "passage": "邓伦代言汇总： 1、宝格丽：品牌代言人 2、雪花秀：品牌亚太区代言人 3、欧莱雅：彩妆品牌代言人 4、联合利华：中国区洗护发代言人，清扬净爽代言人"
            }
        }
    },
```

Each conversation in the dataset contains a unique number, `Turn_no`, `Topic`, `domain`  unique within a conversation, the  `Question`, `query`, `response`, `query-type` with `passage` unique within each turn.

### Download

- **Support Set**
  - [**5-shot Set**](https://hkustgz-my.sharepoint.com/:u:/g/personal/nchen022_connect_hkust-gz_edu_cn/EQSQOzgua51Omi8j-y6V7j4BOOFoIMYXg-Vg4BFNLuyKCw?e=gqFe6U)
  - [**10-shot Set**](https://hkustgz-my.sharepoint.com/:u:/g/personal/nchen022_connect_hkust-gz_edu_cn/EfUGD87nc79BgngLJkPKmEoB9OhI4vSckrQhiwg3YZ7dIQ?e=fesXnX)
  - [**200-shot Set**](https://hkustgz-my.sharepoint.com/:u:/g/personal/nchen022_connect_hkust-gz_edu_cn/EXXUL8WIpBVMo5PGe-Mef0ABK4wDLhLBCYeRJ81hr_-ggA?e=dTZDiZ)
  
- [**Test Set**]
For the test set, please contact the chennuo26@gmail.com.



## Checkpoints

### Results
Here we report automatic and huaman evaluations results of four baselines in our paper.

![](Results.png) 

###  Download

|Model |  zero-shot| 5-shot | 10-shot | 200-shot |
| :----- | :-------------------:| :------------------: | :------------------: |:------------------: |
| T5 | T5-base| T5-base |  T5-base |  T5-base | 
| BART | [BART-Large](https://hkustgz-my.sharepoint.com/:u:/g/personal/nchen022_connect_hkust-gz_edu_cn/EfQYlzgOAI9Bogfy2f2hSCwBu_z4acFeFY16jTz5G2I8eg?e=LBQ1pZ)    |[BART-Large](https://hkustgz-my.sharepoint.com/:f:/g/personal/nchen022_connect_hkust-gz_edu_cn/EmA0EfSvS85KtYXferY0MjIBoiojfGMRDZxBs8KbruY6VQ?e=uSdZrQ)  | [BART-Large](https://hkustgz-my.sharepoint.com/:u:/g/personal/nchen022_connect_hkust-gz_edu_cn/EemWOJFzZy1JmmWhm3vCJmQBpyOJfGLbG7o-VSC6Ord16A?e=lANCbq)  | [BART-Large](https://hkustgz-my.sharepoint.com/:f:/g/personal/nchen022_connect_hkust-gz_edu_cn/EpFR5MDb-zZKnPtAQNzmvaEBvrVNGrjPMfszJT8hcTCGdw?e=QeQPWr)  |


## Evaluation


## Citation

```
@inproceedings{Chen2023NaturalRG,
  title={Natural Response Generation for Chinese Reading Comprehension},
  author={Nuo Chen and Hongguang Li and Yinan Bao and Baoyuan Wang and Jia Li},
  year={2023}
}
```
