import sys

sys.path.append('../')
from logparser.UniParser import UniParser
from logparser.utils import evaluator

import os
import pandas as pd
import torch

# from torch.utils.data.distributed import DistributedSampler
# torch.distributed.init_process_group(backend="nccl")

input_dir = '../logs/'  # The input directory of log file
output_dir = 'UniparserResult/'  # The output directory of parsing results

benchmark_settings = {

    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
        "filter": '(\s+blk_)|(:)|(\s)',
        'k': 3,
        'vd': 3,
        'nr_epochs': 3,
        "batch": 256,
        "batch_test": 512
    },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        # "filter": r"([\s=,，。:\[\]\(\)<>])",
        "filter": r"(\s+appattempt_)|(job_)|([\s=,，。\[\]\(\)<>])",
        'k': 3,
        'vd': 3,
        'nr_epochs': 4,
        "batch": 256,
        "batch_test": 512
    },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
        "filter": '([ ])|(\d+\sB)|(\d+\sKB)|(\d+\.){3}\d+|\b[KGTM]?B\b|([\w-]+\.){2,}[\w-]+',
        'k': 3,
        'vd': 3,
        'nr_epochs': 4,
        "batch": 256,
        "batch_test": 512
    },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
        "filter": r"([\s=,，。:\[\]\(\)<>])",
        'k': 3,
        'vd': 3,
        'nr_epochs': 4,
        "batch": 256,
        "batch_test": 512
    },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+'],
        "filter": '([ |:|\(|\)|=|,])|(core.)|(\.{2,})',
        'k': 3,
        'vd': 3,
        'nr_epochs': 4,
        "batch": 256,
        "batch_test": 512
    },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'regex': [r'=\d+'],
        "filter": '([ |=])',
        'k': 3,
        'vd': 3,
        'nr_epochs': 4,
        "batch": 256,
        "batch_test": 512
    },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        "filter": r"([\s=,，。\[\]\(\)<>])",
        'k': 3,
        'vd': 3,
        'nr_epochs': 4,
        "batch": 256,
        "batch_test": 512
    },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'regex': [r'0x.*?\s'],
        "filter": '([ ])',
        'k': 3,
        'vd': 3,
        'nr_epochs': 4,
        "batch": 256,
        "batch_test": 512
    },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
        "filter": r"([\s=,，。:\[\]\(\)<>])",
        'k': 3,
        'vd': 3,
        'nr_epochs': 4,
        "batch": 256,
        "batch_test": 512
    },

    'Andriod': {
        'log_file': 'Andriod/Andriod_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
        "filter": '([ |:|\(|\)|=|,|"|\{|\}|@|$|\[|\]|\||;])',
        'k': 3,
        'vd': 3,
        'nr_epochs': 4,
        "batch": 256,
        "batch_test": 512
    },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'regex': [],
        "filter": '([ ])',
        'k': 3,
        'vd': 3,
        'nr_epochs': 4,
        "batch": 256,
        "batch_test": 512
    },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        "filter": '([ ])',
        'k': 3,
        'vd': 3,
        'nr_epochs': 4,
        "batch": 256,
        "batch_test": 512
    },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
        "filter": r"([\s=,，。:\[\]\(\)<>])",
        'k': 3,
        'vd': 3,
        'nr_epochs': 4,
        "batch": 256,
        "batch_test": 512
    },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
        "filter": r"([\s=,，。:\[\]\(\)<>])",
        'k': 3,
        'vd': 3,
        'nr_epochs': 4,
        "batch": 256,
        "batch_test": 512
    },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],
        "filter": '([ |:|\(|\)|"|\{|\}|@|$|\[|\]|\||;])',
        'k': 3,
        'vd': 3,
        'nr_epochs': 4,
        "batch": 256,
        "batch_test": 512
    },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'regex': [r'([\w-]+\.){2,}[\w-]+'],
        "filter": '([ ])|([\w-]+\.){2,}[\w-]+',
        'k': 3,
        'vd': 3,
        'nr_epochs': 4,
        "batch": 256,
        "batch_test": 512
    }

}



def mla(pre, label):
    assert len(pre) == len(label)
    score = 0
    for i in range(len(pre)):
        if pre[i] == label[i]:
            score += 1
    return score / len(pre)


import copy
from Cal_Score import cal_score

bechmark_result = []
datas = []
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("../logs/model.model")
for index, (dataset, setting) in enumerate(benchmark_settings.items()):
    print('\n=== Data Processing on %s ===' % dataset)
    indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
    log_file = os.path.basename(setting['log_file'] + "_structured.csv")
    parser = UniParser.LogParser(indir=indir, outdir=output_dir, regex=setting['regex'], filter=setting["filter"],
                                 k=setting['k'], log_format=setting['log_format'])
    # data, _ = parser.preprocessing_data_stp(sp, log_file)
    data, _ = parser.preprocessing_data(log_file)
    datas.append(data)

result = []
result_score = []
for index, (dataset, setting) in enumerate(benchmark_settings.items()):
    # if index != 0:
    #     continue
    print('\n=== Evaluation on %s ===' % dataset)
    data_try = copy.deepcopy(datas)
    test_data = data_try[index]
    # train/test spliting
    data_try.remove(test_data)
    train_data = []
    for i in data_try:
        train_data += i
    parser = UniParser.LogParser(indir=indir,
                                 outdir=output_dir,
                                 regex=setting['regex'],
                                 filter=setting["filter"],
                                 k=setting['k'],
                                 log_format=setting['log_format'])
    parser.parse(train_data,
                 nr_epochs=setting['nr_epochs'],
                 batch=setting["batch"],
                 batch_test=setting["batch_test"],
                 mode="train")
    predict = parser.parse(test_data,
                           nr_epochs=setting['nr_epochs'],
                           batch=setting["batch"],
                           batch_test=setting["batch_test"],
                           mode="test")

    predict_if_para, tokens, pattern, label, pattern_tem = predict

    pattern_text = []
    for ii in pattern:
        pt = ii[0]
        for iii in range(1, len(ii)):
            if ii[iii] == "<*>" and ii[iii - 1] == "<*>":
                continue
            pt += ii[iii]
        pattern_text.append(pt)
    pattern_label = ["".join(i) for i in pattern_tem]
    sc = cal_score(pattern_text, pattern_label)
    precision = sc.precision()
    recall = sc.recall()
    f1 = sc.f_measure()
    pa = sc.parsing_accuracy()

    mla_sc = mla(predict_if_para, label)
    score = (precision, recall, f1, pa, mla_sc)
    with open('./UniparserResult/%s_result.txt' % dataset, 'w') as file:
        for metrics in score:
            file.write(str(metrics))
            file.write('\n')
        file.close()
    result.append(predict)
    result_score.append(score)
    # if index == 0:
    #     break

