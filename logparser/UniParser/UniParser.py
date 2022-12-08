

import uuid
import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, RandomSampler, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
# from torch.optim.lr_scheduler import ExponentialLR
import random
import re
import os
import time


class LogTokenizer:
    def __init__(self, filters='([ |:|\(|\)|=|,])|(core.)|(\.{2,})'):
        self.filters = filters

    def tokenize(self, sent):
        sent = sent.replace('\'', '')
        filtered = re.split(self.filters, sent)
        new_filtered = []
        for f in filtered:
            if f != None and f != '':
                new_filtered.append(f)
        return new_filtered


class LogParser:
    def __init__(self, indir, outdir, regex, filter, k, log_format):
        self.path = indir
        self.logName = None
        self.savePath = outdir
        # self.filters = filters
        self.rex = regex
        self.filter = filter
        self.temp_filter = "([a-zA-Z]+_<\*>)|(<\*>)|" + filter
        # self.temp_filter = "(<*>)"+filter
        # for i in filter:
        #     if i not in "<*>":
        #         self.temp_filter += i
        self.k = k
        self.df_log = None
        self.log_format = log_format
        # self.tokenizer = LogTokenizer(filters)
        self.parser = LogParserUniparser(emb_dim=64,
                                         hidden_dim=64)

    def parse(self, datas, nr_epochs=5, batch=5, batch_test=5, mode="train"):
        # self.logName = logName

        # datas, _ = self.preprocessing_data(logName)
        print("Start training")
        if mode == "train":
            self.parser.canonical_train(datas,
                                        nr_epochs=nr_epochs,
                                        batch_size=batch,
                                        lr=0.002)
            return
        else:
            predict = self.parser.canonical_inference(datas,
                                                      batch_size=batch_test)
            tokens = [i["tokens"] for i in datas]
            pattern = [[j if predict[i][index] == 0 else "<*>" for index, j in enumerate(datas[i]["tokens"])]
                       for i in range(len(datas))]
            label = [i["if_para"] for i in datas]
            pattern_label = [i["event_temp"] for i in datas]
            return predict, tokens, pattern, label, pattern_label

    # def load_data(self):
    #     headers, regex = self.generate_logformat_regex(self.log_format)
    #     self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)

    def preprocessing_data_stp(self, stp, logName):
        self.logName = logName
        self.load_csv_data()
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        df_len = self.df_log.shape[0]
        datas = []
        error_data = []
        for i in range(df_len):
            # tokenized = self.tokenizer.tokenize(self.df_log.iloc[i].Content)
            # pattern_tokenized = self.tokenizer.tokenize(self.df_log.iloc[i].EventTemplate)
            # tokenized = self.preprocess(self.df_log.iloc[i].Content).strip().split(self.filter)
            tokenized = stp.EncodeAsPieces(self.df_log.iloc[i].Content.strip())
            tokenized = [ii for ii in tokenized if ii and ii != " "]
            # pattern_tokenized = self.preprocess(self.df_log.iloc[i].EventTemplate).strip().split()
            pattern_tokenized = stp.EncodeAsPieces(self.df_log.iloc[i].EventTemplate.strip())
            if len(tokenized) == len(pattern_tokenized):
                if_para = [0 if j == tokenized[i] else 1 for i, j in enumerate(pattern_tokenized)]
                datas.append({"tokens": tokenized,
                              "if_para": if_para,
                              "event_temp": pattern_tokenized,
                              "pattern_tokens": self.df_log.iloc[i].EventTemplate,
                              "raw_log": self.df_log.iloc[i].Content,
                              "log_template": [tokenized[i] if if_para[i] == 0 else "<*>" for i in
                                               range(len(tokenized))]
                              })
            else:
                j = 0
                if_para = []
                for tokens in tokenized:
                    if j >= len(pattern_tokenized):
                        if_para.append(1)
                        continue
                    if tokens == pattern_tokenized[j]:
                        if_para.append(0)
                        j += 1
                    elif "<*>" in pattern_tokenized[j]:
                        if_para.append(1)
                        j += 1
                    else:
                        if_para.append(1)
                if len(if_para) == len(tokenized):
                    datas.append({"tokens": tokenized,
                                  "if_para": if_para,
                                  "event_temp": pattern_tokenized,
                                  "pattern_tokens": self.df_log.iloc[i].EventTemplate,
                                  "raw_log": self.df_log.iloc[i].Content,
                                  "log_template": [tokenized[i] if if_para[i] == 0 else "<*>" for i in
                                                   range(len(tokenized))]
                                  })
                else:
                    error_data.append({"tokens": tokenized,
                                       "if_para": if_para,
                                       "event_temp": pattern_tokenized,
                                       "pattern_tokens": self.df_log.iloc[i].EventTemplate,
                                       "raw_log": self.df_log.iloc[i].Content,
                                       "log_template": [tokenized[i] if if_para[i] == 0 else "<*>" for i in
                                                        range(len(tokenized))]
                                       })
        return datas, error_data

    def preprocessing_data(self, logName):
        self.logName = logName
        self.load_csv_data()
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        df_len = self.df_log.shape[0]
        datas = []
        error_data = []
        for i in range(df_len):
            # tokenized = self.tokenizer.tokenize(self.df_log.iloc[i].Content)
            # pattern_tokenized = self.tokenizer.tokenize(self.df_log.iloc[i].EventTemplate)
            # tokenized = self.preprocess(self.df_log.iloc[i].Content).strip().split(self.filter)
            tokenized = re.split(self.filter, self.df_log.iloc[i].Content.strip())
            tokenized = [ii for ii in tokenized if ii and ii != " "]
            # pattern_tokenized = self.preprocess(self.df_log.iloc[i].EventTemplate).strip().split()
            pattern_tokenized = re.split(self.temp_filter,
                                         re.sub(r"<\*>", "变量", self.df_log.iloc[i].EventTemplate.strip()))
            pattern_tokenized = [ii for ii in pattern_tokenized if ii and ii != " "]
            pattern_tokenized = [ii if "变量" not in ii else "<*>" for ii in pattern_tokenized]
            # ii = 0
            # pattern_result = []
            # while ii < len(pattern_tokenized):
            #     if ii < len(pattern_tokenized)-2:
            #         if pattern_tokenized[ii][-1] == "<" and pattern_tokenized[ii+1] == "*" and pattern_tokenized[ii+2] == ">":
            #             pattern_result += ["".join(pattern_tokenized[ii:ii+3])]
            #             ii += 3
            #             continue
            #     pattern_result += [pattern_tokenized[ii]]
            #     ii += 1
            if len(tokenized) == len(pattern_tokenized):
                datas.append({"tokens": tokenized,
                              "if_para": [0 if j == tokenized[i] else 1 for i, j in enumerate(pattern_tokenized)],
                              "event_temp": pattern_tokenized
                              })
            else:
                j = 0
                if_para = []
                for tokens in tokenized:
                    if j >= len(pattern_tokenized):
                        if_para.append(1)
                        continue
                    if tokens == pattern_tokenized[j]:
                        if_para.append(0)
                        j += 1
                    elif "<*>" in pattern_tokenized[j]:
                        if_para.append(1)
                        j += 1
                    else:
                        if_para.append(1)
                if len(if_para) == len(tokenized):
                    datas.append({"tokens": tokenized,
                                  "if_para": if_para,
                                  "event_temp": pattern_tokenized})
                else:
                    error_data.append({"tokens": tokenized,
                                       "if_para": if_para,
                                       "event_temp": pattern_tokenized})
        return datas, error_data

    def preprocessing_our_data(self, logName):
        self.logName = logName
        self.load_csv_data()
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        df_len = self.df_log.shape[0]
        datas = []
        error_data = []
        for i in range(df_len):
            # tokenized = self.tokenizer.tokenize(self.df_log.iloc[i].Content)
            # pattern_tokenized = self.tokenizer.tokenize(self.df_log.iloc[i].EventTemplate)
            # tokenized = self.preprocess(self.df_log.iloc[i].Content).strip().split(self.filter)
            tokenized = re.split(self.filter, self.df_log.iloc[i].log_message.strip())
            tokenized = [ii for ii in tokenized if ii and ii != " "]
            # pattern_tokenized = self.preprocess(self.df_log.iloc[i].EventTemplate).strip().split()
            pattern_tokenized = re.split(self.temp_filter,
                                         re.sub(r"<\*>", "变量", self.df_log.iloc[i].pattern.strip()))
            pattern_tokenized = [ii for ii in pattern_tokenized if ii and ii != " "]
            pattern_tokenized = [ii if "变量" not in ii else "<*>" for ii in pattern_tokenized]
            # ii = 0
            # pattern_result = []
            # while ii < len(pattern_tokenized):
            #     if ii < len(pattern_tokenized)-2:
            #         if pattern_tokenized[ii][-1] == "<" and pattern_tokenized[ii+1] == "*" and pattern_tokenized[ii+2] == ">":
            #             pattern_result += ["".join(pattern_tokenized[ii:ii+3])]
            #             ii += 3
            #             continue
            #     pattern_result += [pattern_tokenized[ii]]
            #     ii += 1
            if len(tokenized) == len(pattern_tokenized):
                datas.append({"tokens": tokenized,
                              "if_para": [0 if j == tokenized[i] else 1 for i, j in enumerate(pattern_tokenized)],
                              "event_temp": pattern_tokenized
                              })
            else:
                j = 0
                if_para = []
                for tokens in tokenized:
                    if j >= len(pattern_tokenized):
                        if_para.append(1)
                        continue
                    if tokens == pattern_tokenized[j]:
                        if_para.append(0)
                        j += 1
                    elif "<*>" in pattern_tokenized[j]:
                        if_para.append(1)
                        j += 1
                    else:
                        if_para.append(1)
                if len(if_para) == len(tokenized):
                    datas.append({"tokens": tokenized,
                                  "if_para": if_para,
                                  "event_temp": pattern_tokenized})
                else:
                    error_data.append({"tokens": tokenized,
                                       "if_para": if_para,
                                       "event_temp": pattern_tokenized})
        return datas, error_data

    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def load_csv_data(self):
        # headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = pd.read_csv(os.path.join(self.path, self.logName))
        # self.csv_log_to_dataframe(os.path.join(self.path, self.logName), regex, headers,
        #                                         self.log_format, data_name="Content", template_name="EventTemplate")

    def csv_log_to_dataframe(self, csv_file, regex, headers, logformat, data_name, template_name):

        logdf = pd.read_csv(csv_file)
        # logdf = logdf[data_name].tolist()
        #
        # with open(csv_file, 'r') as fin:
        #     reader = csv.reader(fin)
        #     for line in reader:
        #         try:
        #             log = line[data_name]
        #             match = regex.search(log.strip())
        #             message = [match.group(header) for header in headers]
        #             log_messages.append(message)
        #             linecount += 1
        #         except Exception as e:
        #             pass
        # logdf = pd.DataFrame(log_messages, columns=headers)
        # logdf.insert(0, 'LineId', None)
        # logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex


class LogParserUniparser():
    def __init__(self,
                 emb_dim=512,
                 hidden_dim=100,
                 dropout=0.0,
                 context_width=3,
                 neg_num=3,
                 with_cuda=True,
                 cuda_devices=None
                 ):
        feature = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890[]\;\'" \
                  ",./~!@#$%^&*()_+{}|:\"<>?，。、；【】-=`！￥…（）—「」：《》？ "
        self.feature = {feature: index + 2 for index, feature in enumerate(feature)}
        self.feature["<PAD>"] = 0
        self.feature["<uncognized_feature>"] = 1
        self.n_feature = len(self.feature)
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.context_width = context_width
        self.neg_num = neg_num
        self.with_cuda = with_cuda
        self.cuda_devices = cuda_devices
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = [0]
        self.model = torch.nn.DataParallel(self.model_build(), device_ids=self.device).cuda()
        # self.model.to(self.device)
        self.name = "Uniparser"

    def get_name(self):
        return self.name

    def model_build(self):
        return Model(n_feature=self.n_feature,
                     emb_dim=self.emb_dim,
                     hidden_dim=self.hidden_dim,
                     dropout=self.dropout)

    def canonical_train(self,
                        data,
                        nr_epochs=1,
                        batch_size=5,
                        lr=0.002,
                        betas=(0.9, 0.999),
                        weight_decay=0.005,
                        contrastive_loss_weigh=0.01):
        train_dataloader = self.get_train_dataloader(data, batch_size=batch_size)
        model_opt = torch.optim.Adam(self.model.parameters(),
                                     lr=lr,
                                     betas=betas,
                                     weight_decay=weight_decay)
        # scheduler = ReduceLROnPlateau(optimizer=model_opt,mode='min',factor=0.5, min_lr=-1e-7, patience=10)
        scheduler = ExponentialLR(optimizer=model_opt, gamma=0.9)
        print("epoch1")
        for epoch in range(nr_epochs):
            self.model.train()
            self.run_epoch(train_dataloader, self.model, model_opt, scheduler, contrastive_loss_weigh)

    def contrastive_loss(self, contrastive_loss_weigh, prob, y_label, context, pos_context, neg_context):
        model_loss = nn.CrossEntropyLoss()
        prob_loss = model_loss(prob, y_label)

        pos_dis = torch.sum(torch.mul(context, pos_context), axis=1)
        neg_dis = torch.sum(torch.mul(context, neg_context), axis=2)
        pos_dis = torch.reshape(pos_dis, [pos_dis.shape[0], 1])
        neg_dis = torch.transpose(neg_dis, dim0=0, dim1=1)

        dis = torch.cat((pos_dis, neg_dis), dim=1)
        contras_label = torch.zeros(pos_dis.shape[0], dtype=torch.int64).to('cuda')
        contras_loss = model_loss(dis, contras_label)

        loss = prob_loss + contrastive_loss_weigh * contras_loss
        return loss

    def canonical_inference(self, data, update_pattern_dict=False, batch_size=5):
        return self.parse(data, update_pattern_dict=update_pattern_dict, batch_size=batch_size)

    def run_epoch(self, dataloader, model, model_opt, scheduler, contrastive_loss_weigh):
        total_loss = 0
        token_num = 0
        start = time.time()
        for i, batch in enumerate(dataloader):
            batch = Batch(batch)
            # prob = model.forward(batch.tokens_feature, batch.context_feature)
            _, context_vec, prob = model.forward(batch.tokens_feature, batch.context_feature)
            pos_context = model.module.context_vec(batch.positive_context_feature)
            neg_context = model.module.neg_context_vec(batch.negative_context_feature)
            loss = self.contrastive_loss(contrastive_loss_weigh,
                                         prob,
                                         batch.if_para,
                                         context_vec,
                                         pos_context,
                                         neg_context)

            model_opt.zero_grad()
            loss.backward()
            model_opt.step()
            scheduler.step()
            total_loss += loss * prob.shape[0]
            token_num += prob.shape[0]
            if i % 5 == 1:
                elapsed = time.time() - start
                print("Epoch Step: %d / %d Loss: %f Tokens per Sec: %f" %
                      (i, len(dataloader), total_loss / (token_num + 0.00001), token_num / (elapsed + 0.00001)))
                start = time.time()
                total_loss = 0
                token_num = 0
        return total_loss / (token_num + 0.00001)

    def run_test(self, test_dataloader):
        self.model.eval()
        with torch.no_grad():
            result = []
            start = time.time()
            for i, batch in enumerate(test_dataloader):
                batch = BatchTest(batch)
                _, context_vec, prob = self.model.forward(batch.tokens_feature, batch.context_feature)
                # prob = self.model.forward(batch.tokens_feature, batch.context_feature)
                pred_class = np.argsort(-prob.cpu().numpy(), axis=1)
                pred_class = [i[0] for i in pred_class]
                index = 0
                for j in batch.tokens_length:
                    result += [pred_class[index:index + j]]
                    index += j
                if i % 20 == 1:
                    elapsed = time.time() - start
                    print("Epoch Step: %d / %d Tokens per Sec: %f" %
                          (i, len(test_dataloader), prob.shape[0] / elapsed))
                    start = time.time()
        return result

    def parse(self, date, update_pattern_dict=True, batch_size=5):
        test_dataloader = self.get_test_dataloader(date, batch_size=batch_size)
        result = self.run_test(test_dataloader)
        return result

    def get_train_dataloader(self, train_data, batch_size=5):
        train_data = TrainDataset(train_data, self.feature, self.context_width, neg_num=self.neg_num)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size,
                                      collate_fn=self.collate_train_fn)
        return train_dataloader

    def get_test_dataloader(self, test_data, batch_size):
        test_data = TestDataset(test_data, self.feature, self.context_width)
        test_dataloader = DataLoader(test_data, batch_size=batch_size,
                                     collate_fn=self.collate_test_fn)
        return test_dataloader

    def collate_train_fn(self, batch):
        batch = list(zip(*batch))
        tokens_feature, context_feature, positive_context_feature, negative_context_feature, \
        if_para, tokens_length = [], [], [], [], [], []
        for i in range(len(batch[0])):
            tokens_feature += batch[0][i]
            context_feature += batch[1][i]
            positive_context_feature += batch[2][i]
            negative_context_feature += batch[3][i]
            if_para += batch[4][i]
            tokens_length += batch[5][i]
        tokens_feature = torch.tensor(np.array(tokens_feature), dtype=torch.float32)
        context_feature = torch.tensor(np.array(context_feature), dtype=torch.float32)
        positive_context_feature = torch.tensor(np.array(positive_context_feature), dtype=torch.float32)
        negative_context_feature = torch.tensor(np.array(negative_context_feature), dtype=torch.float32). \
            transpose(0, 1).contiguous()
        if_para = torch.tensor(if_para, dtype=torch.int64)
        tokens_length = torch.tensor(tokens_length, dtype=torch.int64)
        del batch
        return tokens_feature.to('cuda'), context_feature.to('cuda'), \
               positive_context_feature.to('cuda'), negative_context_feature.to('cuda'), \
               if_para.to('cuda'), tokens_length.to('cuda')

    def collate_test_fn(self, batch):
        batch = list(zip(*batch))
        tokens_feature, context_feature, tokens_length = [], [], []
        for i in range(len(batch[0])):
            tokens_feature += batch[0][i]
            context_feature += batch[1][i]
            tokens_length += batch[2][i]
        tokens_feature = torch.tensor(np.array(tokens_feature), dtype=torch.float32)
        context_feature = torch.tensor(np.array(context_feature), dtype=torch.float32)
        tokens_length = torch.tensor(np.array(tokens_length), dtype=torch.int64)
        del batch
        return tokens_feature.to('cuda'), context_feature.to('cuda'), tokens_length.to('cuda')


class Model(nn.Module):
    def __init__(self,
                 n_feature=100,
                 emb_dim=1000,
                 hidden_dim=100,
                 dropout=0.01,
                 k=3):
        super(Model, self).__init__()
        self.token_emb = nn.Sequential(nn.LayerNorm(normalized_shape=n_feature),
                                       nn.Linear(in_features=n_feature, out_features=emb_dim),
                                       nn.LayerNorm(normalized_shape=emb_dim)
                                       )

        self.lstm = nn.Sequential(
            nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim,
                    bidirectional=True, batch_first=True, dropout=dropout, num_layers=2),
        )
        self.prob_layer = nn.Sequential(
            nn.LayerNorm(normalized_shape=emb_dim + hidden_dim * 2 * k * 2),
            nn.Linear(in_features=emb_dim + hidden_dim * 2 * k * 2,
                      out_features=64),
            # nn.Linear(in_features=hidden_dim * 2 * k * 2,
            #           out_features=64),
            nn.Sigmoid(),
            nn.Linear(in_features=64,
                      out_features=2),
        )

    def forward(self, x_token, x_context):
        token_vector = self.token_emb(x_token)
        context_vector = self.context_vec(x_context)
        prob = torch.cat((token_vector, context_vector), dim=1)
        prob = self.prob_layer(prob)
        return token_vector, context_vector, prob


    def context_vec(self, context):
        context_token_vector = self.token_emb(context)
        context_vector, _ = self.lstm(context_token_vector)
        context_vector = torch.reshape(context_vector, (context_vector.shape[0], -1))
        return context_vector

    def neg_context_vec(self, neg_context):
        neg_num = neg_context.shape[0]
        context_num = neg_context.shape[1]
        neg_context = torch.reshape(neg_context, [neg_num * context_num,
                                                  neg_context.shape[2],
                                                  neg_context.shape[3]])
        neg_context_vector = self.context_vec(neg_context)
        neg_context_vector = torch.reshape(neg_context_vector, [neg_num,
                                                                context_num,
                                                                neg_context_vector.shape[1]])
        return neg_context_vector


class TestDataset(Dataset):
    def __init__(self, datas, feature, context_width):
        self.feature = feature
        self.context_width = context_width
        self.tokens_list = [i["tokens"] for i in datas]
        self.tokens_length = [len(i) for i in self.tokens_list]

    def __getitem__(self, item):
        tokens = self.tokens_list[item]
        context = self.get_context(tokens)
        tokens_length = [self.tokens_length[item]]
        tokens_feature = [self.get_vec(token) for token in tokens]
        context_feature = [[self.get_vec(j) for j in i] for i in context]
        return tokens_feature, context_feature, tokens_length

    def __len__(self):
        return len(self.tokens_list)

    def get_context(self, tokens):
        tokens_copy = ["<PAD>"] * self.context_width + tokens + ["PAD"] * self.context_width
        context = []
        for index in range(len(tokens)):
            context.append(tokens_copy[index:index + self.context_width] +
                           tokens_copy[index + self.context_width + 1:index + 1 + self.context_width * 2])
        return context

    def get_vec(self, token):
        vec = np.zeros(len(self.feature))
        if token == "<PAD>":
            vec[0] += 1
        for i in token:
            if i in self.feature:
                vec[self.feature[i]] += 1
            else:
                vec[self.feature["<uncognized_feature>"]] += 1
        # vec = self.norm_feature(vec)
        return vec

    @staticmethod
    def norm_feature(vec):
        vec = vec - np.mean(vec)
        vec = vec / (np.std(vec) + 0.0001)
        return vec


class TrainDataset(TestDataset):
    def __init__(self, datas, feature, context_width, neg_num):
        self.feature = feature
        self.context_width = context_width
        self.neg_num = neg_num
        self.tokens_list = [i["tokens"] for i in datas]
        self.tokens_length = [len(i) for i in self.tokens_list]
        self.if_para = [i["if_para"] for i in datas]
        self.cluster = self.cluster_datas()

    def __getitem__(self, item):
        tokens = self.tokens_list[item]
        context = self.get_context(tokens)
        if_para = self.if_para[item]
        tokens_length = [self.tokens_length[item]]
        positive_tokens, negative_tokens = self.get_constrate_data(self.cluster, tokens)
        positive_context = self.get_context(positive_tokens)
        negative_context = [self.get_context(i) for i in negative_tokens]
        # positive_context = [random.choice(positive_context) for i in context]
        negative_context = [[random.choice(j) for j in negative_context] for i in range(len(context))]

        tokens_feature = [self.get_vec(token) for token in tokens]
        context_feature = [[self.get_vec(j) for j in i] for i in context]
        positive_context_feature = [[self.get_vec(j) for j in i] for i in positive_context]
        negative_context_feature = [[[self.get_vec(k) for k in j]
                                     for j in i] for i in negative_context]

        return tokens_feature, context_feature, positive_context_feature, negative_context_feature, \
               if_para, tokens_length

    def __len__(self):
        return len(self.tokens_list)

    def cluster_datas(self):
        cluster = {}
        for i in self.tokens_list:
            symbol = str(len(i)) + "->" + i[0]
            if symbol in cluster:
                cluster[symbol].append(i)
            else:
                cluster[symbol] = []
                cluster[symbol].append(i)
        return cluster

    def get_constrate_data(self, cluster, tokens):
        symbol = str(len(tokens)) + "->" + tokens[0]
        positive_data = random.choice(cluster[symbol])
        negative_data = []
        while len(negative_data) < self.neg_num:
            neg_class = random.choice(list(cluster.keys()))
            if neg_class != symbol:
                negative_data.append(random.choice(cluster[neg_class]))
        return positive_data, negative_data


class Batch:
    def __init__(self, batch):
        self.tokens_feature = batch[0]
        self.context_feature = batch[1]
        self.positive_context_feature = batch[2]
        self.negative_context_feature = batch[3]
        self.if_para = batch[4]
        self.tokens_length = batch[5]


class BatchTest:
    def __init__(self, batch):
        self.tokens_feature = batch[0]
        self.context_feature = batch[1]
        self.tokens_length = batch[2]


if __name__ == "__main__":
    data = [{"tokens": ["Node", "001", "is", "unconnected"], "if_para": [0, 1, 0, 0]},
            {"tokens": ["Node", "001", "is", "unconnected"], "if_para": [0, 1, 0, 0]},
            {"tokens": ["Node", "001", "is", "unconnected"], "if_para": [0, 1, 0, 0]},
            {"tokens": ["Node", "001", "is", "unconnected"], "if_para": [0, 1, 0, 0]},
            {"tokens": ["Node", "001", "is", "unconnected"], "if_para": [0, 1, 0, 0]},
            {"tokens": ["Node", "001", "002", "is", "unconnected"], "if_para": [0, 1, 1, 0, 0]},
            {"tokens": ["Node", "001", "002", "is", "unconnected"], "if_para": [0, 1, 1, 0, 0]},
            {"tokens": ["Node", "001", "002", "is", "unconnected"], "if_para": [0, 1, 1, 0, 0]},
            {"tokens": ["Node", "001", "002", "is", "unconnected"], "if_para": [0, 1, 1, 0, 0]},
            {"tokens": ["Node", "001", "002", "is", "unconnected"], "if_para": [0, 1, 1, 0, 0]},
            ]
    parser = LogParserUniparser()
    parser.canonical_train(data, nr_epochs=5, batch_size=5)
    parser.canonical_inference(data)
    a = 1
