# coding: UTF-8
import os
import pickle as pkl
import torch
import numpy as np
from train_eval import init_network
from importlib import import_module
from utils import build_iterator, build_vocab

UNK, PAD = '<UNK>', '<PAD>'

# todo: 改成一个可以调用的方法，供UI调用，两个输入一个输出
if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    # todo ： 这里做成输入参数，输入为'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer里面的一个，可以减少几个
    model_name = 'TextCNN'  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer

    x = import_module('models.TextCNN')
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    # 读取vocab
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    # 读取模型
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    # 读取分类数据，转成映射表，用于最后预测结果的输出
    class_filepath = './THUCNews/data/class.txt'
    with open(class_filepath, 'r', encoding='utf-8') as file:
        classes = [line.strip() for line in file.readlines()]
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    label_map_reverse = {idx: cls for idx, cls in enumerate(classes)}
    print(label_map_reverse)
    # todo : 这个做成输入
    content = "古巴法国重开政治对话关系转暖"
    # 将输入内容格式化
    print("input:" + content)
    tokenizer = lambda x: [y for y in x]  # char-level
    token = tokenizer(content)
    seq_len = len(token)
    words_line = []
    if config.pad_size:
        pad_size = config.pad_size
    else:
        pad_size = 32
    if len(token) < config.pad_size:
        token.extend([PAD] * (pad_size - len(token)))
    else:
        token = token[:pad_size]
        seq_len = pad_size
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))
    contents = []
    contents.append((words_line, int(1), seq_len))
    test_iter = build_iterator(contents, config)


    with torch.no_grad():
        for texts, labels in test_iter:
            outputs = model(texts)

            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            predicted_class = label_map_reverse[predict[0]]
            # todo:这里可以将结果做成输出
            print(f"Predicted class: {predicted_class}")





