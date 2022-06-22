"""
Description:  Fastai - version 1.0.61; pytorch-transformers - version 1.2.0; transformers - version 3.4.0;
              tensorflow - version 2.3.0; urllib3 - version 1.25.10.
              Must run on GPU
"""

import torch
from transformers import LongformerModel, LongformerTokenizer
from fastai.text import *
from fastai.metrics import *
import torch.nn as nn
from datetime import date
from gensim.parsing.preprocessing import STOPWORDS
import re  # regex
import nltk.data


# Config class stores value of hyper-parameters and folder destinations for training
class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


class FastAiLongformerTokenizer(BaseTokenizer):
    """Wrapper around longformerTokenizer to be compatible with fastai"""
    def __init__(self, tokenizer: LongformerTokenizer, max_seq_len: int = 128, **kwargs):
        self._pretrained_tokenizer = tokenizer  # tokenizer is LongformerTokenizer
        self.max_seq_len = max_seq_len # the longest length of a sentence

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Adds longformer bos and eos tokens and limits the maximum sequence length"""
        # starting of the sequence is <s> and ending is </s>
        return ["<s>"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["</s>"]


# Setting up pre-processors
class LongformerTokenizeProcessor(TokenizeProcessor):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)


class LongformerNumericalizeProcessor(NumericalizeProcessor):
    def __init__(self, fastai_longformer_vocab, *args, **kwargs):
        super().__init__(*args, vocab=fastai_longformer_vocab, **kwargs)


# Creating a longformer specific DataBunch class. In side Databunch contains
# training and validation data. We adopted this approach from Fastai1v to quickly
# train data with learner. The class longformerDatabunch is inherited from Text DataBunch.
class LongformerDataBunch(TextDataBunch):
    "Create a `TextDataBunch` suitable for training longformer"
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', bs:int=64, val_bs:int=None, pad_idx=1,
               pad_first=True, device:torch.device=None, no_check:bool=False, backwards:bool=False,
               dl_tfms:Optional[Collection[Callable]]=None, **dl_kwargs) -> DataBunch:
        "Function that transform the `datasets` in a `DataBunch` for classification. Passes `**dl_kwargs` on to `DataLoader()`"
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        collate_fn = partial(pad_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=backwards)
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs)
        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True, **dl_kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            lengths = [len(t) for t in ds.x.items]
            sampler = SortSampler(ds.x, key=lengths.__getitem__)
            dataloaders.append(DataLoader(ds, batch_size=val_bs, sampler=sampler, **dl_kwargs))
        return cls(*dataloaders, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)


class LongformerTextList(TextList):
    _bunch = LongformerDataBunch
    _label_cls = TextList


# defining Longformer architecture
class CustomLongformerModel(nn.Module):
    def __init__(self,config):
        super(CustomLongformerModel, self).__init__()
        self.num_labels = config.num_labels  # get number of labels
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-large-4096', return_dict=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # best configuration is 0.05
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)  # if base model is use, hidden_size should 768

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.longformer(input_ids, token_type_ids, attention_mask)
        pooled_output = outputs.pooler_output
        # get logits value for classifier
        logits = self.classifier(pooled_output)
        return logits


def get_pretrained_model(model_path):
    config = Config(
        # change to your own path
        file_path="models",
        date=date.today().strftime('%Y%m%d'),
        seed=2020,
        longformer_model_name='allenai/longformer-large-4096',
        max_lr=1e-5,
        epochs=3,  # 2 hour for one epoch
        bs=4,  # larger than 16 can cause CUDA or OOM problems
        max_seq_len=2000,  # the max length of abstract is 491
        num_labels=100,  # number of labels is 100
        hidden_dropout_prob=.05,  # best configuration is 0.05
        hidden_size=1024,  # if base model is use, hidden_size should 768
        valid_pct=0.20,  # 20 percent of dataset used for validation
        start_tok="<s>",
        end_tok="</s>",
        text_column_name='abstract',
        target_column_name='label'
    )
    # define link to model
    config.model_path = f'Epoch_{config.epochs}_len_{config.max_seq_len}_{config.date}.pkl'
    # define link to prediction
    config.pred_path = f'{config.longformer_model_name}_Epoch_{config.epochs}_len_{config.max_seq_len}_{config.date}.csv'
    # define link to training dataset
    config.train_file_path = f'{config.file_path}/train_data_labels.csv'
    # define link to testing dataset
    config.test_file_path = f'{config.file_path}/test_data.csv'

    longformer_tok = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    # we will adopt tokenizer (BaseTokenizer) from fastai
    fastai_tokenizer = Tokenizer(tok_func=FastAiLongformerTokenizer(longformer_tok, max_seq_len=config.max_seq_len),
                                 pre_rules=[], post_rules=[])

    learn = load_learner(config.file_path,model_path)  # Load pretrained pickle model file

    return learn


def get_article_category_index(pretrained_model, article):
    result = pretrained_model.predict(article)
    label_index = int(result[1])
    return label_index


def get_sents_probs(pretrained_model, label_index, sents_list):
    predict_probs = []
    for i in range(len(sents_list)):
        result = pretrained_model.predict(sents_list[i])[2]
        prob = result[label_index].item()
        predict_probs.append((prob,i))  # [(prob, position), ()...()]
    return predict_probs


def get_top_sents(predict_probs, sents_list, percentage):
    #Sort by probability in decending order
    sorted_by_prob = sorted(predict_probs, key=lambda tup: tup[0],reverse=True) # [(prob, position), ()...()]
    top_probs = sorted_by_prob[:int(round(len(sents_list)*percentage))]
    # Sort by position
    sorted_by_position = sorted(top_probs, key=lambda tup: tup[1])
    top_sents_index = [s[1] for s in sorted_by_position]
    sorted_prob = [s[0] for s in sorted_by_position]
    filter_sents = [sents_list[i] for i in top_sents_index]

    top_sorted_sents = list(zip(filter_sents, sorted_prob, top_sents_index))
    return top_sorted_sents  #[(sent, prob, position)]


if __name__ == "__main__":
    f = open('scientific_paper_test/arxiv/arxiv_test_full.txt', 'r')
    file = f.readlines()
    f.close()
    text = file[0]
    # Split sentences
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    model = get_pretrained_model()

    sents_list = sent_detector.tokenize(text.strip())
    article_cate_index = get_article_category_index(model,text)
    sents_probs = get_sents_probs(model, article_cate_index, sents_list)
    filtered_sents = get_top_sents(sents_probs,sents_list,0.6)
    print(filtered_sents)
    print(len(filtered_sents))


