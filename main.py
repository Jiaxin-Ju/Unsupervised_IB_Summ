"""
Description:
"""

import argparse
import json
import os
import nltk.data
from transformers import *
from multiViewInfoBottle import MultiViewInfoBottle
from article_category import *
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def read_arguments():
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="covid19_test")
    parser.add_argument("--input_file", type=str, default="covid_test_1.txt")
    parser.add_argument("--output_path", type=str, default="test_covid_summ_results")
    parser.add_argument("--output_file", type=str, default="covid_beam_result_1.txt")
    parser.add_argument("--num_local", type=int, default=10)  # the number of local labels
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=1)

    parser.add_argument("--is_avg", type=bool, default=True)  # sentence embedding: avg / [CLS]
    parser.add_argument("--top_n", type=int, default=50)  # Maximum number of sentences
    parser.add_argument("--min_num_sents", type=int, default=10)  # Minimum number of sentences


    parser.add_argument('--model_path',type=str,default='best_model_rerun.pkl')
    return parser.parse_args()

def run_multiView_infoBottle(multi_view_IB, articles, output_file_name, top_n, min_num_sents, file_path, model):


    for i in range(len(articles)):
        print('Starting article {}'.format(str(i)))
        print('Computing global label scores for each sentence')
        ## GLOBAL LABELS compute for each article 
        article_cate_index = get_article_category_index(model,articles[i])
        sents = sent_detector.tokenize(articles[i])
        sents_probs_tuple = get_sents_probs(model, article_cate_index, sents)
        sents_probs_dict = {k:v for v,k in sents_probs_tuple}
        index_list = list(range(len(sents)))
        output_sents = [sents[int(m)] for m in index_list]
        sents_probs = [sents_probs_dict[int(m)] for m in index_list]
        global_sents_info = list(zip(output_sents, sents_probs, index_list))  # [(sent,prob,position),()...()]

        ## sort based on global labels
        sorted_by_score_sents = multi_view_IB.calculate_score(global_sents_info, articles[i])  # [(sent,pos,score)]
        print('sent scores are calculated')

        top_n_sents = multi_view_IB.get_top_N_sents(sorted_by_score_sents, top_n)  # top n sorted by position
        print('Start Beam search ...')
        filtered_sent_index = multi_view_IB.beam_search_NSP(top_n_sents, min_num_sents)
        filtered_sents = [top_n_sents[i] for i in filtered_sent_index]
        output_text = " ".join(filtered_sents)
        with open(output_file_name, 'a+', encoding='utf-8') as f:
            f.write(output_text + '\n')
        print('Finished article {}!'.format(str(i)))

    print('Done')


if __name__ == "__main__":
    # read arguments
    args = read_arguments()
    input_path = args.input_path
    input_file = args.input_file
    output_path = args.output_path
    output_file = args.output_file
    num_local = args.num_local
    alpha = args.alpha
    beta = args.beta
    is_avg = args.is_avg
    top_n = args.top_n
    min_num_sents = args.min_num_sents
    print("arguments", args)

    # Read dataset
    input_file_path = os.path.join(input_path, input_file)
    file = open(input_file_path, 'r')
    input_files = file.readlines()
    file.close()

    # Output file path
    output_file_path = os.path.join(output_path, output_file)

    # Global label
    model_path = args.model_path
    model = get_pretrained_model(model_path)

    # language model - SciBERT
    scibert_path = "allenai/scibert_scivocab_uncased"
    scibert_tokenizer = AutoTokenizer.from_pretrained(scibert_path)
    scibert_model = BertModel.from_pretrained(scibert_path, output_hidden_states=True)
    next_sent_model = BertForNextSentencePrediction.from_pretrained('allenai/scibert_scivocab_uncased',
                                                                         output_hidden_states=False)

    # Initialise an instance
    multi_view_IB = MultiViewInfoBottle(scibert_model, scibert_tokenizer, next_sent_model,
                                        num_local=num_local, alpha=alpha, beta=beta, is_avg=is_avg)



    # Model: local + Global + Next sentence prediction  # select sentences with higher score
    run_multiView_infoBottle(multi_view_IB, input_files, output_file_path, top_n, min_num_sents,
                             input_file_path, model)

