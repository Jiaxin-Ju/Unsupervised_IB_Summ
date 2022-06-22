"""
Description:
"""

import nltk.data
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import math
from transformers import *
from extract_keyphrases import KeyphrasesExtraction
#from article_category import *
from beam_search import *


class MultiViewInfoBottle:
    def __init__(self, lm_model, lm_tokenizer, next_sent_model,ita=1,num_local=10, alpha=1, beta=1,filter_with_threshold=False,is_avg=True):
        self.num_local = num_local
        self.lm_model = lm_model
        self.lm_tokenizer = lm_tokenizer
        self.is_avg = is_avg
        self.filter_with_threshold = filter_with_threshold
        self.ita = ita
        self.alpha = alpha
        self.beta = beta
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.next_sent_model = next_sent_model
        #self.pretrained_longformer = get_pretrained_model()

    def check_sent_length(self, sent):
        sent_tokenizer = RegexpTokenizer(r'\w+')
        sent_no_punc = sent_tokenizer.tokenize(sent)
        if 5 < len(sent_no_punc) < 80:
            return True
        else:
            return False

    def split_sentences(self, article):
        # Split sentences
        sent_list = self.sent_detector.tokenize(article.strip())
        position = list(range(0, len(sent_list)))
        sent_info_list = list(zip(sent_list, position))  # [(sent, position), ()...()]
        return sent_info_list

    def extract_keyphrases(self, article):
        # Get top n keyphrases
        rake = KeyphrasesExtraction()
        extracted_key_phrases = rake.process(self.num_local, article)
        return extracted_key_phrases

    def calculate_score(self, global_sents_info, article):
        sents_list = [s for s in global_sents_info if self.check_sent_length(s[0])]  # satisfy length constrain
        filtered_global_sents = [s[0] for s in sents_list]
        filtered_global_probs = [s[1] for s in sents_list]
        filtered_global_pos = [int(s[2]) for s in sents_list]
        print('pos type',type(filtered_global_pos[0]))
        # Similarity matrix with local labels
        local_sim_matrix = self.filter_local_sents(sents_list, article)

        # Calculate score
        scores = []
        for i in range(len(filtered_global_probs)):
            local_score = sum([p * math.log(p) for p in local_sim_matrix[i] if p > 0])
            global_score = filtered_global_probs[i]*math.log(filtered_global_probs[i])
            score = self.alpha * global_score + self.beta * local_score
            scores.append(score)
        sents_info = list(zip(filtered_global_sents, filtered_global_pos, scores))
        sorted_by_score = sorted(sents_info, key=lambda tup: tup[2], reverse=True)
        return sorted_by_score  # [(sents,pos,score)]

    def get_top_N_sents(self, sorted_by_score_sents, n):
        if len(sorted_by_score_sents) < n:
            # Sort sentence based on position
            sorted_by_pos_sents = sorted(sorted_by_score_sents, key=lambda tup: tup[1])  # [(sents,pos,score)]
        else:
            sorted_by_pos_sents = sorted(sorted_by_score_sents[:n], key=lambda tup: tup[1])  # [(sents,pos,score)]
        sorted_sents = [s[0] for s in sorted_by_pos_sents]
        return sorted_sents  # return sentences only

    def beam_search_NSP(self, sorted_sents, min_num_sents):
        if len(sorted_sents) < (min_num_sents+6):
            return list(range(len(sorted_sents)))
        else:
            print('Calculate prob matrix with shape:',len(sorted_sents))
            prob_matrix = self.calculate_NSP_prob(sorted_sents)
            
            print('start beam search ...')
            best_path_info = []
            for i in range(5):  # search for first 5 sentences
                print('for',i,'sentence')
                beam_search = BeamSearch(prob_matrix, first_node_pos=i, beam_size=5,min_num_sents=min_num_sents)
                nodes_list = beam_search.run_beam_search()
                path_info = beam_search.get_best_path_and_score(nodes_list)  # sentence index list
                best_path_info.append(path_info)  #[(path, score)]
                print('the best path info', path_info)

            best_path = sorted(best_path_info, key=lambda tup: tup[1], reverse=True)[0][0]
            print('========================')
            print('Best path: ', best_path)
            print('========================')
            return best_path  # path index

    def calculate_NSP_prob(self, sorted_sents):
        
        # Calculate prob matrix
        dim = len(sorted_sents)
        prob_matrix = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(i + 1, dim):
                prob = self.get_next_sent_prob(self.next_sent_model, self.lm_tokenizer,
                                               sorted_sents[i], sorted_sents[j])
                prob_matrix[i][j] = prob
        return prob_matrix

    def greedy_search_NSP(self, sorted_sents):
        if len(sorted_sents) < 21:
            return list(range(len(sorted_sents)))
        else:
            print('Calculate prob matrix with shape: ', len(sorted_sents))
            prob_matrix = self.calculate_NSP_prob(sorted_sents)
            
            print('start Greedy search ...')
            sents_index_list = list(range(len(sorted_sents)))

            wind_size = 3  # window size
            is_last = False  # check if this is the last sentence set
            i = 0
            while i < len(sents_index_list):
                # Split sentence list
                if (i + wind_size) < len(sents_index_list):
                    slice_set = sents_index_list[i:i + wind_size]
                else:
                    slice_set = sents_index_list[i:]  # the last sentence set
                    is_last = True
                if len(slice_set) == wind_size:
                    # All possible sentence combination probabilities
                    first_secd_prob = prob_matrix[slice_set[0]][slice_set[1]]
                    first_third_prob = prob_matrix[slice_set[0]][slice_set[2]]
                    secd_third_prob = prob_matrix[slice_set[1]][slice_set[2]]
                    avg_all_prob = prob_matrix[slice_set[0]][slice_set[1]] * prob_matrix[slice_set[1]][slice_set[2]]
                    prob_list = [first_secd_prob, first_third_prob, secd_third_prob, avg_all_prob]

                    # Find the sentence combination with the maximum probability
                    max_index = prob_list.index(max(prob_list))
                    if max_index == 0:
                        sents_index_list.remove(slice_set[2])
                    elif max_index == 1:
                        sents_index_list.remove(slice_set[1])
                    elif max_index == 2:
                        sents_index_list.remove(slice_set[0])
                        i -= 1
                if is_last:
                    break
                i += 1
            print('Final index list:', sents_index_list)
            return sents_index_list

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def get_next_sent_prob(self, model, tokenizer, sent1, sent2):
        encoding = tokenizer(sent1, sent2, return_tensors='pt')
        outputs = model(**encoding, return_dict=True)
        logits = outputs.logits
        logits = list(logits.detach().numpy()[0])
        next_sent_prob = self.softmax(logits)[0]
        return next_sent_prob

    def filter_global_sents(self, article):
        """
        This function is to filter sentences based on the global label
        :return: a list of filtered sentences with its prob and position
        """
        sent_info_list = self.split_sentences(article)
        sent_list = [sent[0] for sent in sent_info_list]  # select sentences

        article_cate_index = get_article_category_index(self.pretrained_longformer, article)
        sents_probs = get_sents_probs(self.pretrained_longformer, article_cate_index, sent_list)
        filtered_sents = get_top_sents(sents_probs, sent_list, self.ita)
        return filtered_sents  # [(sent, prob, position)]

    def filter_local_sents(self, sents_info_list, article):   # sents_list => filtered_sents/full_sents
        sents_list = [s[0] for s in sents_info_list if self.check_sent_length(s[0])]  # satisfy length constrain
        sent_vectors = self.get_sents_embeddings(sents_list)
        phrases_vectors = self.get_local_label_vectors(article)
        sim_matrix = np.zeros((len(sents_list), self.num_local))
        dim = len(sent_vectors[0])
        for i in range(len(sent_vectors)):
            for j in range(len(phrases_vectors)):
                sim_matrix[i][j] = cosine_similarity(sent_vectors[i].reshape(1, dim),
                                                     phrases_vectors[j].reshape(1, dim))[0, 0]
        if self.filter_with_threshold is False:  # Return for calculate score
            return sim_matrix
        else:   # For Ablation study
            filtered_sents = []
            threshold = np.median(sim_matrix)  # Set the mean vaule as the similarity threshold
            n = int(round((sim_matrix > threshold).sum() / len(sents_list)))
            for i in range(len(sents_list)):
                arr = sim_matrix[i]
                # sorted_index
                sorted_index_array = np.argsort(arr)
                # sorted array in ascending order
                sorted_array = arr[sorted_index_array]
                # Sentence should have higher similarity with at least n keyphrases
                if sorted_array[-n] >= threshold:  # only check the smallest one
                    filtered_sents.append(sents_list[i])
        filtered_sents_info = [s for s in sents_info_list if s[0] in filtered_sents]
        return filtered_sents_info

    # get language model embedding from Scibert
    def get_lm_embedding(self, sentence):
        lm_tokenizer = self.lm_tokenizer
        lm_model = self.lm_model
        encoded_dict = lm_tokenizer.encode_plus(
            sentence,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            return_tensors='pt',  # Return pytorch tensors.
        )
        token_ids_list = encoded_dict['input_ids'].tolist()[0]
        if len(token_ids_list) > 512:
            input_ids_list = [token_ids_list[:512]]
            input_ids = torch.tensor(input_ids_list)
        else:
            input_ids = encoded_dict['input_ids']
        lm_model.eval()

        with torch.no_grad():
            outputs = lm_model(input_ids)
            hidden_states = outputs[2]  # the third item will be the hidden states from all layers.

        if self.is_avg:
            token_vecs = hidden_states[-2][0]  # Select the embeddings from the second to last layer.
            # Calculate the average of all token vectors.
            sentence_embedding = torch.mean(token_vecs, dim=0)
        else:
            # [CLS] token embedding as the sentence representation
            token_embeddings = torch.stack(hidden_states, dim=0)  # [# layers, # batches, # tokens, # features]
            token_embeddings = torch.squeeze(token_embeddings, dim=1)  # Remove dimension 1, the "batches".
            token_embeddings = token_embeddings.permute(1, 0, 2)  # Swap dimensions 0 and 1.
            cls_token = token_embeddings[0]
            sentence_embedding = torch.sum(cls_token[-4:], dim=0)  # Summing the last four layers.

        sentence_embedding = sentence_embedding.detach().numpy()  # Convert to numpy array.
        return sentence_embedding

    def get_local_label_vectors(self, article):  # extract top n key phrases
        key_phrases = self.extract_keyphrases(article)
        phrases_vectors = [self.get_lm_embedding(k) for k in key_phrases]
        return phrases_vectors

    def get_sents_embeddings(self, sents_list):  # based on sentences filtered by global label
        sent_vectors = [self.get_lm_embedding(s) for s in sents_list]
        return sent_vectors


