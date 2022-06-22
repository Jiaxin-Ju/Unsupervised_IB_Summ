"""
Description:
"""


class BeamNode:
    def __init__(self, score, path, position):
        self.score = score
        self.path = path
        self.position = position

    def get_score(self):
        return self.score

    def get_path(self):
        return self.path

    def get_position(self):
        return self.position


class BeamSearch:
    def __init__(self, prob_matrix, first_node_pos=0, beam_size=5, min_num_sents=15):
        self.prob_matrix = prob_matrix
        self.beam_size = beam_size
        self.first_node_pos = first_node_pos
        self.first_node = BeamNode(score=1, path=[self.first_node_pos], position=self.first_node_pos)
        self.min_num_sents = min_num_sents
        self.last_node_pos = self.prob_matrix.shape[0] - 1

    def get_top_prob_and_pos(self, parent_node):
        """
        Get top prob for each node
        """
        position = parent_node.get_position()
        prob_list = self.prob_matrix[position].tolist()
        sent_pos = list(range(len(prob_list)))
        prob_info = list(zip(prob_list, sent_pos))  # [(prob, position)]

        prob_info = prob_info[(position + 1):]

        exit_path_len = len(parent_node.get_path())

        if exit_path_len < self.min_num_sents:  # must satisify minimum length
            max_sent = len(prob_list) - (self.min_num_sents - exit_path_len)
            sent_info = [s[1] for s in prob_info]
            max_index = sent_info.index(max_sent)
            prob_info = prob_info[:max_index]

        sorted_by_prob = sorted(prob_info, key=lambda tup: tup[0], reverse=True)  # descending order
        top_n = sorted_by_prob[:self.beam_size]
        return top_n  # first N [(prob, position)]

    def create_top_n_node(self, parent_node, top_n_info):
        """
        Create top N nodes for each parent node
        """
        node_list = []
        parent_path = parent_node.get_path()
        for info in top_n_info:
            node_score = parent_node.score * info[0]
            node_path = parent_path + [info[1]]
            node = BeamNode(node_score, node_path, info[1])
            node_list.append(node)
        return node_list

    def update_parent_node(self, parent_nodes_list):
        node_info = []
        for node in parent_nodes_list:
            if node.get_position() != self.last_node_pos:  # It is not the last node
                top_n = self.get_top_prob_and_pos(node)
                node_list = self.create_top_n_node(node, top_n)
                for n in node_list:
                    node_info.append((n, n.get_score()))  # [(node, score)]
            else:
                node_info.append((node, node.get_score()))
        # Sort by descending order
        new_parent_nodes = sorted(node_info, key=lambda tup: tup[1], reverse=True)[:self.beam_size]
        new_parent_nodes_list = [n[0] for n in new_parent_nodes]
        return new_parent_nodes_list  # new five parent nodes

    def is_stop(self, updated_nodes):
        i = 0
        for node in updated_nodes:
            if node.get_position() == self.last_node_pos:
                i += 1
        if i == self.beam_size:
            return True
        else:
            return False

    def run_beam_search(self):
        first_search = self.get_top_prob_and_pos(self.first_node)  # run root node
        first_top_node = self.create_top_n_node(self.first_node, first_search)  # first 5 parent nodes

        updated_nodes = self.update_parent_node(first_top_node)  # select top 5 nodes from 25 nodes

        is_stop = False
        while not is_stop:
            updated_nodes = self.update_parent_node(updated_nodes)
            is_stop = self.is_stop(updated_nodes)
        return updated_nodes

    def get_best_path_and_score(self, final_nodes):
        node_info = [(node, node.get_score()) for node in final_nodes]
        best_node = sorted(node_info, key=lambda tup: tup[1], reverse=True)[0][0]
        best_path = best_node.get_path()
        best_score = best_node.get_score()
        best_path_info = (best_path, best_score)
        return best_path_info


