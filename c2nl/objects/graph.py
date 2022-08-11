import numpy as np
from c2nl.inputters.vocabulary import Vocabulary


class Graph(object):
    def __init__(self, _id=None):
        self._id = _id
        self._text = None
        self._edges = []
        self._tokens = []  # 图上所有节点label
        self._tokens_length = None
        self._backbone_sequence = []
        self.src_vocab = Vocabulary(no_special_token=True)  # required for Copy Attention
        self.code_tokens = []  # 图中叶子节点的label

    @property
    def id(self) -> str:
        return self._id

    @property
    def text(self) -> str:
        return self._text

    @property
    def tokens_length(self) -> str:
        return self._tokens_length

    @text.setter
    def text(self, param: str) -> None:
        self._text = param

    @property
    def edges(self) -> list:
        return self._edges

    @edges.setter
    def edges(self, param: list) -> None:
        assert isinstance(param, list)
        self._edges = param

    @property
    def tokens(self) -> list:
        return self._tokens

    @tokens.setter
    def tokens(self, param: list) -> None:
        assert isinstance(param, list)
        self._tokens_length = len(param)
        self._tokens = param

    def form_src_vocab(self) -> None:
        for i in range(len(self.backbone_sequence)):
            token = self.tokens[self.backbone_sequence[i]]
            self.src_vocab.add(token)
            self.code_tokens.append(token)

    @property
    def backbone_sequence(self) -> list:
        return self._backbone_sequence

    @backbone_sequence.setter
    def backbone_sequence(self, param: list) -> None:
        assert isinstance(param, list)
        for i in range(len(param)):
            token = self.tokens[param[i]]
            if token == "":  # for python dataset
                self.tokens[param[i]] = "pythonblank"
                token = "pythonblank"
            self.src_vocab.add(token)
            self.code_tokens.append(token)
        self._backbone_sequence = param

    def vectorize(self, word_dict, _type='word') -> list:
        if _type == 'word':
            return [word_dict[w] for w in self.tokens]
        elif _type == 'char':
            return [word_dict.word_to_char_ids(w).tolist() for w in self.tokens]
        else:
            assert False

    def code_vectorize(self, word_dict, _type='word') -> list:
        if _type == 'word':
            return [word_dict[w] for w in self.code_tokens]
        elif _type == 'char':
            return [word_dict.word_to_char_ids(w).tolist() for w in self.code_tokens]
        else:
            assert False

    def getmatrix(self, edge_type):
        matrix = np.zeros((self._tokens_length, self._tokens_length * len(edge_type) * 2))
        for edge in self.edges:
            if edge[0] in edge_type:
                e_type = edge_type[edge[0]]
                src_idx = edge[1]
                tgt_idx = edge[2]
                matrix[tgt_idx][e_type * self._tokens_length + src_idx] = 1
                matrix[src_idx][(e_type + len(edge_type)) * self._tokens_length + tgt_idx] = 1
        return matrix
