import logging
import random
import string
from collections import Counter
from tqdm import tqdm

from c2nl.objects import Code, Summary, Graph
from c2nl.inputters.vocabulary import Vocabulary, UnicodeCharsVocabulary
from c2nl.inputters.constants import BOS_WORD, EOS_WORD, PAD_WORD, \
    UNK_WORD, TOKEN_TYPE_MAP, AST_TYPE_MAP, DATA_LANG_MAP, LANG_ID_MAP
from c2nl.utils.misc import count_file_lines, count_gz_file_lines

# by zhou
import codecs
from typing import Iterator, Any
import gzip
import json

# import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def generate_random_string(N=8):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N))


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------
# node_num = []
def process_examples(lang_id,
                     source,
                     source_tag,
                     target,
                     graph,
                     max_src_len,
                     max_tgt_len,
                     code_tag_type,
                     uncase=False,
                     test_split=True):
    source = source.lower() if uncase else source
    code_tokens = source.split()
    code_type = []
    if source_tag is not None:
        code_type = source_tag.split()
        if len(code_tokens) != len(code_type):
            return None

    code_tokens = code_tokens[:max_src_len]
    code_type = code_type[:max_src_len]
    if len(code_tokens) == 0:
        return None

    TAG_TYPE_MAP = TOKEN_TYPE_MAP if \
        code_tag_type == 'subtoken' else AST_TYPE_MAP
    code = Code()
    code.text = source
    code.language = lang_id
    code.tokens = code_tokens
    code.type = [TAG_TYPE_MAP.get(ct, 1) for ct in code_type]
    if code_tag_type != 'subtoken':
        code.mask = [1 if ct == 'N' else 0 for ct in code_type]

    if target is not None:
        summ = target.lower() if uncase else target
        summ_tokens = summ.split()
        if not test_split:
            summ_tokens = summ_tokens[:max_tgt_len]
        if len(summ_tokens) == 0:
            return None
        summary = Summary()
        summary.text = ' '.join(summ_tokens)
        summary.tokens = summ_tokens
        summary.prepend_token(BOS_WORD)
        summary.append_token(EOS_WORD)
    else:
        summary = None

    if graph is not None:
        gnn = Graph()
        edges = graph['edges']
        nodes = graph['node_labels']
        backbone_sequence = graph['backbone_sequence'][:max_src_len]
        # nodes_lower = ' '.join(nodes).lower().split()
        nodes_lower = [node.lower() for node in nodes]
        if len(nodes) >= 600 or len(edges) == len(nodes_lower) == len(backbone_sequence) == 0:
            nodes_lower = code_tokens
            backbone_sequence = [i for i in range(len(nodes_lower))]
            edges = addNextTokenEdge(len(nodes_lower))

        # global node_num
        # node_num.append(len(nodes_lower))
        gnn.edges = edges
        gnn.tokens = nodes_lower
        gnn.backbone_sequence = backbone_sequence
    else:
        gnn = None

    example = dict()
    example['code'] = code
    example['summary'] = summary
    example['gnn'] = gnn
    return example


def addNextTokenEdge(num_edge):
    edges = []
    for i in range(num_edge - 1):
        edge = ['NextToken', i, i + 1]
        edges.append(edge)
    return edges


def load_data(args, filenames, max_examples=-1, dataset_name='java',
              test_split=False):
    """Load examples from preprocessed file. One example per line, JSON encoded."""

    with open(filenames['src'], encoding='utf-8') as f:
        sources = [line.strip() for line in
                   tqdm(f, total=count_file_lines(filenames['src']))]

    if filenames['tgt'] is not None:
        with open(filenames['tgt'], encoding='utf-8') as f:
            targets = [line.strip() for line in
                       tqdm(f, total=count_file_lines(filenames['tgt']))]
    else:
        targets = [None] * len(sources)

    if filenames['src_tag'] is not None:
        with open(filenames['src_tag'], encoding='utf-8') as f:
            source_tags = [line.strip() for line in
                           tqdm(f, total=count_file_lines(filenames['src_tag']))]
    else:
        source_tags = [None] * len(sources)

    # by zhou
    if filenames['gnn'] is not None:
        allgraphs = iteratate_jsonl_gz(filenames['gnn'])
        graphs = [line for line in
                  tqdm(allgraphs, total=count_gz_file_lines(filenames['gnn']))]
    else:
        graphs = [None] * len(sources)
    # end zhou

    assert len(sources) == len(source_tags) == len(targets) == len(graphs)

    examples = []
    for src, src_tag, tgt, graph in tqdm(zip(sources, source_tags, targets, graphs),
                                         total=len(sources)):
        if dataset_name in ['java', 'python']:
            _ex = process_examples(LANG_ID_MAP[DATA_LANG_MAP[dataset_name]],
                                   src,
                                   src_tag,
                                   tgt,
                                   graph,
                                   args.max_src_len,
                                   args.max_tgt_len,
                                   args.code_tag_type,
                                   uncase=args.uncase,
                                   test_split=test_split)
            if _ex is not None:
                examples.append(_ex)

        if max_examples != -1 and len(examples) > max_examples:
            break
    # global node_num
    # print(sum(node_num))
    # plt.ylim(0,2000)
    # plt.bar(range(0, len(node_num)), node_num)
    # plt.savefig(str(sum(node_num)) + '.jpg')
    return examples


# ------------------------------------------------------------------------------
# Dictionary building
# ------------------------------------------------------------------------------


def index_embedding_words(embedding_file):
    """Put all the words in embedding_file into a set."""
    words = set()
    with open(embedding_file) as f:
        for line in tqdm(f, total=count_file_lines(embedding_file)):
            w = Vocabulary.normalize(line.rstrip().split(' ')[0])
            words.add(w)

    words.update([BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD])
    return words


def load_words(args, examples, fields, dict_size=None):
    """Iterate and index all the words in examples (documents + questions)."""

    def _insert(iterable):
        words = []
        for w in iterable:
            w = Vocabulary.normalize(w)
            words.append(w)
        word_count.update(words)

    word_count = Counter()
    for ex in tqdm(examples):
        for field in fields:
            _insert(ex[field].tokens)

    # -2 to reserve spots for PAD and UNK token
    dict_size = dict_size - 2 if dict_size and dict_size > 2 else dict_size
    most_common = word_count.most_common(dict_size)
    words = set(word for word, _ in most_common)
    return words


def build_word_dict(args, examples, fields, dict_size=None,
                    no_special_token=False):
    """Return a dictionary from question and document words in
    provided examples.
    """
    word_dict = Vocabulary(no_special_token)
    for w in load_words(args, examples, fields, dict_size):
        word_dict.add(w)
    return word_dict


def build_word_and_char_dict(args, examples, fields, dict_size=None,
                             no_special_token=False):
    """Return a dictionary from question and document words in
    provided examples.
    """
    words = load_words(args, examples, fields, dict_size)
    dictioanry = UnicodeCharsVocabulary(words,
                                        args.max_characters_per_token,
                                        no_special_token)
    return dictioanry


def top_summary_words(args, examples, word_dict):
    """Count and return the most common question words in provided examples."""
    word_count = Counter()
    for ex in examples:
        for w in ex['summary'].tokens:
            w = Vocabulary.normalize(w)
            if w in word_dict:
                word_count.update([w])
    return word_count.most_common(args.tune_partial)


# by zhou
def iteratate_jsonl_gz(filename: str) -> Iterator[Any]:
    reader = codecs.getreader('utf-8')
    with gzip.open(filename) as f:
        for line in reader(f):
            yield json.loads(line)
