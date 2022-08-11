import gzip
import codecs
import json
import pickle
from typing import Any, Iterator, Callable, Iterable, List

import xmltodict

import re
from collections import defaultdict


def load_xml_gz(filename: str, func: Callable, depth: int) -> Any:
    with gzip.open(filename) as f:
        return xmltodict.parse(f, item_depth=depth, item_callback=func)

def load_xml(filename: str, func: Callable, depth: int) -> Any:
    with open(filename, 'rb') as f:
        return xmltodict.parse(f, item_depth=depth, item_callback=func)

def save_pickle_gz(data: Any, filename: str) -> None:
    with gzip.GzipFile(filename, 'wb') as outfile:
        pickle.dump(data, outfile)

def iteratate_jsonl_gz(filename: str) -> Iterator[Any]:
    reader = codecs.getreader('utf-8')
    with gzip.open(filename) as f:
        for line in reader(f):
            yield json.loads(line)

def save_jsonl_gz(filename:str, data: Iterable[Any])-> None:
    with gzip.GzipFile(filename, 'wb') as out_file:
        writer = codecs.getwriter('utf-8')
        for element in data:
            writer(out_file).write(json.dumps(element))
            writer(out_file).write('\n')

def load_gz_per_line(filename:str)-> Iterator[str]:
    reader = codecs.getreader('utf-8')
    with gzip.open(filename) as f:
        yield from reader(f)


def subtokenizer(identifier: str)-> List[str]:
    # Tokenizes code identifiers
    splitter_regex = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

    identifiers = re.split('[._\-]', identifier)
    subtoken_list = []

    for identifier in identifiers:
        matches = splitter_regex.finditer(identifier)
        for subtoken in [m.group(0) for m in matches]:
            subtoken_list.append(subtoken)

    return subtoken_list


def extract_path(edges, edge_type):
    next_node_dict = defaultdict(lambda: None)
    for edge in edges:
        if edge[0] == edge_type:
            next_node_dict[edge[1]] = edge[2]
            
    start_node_set = set(next_node_dict.keys()) - set(next_node_dict.values()) 
    curr_node = start_node_set.pop()
    path = []
    while curr_node is not None:
        path.append(curr_node)
        curr_node = next_node_dict[curr_node]
    return path
