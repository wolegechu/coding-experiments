import unicodedata
import re
from lang import Lang
import random
import codecs

import torch
from torch.autograd import Variable

MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

def unicode_to_ascii(s):
    '''Turn a Unicode string to plain ASCII

    Thanks to http://stackoverflow.com/a/518232/2809427
    '''

    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    '''Lowercase, trim, and remove non-letter characters
    '''

    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_langs(lang1, lang2, reverse=False):
    print "Reading lines..."

    # Read the file and split into lines
    lines = codecs.open('data/%s-%s.txt' % (lang1, lang2), 'r', 'utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

eng_prefixes = (
    "i am", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print "Read %s sentence pairs" % len(pairs)
    pairs = filter_pairs(pairs)
    print "Trimmed to %s sentence pairs" %len(pairs)
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print "Counted words:"
    print input_lang.name, input_lang.n_words
    print output_lang.name, output_lang.n_words
    return input_lang, output_lang, pairs

def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def variable_from_pair(input_lang, out_put_lang, pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(out_put_lang, pair[1])
    return (input_variable, target_variable)
