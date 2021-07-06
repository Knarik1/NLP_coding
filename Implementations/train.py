import math

import torch
from torch import Tensor
import torch.nn as nn
import torchtext

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from typing import Iterable, List

import Seq2SeqTransformer


if __name__ == '__main__':
    """
     Used sources
        https://pytorch.org/tutorials/beginner/translation_transformer.html
        
    """
    print(torch.__version__)  # 1.9.0+cu102
    print(torchtext.__version__)  # 0.10.0

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    SRC_LANGUAGE = 'de'
    TGT_LANGUAGE = 'en'

    # Place-holders
    token_transform = {}
    vocab_transform = {}


    # Create source and target language tokenizer. Make sure to install the dependencies.
    # pip install -U spacy
    # python -m spacy download en_core_web_sm
    # python -m spacy download de_core_news_sm
    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
    token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')


    # helper function to yield list of tokens
    def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
        language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

        for data_sample in data_iter:
            yield token_transform[language](data_sample[language_index[language]])

    # Define special symbols and indices
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    # Make sure the tokens are in order of their indices to properly insert them in vocab
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        # Training data Iterator
        train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)

    # Set UNK_IDX as the default index. This index is returned when the token is not found.
    # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)


    torch.manual_seed(0)

    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
