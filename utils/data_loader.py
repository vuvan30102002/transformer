import torch, torchtext
torchtext.disable_torchtext_deprecation_warning()
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer

class DataPrep:
    def __init__(self, tokenizer_en, tokenizer_de, init_token="<sos>", eos_token="<eos>"):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_de = tokenizer_de
        self.init_token   = init_token
        self.eos_token    = eos_token
        self.vocab_src = None
        self.vocab_trg = None

    # Tạo iterator dataset
    def make_dataset(self):
        train_iter, valid_iter, test_iter = Multi30k(split=('train', 'valid', 'test'), language_pair=('de', 'en'))
        return train_iter, valid_iter, test_iter

    # Hàm tạo vocab
    def build_vocab(self, train_iter, min_freq=2):
        def yield_tokens(data_iter, tokenizer, idx):
            for src, trg in data_iter:
                yield tokenizer(src if idx==0 else trg)
        
        # Xây vocab
        self.vocab_src = build_vocab_from_iterator(yield_tokens(train_iter, self.tokenizer_de, 0),
                                                   specials=[self.init_token, self.eos_token], min_freq=min_freq)
        self.vocab_trg = build_vocab_from_iterator(yield_tokens(train_iter, self.tokenizer_en, 1),
                                                   specials=[self.init_token, self.eos_token], min_freq=min_freq)
        self.vocab_src.set_default_index(self.vocab_src["<unk>"])
        self.vocab_trg.set_default_index(self.vocab_trg["<unk>"])

    # Chuyển text thành tensor và pad
    def tensorize_batch(self, batch, idx):
        sequences = []
        for src, trg in batch:
            text = src if idx==0 else trg
            tokens = [self.vocab_src[self.init_token]] + [self.vocab_src[t] for t in self.tokenizer_de(text)] + [self.vocab_src[self.eos_token]] if idx==0 \
                     else [self.vocab_trg[self.init_token]] + [self.vocab_trg[t] for t in self.tokenizer_en(text)] + [self.vocab_trg[self.eos_token]]
            sequences.append(torch.tensor(tokens, dtype=torch.long))
        return pad_sequence(sequences, batch_first=True, padding_value=0)

    # DataLoader mới
    def make_iter(self, dataset_iter, batch_size=32, device='cpu'):
        # Chuyển iterator thành list
        dataset = list(dataset_iter)
        
        def collate_fn(batch):
            src_batch = self.tensorize_batch(batch, idx=0).to(device)
            trg_batch = self.tensorize_batch(batch, idx=1).to(device)
            return src_batch, trg_batch

        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


print("alo")