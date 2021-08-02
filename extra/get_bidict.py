# get_bidict.py
import numpy as np
import sys

# SORBIAN_CHARS = "AaBbCcČčĆćDdEeĚěFfGgHhIiJjKkŁłLlMmNnŃńOoÓóPpRrŘřŔŕSsŠšŚśTtUuWwYyZzŽžŹź"

def read_vocab(vocab_file):
    vocab = {}
    total_freq = 0
    with open(vocab_file, 'r', encoding='utf8') as file:
        for line in file:
            word, freq = line.split()
            vocab[word] = int(freq)
            total_freq += int(freq)
    return vocab, total_freq


def read_embs(emb_file):
    """ Reads embedding file 
        Returns: word to index, index to word and embedding matrix
    """
    idx2word = []
    word2idx = {}
    embs = []
    with open(emb_file, 'r', encoding='utf8') as file:
        idx = 0
        for line in file:
            split = line.split()
            if len(split) == 2:
                continue
            word = split[0]
            idx2word.append(word)
            word2idx[word] = idx

            vec = np.array([float(s) for s in split[1:]])
            embs.append(vec)
            idx += 1

    embs = np.stack(embs)
    return word2idx, idx2word, embs

def cossim(a, b):
    return (a @ b)/(np.linalg.norm(a) * np.linalg.norm(b))

def write_bidict(src_embs, tgt_embs, src_idx2word, tgt_idx2word, out_file, vocab=None, total_freq=None):
    same = 0
    with open(out_file, 'w', encoding='utf8') as file:
        for i in range(len(src_embs)):
            src_word = src_idx2word[i]
            if src_word == "</s>":
                continue
            if (vocab is not None and src_word in vocab):
                if vocab[src_word] / float(total_freq) > 0.00001: 
                    continue
                else:
                    print(src_word, vocab[src_word], flush=True)

            cossims = cossim(tgt_embs, src_embs[i])
            closest_idx = np.argmax(cossims)
            closest_word = tgt_idx2word[closest_idx]
            if closest_word == src_word:
                same += 1
            file.write("\t".join([src_word, closest_word, str(cossims[closest_idx])]) + "\n")
    print("SAME:", same)

if __name__ == "__main__":
    src_word2idx, src_idx2word, src_embs = read_embs(sys.argv[1])
    tgt_word2idx, tgt_idx2word, tgt_embs = read_embs(sys.argv[2])
    vocab = None
    total_freq = None
    if len(sys.argv) > 4:
        vocab, total_freq = read_vocab(sys.argv[4])

    write_bidict(src_embs, tgt_embs, src_idx2word, tgt_idx2word, sys.argv[3], vocab, total_freq)
