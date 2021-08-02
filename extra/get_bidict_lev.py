# get_bidict.py
import numpy as np
import sys
import unidecode
from Levenshtein import distance
# SORBIAN_CHARS = "AaBbCcČčĆćDdEeĚěFfGgHhIiJjKkŁłLlMmNnŃńOoÓóPpRrŘřŔŕSsŠšŚśTtUuWwYyZzŽžŹź"

def strip_accents(text):
    return unidecode.unidecode(text)

def read_vocab(vocab_file):
    vocab = {}
    total_freq = 0.0
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

def write_bidict(src_embs, tgt_embs, src_idx2word, tgt_idx2word, out_file, vocabs=None, total_freqs=None):
    same = 0
    aux_vocab = vocabs[-1]
    aux_total_freq = total_freqs[-1]
    with open(out_file, 'w', encoding='utf8') as file:
        for i in range(len(src_embs)):
            src_word = src_idx2word[i]
            if src_word == "</s>":
                continue
            if (aux_vocab is not None and src_word in aux_vocab):
                if aux_vocab[src_word] / float(aux_total_freq) > 0.00001: 
                    continue

            cossims = cossim(tgt_embs, src_embs[i])
            idxs = np.argsort(cossims)
            min_dist = 10000000
            best_word = tgt_idx2word[idxs[-1]]
            best_idx = 0
            for i in range(10):
                closest_idx = idxs[-1-i]
                closest_word = tgt_idx2word[closest_idx]
                if closest_word not in vocabs[1]: 
                    continue
                
                lev_dist = distance(strip_accents(src_word), strip_accents(closest_word))
                if lev_dist < min_dist:
                    min_dist = lev_dist
                    best_word = closest_word
                    best_idx = closest_idx

            if best_word == src_word:
                same += 1

            file.write("\t".join([src_word, best_word, str(cossims[best_idx])]) + "\n")
    print("SAME:", same)

if __name__ == "__main__":
    src_word2idx, src_idx2word, src_embs = read_embs(sys.argv[1])
    tgt_word2idx, tgt_idx2word, tgt_embs = read_embs(sys.argv[2])

    src_vocab, src_total_freq = read_vocab(sys.argv[4]) # vocab.dsb
    tgt_vocab, tgt_total_freq = read_vocab(sys.argv[5]) # vocab.hsb
    aux_vocab, aux_total_freq = read_vocab(sys.argv[6]) # vocab.de

    vocabs = [src_vocab, tgt_vocab, aux_vocab]
    total_freqs = [src_total_freq, tgt_total_freq, aux_total_freq]

    write_bidict(src_embs, tgt_embs, src_idx2word, tgt_idx2word, sys.argv[3], vocabs, total_freqs)
