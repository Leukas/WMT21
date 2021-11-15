# WMT21
This repo hosts the code for University of Groningen's submission to the WMT21 Unsupervised MT Task for German--Lower Sorbian translation. Our code consists of modifications to the [MASS](https://github.com/microsoft/MASS) model, scripts for our novel vocabulary transfer method, and scripts for running our 6 step process described in our paper (WMT link: http://www.statmt.org/wmt21/pdf/2021.wmt-1.104.pdf, arXiv link: https://arxiv.org/pdf/2109.12012.pdf): 

Lukas Edman, Ahmet Üstün, Antonio Toral, and Gertjan van Noord. 2021. Unsupervised Translation of German–Lower Sorbian: Exploring Training and Transfer Methods on a Low-Resource Language. _Proceedings of the Sixth Conference on Machine Translation._ (Accepted)

In the preliminary results from the task organizers, our system ranked tied first place for Lower Sorbian→German (according to BERTScore) and third for German→Lower Sorbian (according to BLEU).


### Requirements
- Python 3 (tested on 3.7.9)
- [PyTorch](https://pytorch.org/) (tested on 1.7.1)
- [Moses](https://github.com/moses-smt/mosesdecoder)
- [fastBPE](https://github.com/glample/fastBPE)
- [fastText](https://github.com/facebookresearch/fastText)
- [VecMap](https://github.com/artetxem/vecmap)

### Examples
Prior to training, you will need to preprocess the data following the format expected from MASS. 

For training steps 1, 2, and 4-6, the file ```extra/mass_steps.sh``` can be run, provided the paths in the first few lines are updated. For example, step 1 can be run as such:

```extra/mass_steps.sh step1```

#### Vocab Transfer
For vocab transfer (step 3), you first need to run fastText on the training data for Lower and Upper Sorbian. 
```
fasttext skipgram -epoch 10 -minCount 5 -dim 512 -thread 10 -ws 5 -neg 10 -input train.dsb -output train.dsb
fasttext skipgram -epoch 10 -minCount 5 -dim 512 -thread 10 -ws 5 -neg 10 -input train.hsb -output train.hsb
```
Next, you need to map these together with VecMap:
```
python vecmap/map_embeddings.py --identical \
    train.dsb.vec \
    train.hsb.vec \
    train.dsb.vec.map.dsb-hsb \
    train.hsb.vec.map.dsb-hsb \
    --cuda -v --batch_size 5000
```
You also need to get the vocabularies of the training sets for Lower Sorbian, Upper Sorbian, and German, using fastBPE:
```
fast getvocab train.dsb > vocab.dsb
fast getvocab train.hsb > vocab.hsb
fast getvocab train.de > vocab.de
```

Finally, you can run either ```extra/get_bidict.py``` or ```extra/get_bidict_lev.py``` for the simple or Levenshtein versions, respectively:
```
python get_bidict.py \
                train.dsb.vec.map.dsb-hsb \
                train.hsb.vec.map.dsb-hsb \
                bidict.dsb-hsb \
                vocab.de

python get_bidict_lev.py \
                train.dsb.vec.map.dsb-hsb \
                train.hsb.vec.map.dsb-hsb \
                bidict_lev.dsb-hsb \
                vocab.dsb vocab.hsb vocab.de
```

These bidicts are then used in step 4, or in any step by providing 2 arguments to MASS's ```train.py```:

```--tie_lang_embs "dsb,hsb" --transfer_vocab bidict.dsb-hsb```
