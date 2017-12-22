# word2vec_vs_glove
Simple python implementation of two popular word embedding algorithm: Word2vec and GloVe.

### Note
* The project is only for educational purposes.
* The word2vec code is build under the instructions of [cs224n assigment #1](http://web.stanford.edu/class/cs224n/assignment3/index.html).
* The glove implementation is followed along with Hans [blog](http://www.foldl.me/2014/glove-python/).

### Dataset
* The existing dataset in this project is SST(Stanford Sentiment Treebank)
* SST contain sentiment analysis labels which can be used to evaluating the pros & cons of each embedding model.

### Usage
* Install the dependencies (Python2.7)
```bash
pip install -r requirement.txt
```
* Download dataset
```bash
sh get_datasets.sh
```
* train word2vec
```bash
python train.py -m word2vec --save-every=True --vector-path=./model/word2vec -s 10 --learning-rate=0.3 -w 5 --iterations=40000
```
* train glove
```bash
python train.py -m glove -s 50 --learning-rate=0.05 --iterations=200 --save-every=True --vector-path=./model/glove
```
