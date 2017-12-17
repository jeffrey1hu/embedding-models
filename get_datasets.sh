#!/bin/bash

DATASETS_DIR="utils/datasets"
mkdir -p $DATASETS_DIR

cd $DATASETS_DIR

# Get Stanford Sentiment Treebank
if hash wget 2>/dev/null; then
  wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
else
  curl -O http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
fi
unzip stanfordSentimentTreebank.zip
rm stanfordSentimentTreebank.zip
