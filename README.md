# Fast-TransE

An implementation of the TransE model (https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf) based on the version ("https://github.com/thunlp/KB2E"). The overall framework is similar with some underlying design changes for acceleration. And this implementation can support multi-threaded training to save more time.

# Evaluation Results

Because the overall framework is similar, we just list the result of transE(this and previous model) in dateset FB15k.

| Model | MeanRank(Raw)	| MeanRank(Filter)	| Hit@10(Raw)	| Hit@10(Filter)|time(min)|
| ----- |:-------------:| :----------------:|:-----------:|:-------------:|:---:|
|TransE(n = 50, rounds = 3000)|224|76|43.2|65.6|156|
|Fast-TransE(n = 50, threads = 8, rounds = 3000)|212|70|44.5|66.3|4|

The more results can be found in ("https://github.com/thunlp/KB2E").

# Data

Datasets are required in the following format, containing three files:

triple2id.txt: training file, the first line is the number of triples for training. Then the follow lines are all in the format (e1, e2, rel).

entity2id.txt: all entities and corresponding ids, one per line. The first line is the number of entities.

relation2id.txt: all relations and corresponding ids, one per line. The first line is the number of relations.

You can download FB15K from [[Download]](http://pan.baidu.com/s/1eRD9B4A), and the more datasets can also be found in ("https://github.com/thunlp/KB2E").

# Compile

g++ transE.cpp -o transE -pthread -O3 -march=native


