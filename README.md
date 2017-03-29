# Fast-TransX

An extremely fast implementation of TransE [1], TransH [2], TransR [3], TransD [4], TranSparse [5] for knowledge representation learning (KRL) based on our previous pakcage KB2E ("https://github.com/thunlp/KB2E") for KRL. The overall framework is similar to KB2E, with some underlying design changes for acceleration. This implementation also supports multi-threaded training to save time.

# Evaluation Results

Because the overall framework is similar, we just list the result of transE(previous model) and new implemented models in dateset FB15k.

CPU : Intel Core i7-6700k 4.00GHz.

| Model | MeanRank(Raw)	| MeanRank(Filter)	| Hit@10(Raw)	| Hit@10(Filter)|Time|
| ----- |:-------------:| :----------------:|:-----------:|:-------------:|:---:|
|TransE (n = 50, rounds = 1000)|210|82|41.9|61.3|59m47s|
|Fast-TransE (n = 50, threads = 8, rounds = 1000)|212|82|41.4|59.7|45s|
|Fast-TransH (n = 50, threads = 8, rounds = 1000)|203|67|44.6|63.1|2m24s|
|Fast-TransR (n = 50, threads = 8, rounds = 1000)|196|76|45.6|69.1|19m34s|
|Fast-TransD (n = 50, threads = 8, rounds = 1000)|210|73|43.5|64|3m19s|

More results can be found in ("https://github.com/thunlp/KB2E").

# Data

Datasets are required in the following format, containing three files:

triple2id.txt: training file, the first line is the number of triples for training. Then the follow lines are all in the format (e1, e2, rel).

entity2id.txt: all entities and corresponding ids, one per line. The first line is the number of entities.

relation2id.txt: all relations and corresponding ids, one per line. The first line is the number of relations.

You can download FB15K from [[Download]](http://pan.baidu.com/s/1eRD9B4A), and the more datasets can also be found in ("https://github.com/thunlp/KB2E").

# Compile

g++ transX.cpp -o transX -pthread -O3 -march=native

# Citation

If you use the code, please kindly cite the following paper and other papers listed in our reference:

Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, Xuan Zhu. Learning Entity and Relation Embeddings for Knowledge Graph Completion. The 29th AAAI Conference on Artificial Intelligence (AAAI'15). [[pdf]](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/aaai2015_transr.pdf)

# Reference

[1] Bordes, Antoine, et al. Translating embeddings for modeling multi-relational data. Proceedings of NIPS, 2013.

[2]	Zhen Wang, Jianwen Zhang, et al. Knowledge Graph Embedding by Translating on Hyperplanes. Proceedings of AAAI, 2014.

[3] Yankai Lin, Zhiyuan Liu, et al. Learning Entity and Relation Embeddings for Knowledge Graph Completion. Proceedings of AAAI, 2015.

[4] Guoliang Ji, Shizhu He, et al. Knowledge Graph Embedding via Dynamic Mapping Matrix. Proceedings of ACL, 2015.

[5] Guoliang Ji, Kang Liu, et al. Knowledge Graph Completion with Adaptive Sparse Transfer Matrix. Proceedings of AAAI, 2016.
