# Fast-TransX

This repository is a subproject of THU-OpenSK, and all subprojects of THU-OpenSK are as follows.

- [OpenNE](https://www.github.com/thunlp/OpenNE)
- [OpenKE](https://www.github.com/thunlp/OpenKE)
  - [KB2E](https://www.github.com/thunlp/KB2E)
  - [TensorFlow-Transx](https://www.github.com/thunlp/TensorFlow-Transx)
  - [Fast-TransX](https://www.github.com/thunlp/Fast-TransX)
- [OpenNRE](https://www.github.com/thunlp/OpenNRE)
  - [JointNRE](https://www.github.com/thunlp/JointNRE)

An extremely fast implementation of TransE [1], TransH [2], TransR [3], TransD [4], TranSparse [5] for knowledge representation learning (KRL) based on our previous pakcage KB2E ("https://github.com/thunlp/KB2E") for KRL. The overall framework is similar to KB2E, with some underlying design changes for acceleration. This implementation also supports multi-threaded training to save time.

These codes will be gradually integrated into the new framework [[OpenKE]](https://github.com/thunlp/openke).

# Evaluation Results

Because the overall framework is similar, we just list the result of transE(previous model) and new implemented models in datesets FB15k and WN18.

CPU : Intel Core i7-6700k 4.00GHz.

FB15K:

| Model | MeanRank(Raw)	| MeanRank(Filter)	| Hit@10(Raw)	| Hit@10(Filter)|Time|
| ----- |:-------------:| :----------------:|:-----------:|:-------------:|:---:|
|TransE (n = 50, rounds = 1000)|210|82|41.9|61.3|3587s|
|Fast-TransE (n = 50, threads = 8, rounds = 1000)|205|69|43.8|63.5|42s|
|Fast-TransH (n = 50, threads = 8, rounds = 1000)|202|67|43.7|63.0|178s|
|Fast-TransR (n = 50, threads = 8, rounds = 1000)|196|73|48.8|69.8|1572s|
|Fast-TransD (n = 100, threads = 8, rounds = 1000)|236|95|49.9|75.2|231s|


WN18:

| Model | MeanRank(Raw)	| MeanRank(Filter)	| Hit@10(Raw)	| Hit@10(Filter)|Time|
| ----- |:-------------:| :----------------:|:-----------:|:-------------:|:---:|
|TransE (n = 50, rounds = 1000)|251|239|78.9|89.8|1674s|
|Fast-TransE (n = 50, threads = 8, rounds = 1000)|273|261|71.5|83.3|12s|
|Fast-TransH (n = 50, threads = 8, rounds = 1000)|285|272|79.8|92.5|121s|
|Fast-TransR (n = 50, threads = 8, rounds = 1000)|284|271|81.0|94.6|296s|
|Fast-TransD (n = 100, threads = 8, rounds = 1000)|309|297|78.5|91.9|201s|

More results can be found in ("https://github.com/thunlp/KB2E").

# Data

Datasets are required in the following format, containing three files:

entity2id.txt: all entities and corresponding ids, one per line. The first line is the number of entities.

relation2id.txt: all relations and corresponding ids, one per line. The first line is the number of relations.

train2id.txt: training file, the first line is the number of triples for training. Then the follow lines are all in the format (e1, e2, rel). **Note that train2id.txt contains ids from entitiy2id.txt and relation2id.txt instead of the names of the entities and relations.**

We provide FB15K and WN18, and more datasets can be found in ("https://github.com/thunlp/KB2E"). **If you use your own datasets, please check the format of your training file. Files in the wrong format may cause segmentation fault. Datasets in KB2E also need to change their formats before training.**

# Compile

	g++ transX.cpp -o transX -pthread -O3 -march=native
	
	g++ test_transX.cpp -o test_transX -pthread -O3 -march=native

# Train

	./transX [-size SIZE] [-sizeR SIZER]
	         [-input INPUT] [-output OUTPUT] [-load LOAD]
	         [-load-binary 0/1] [-out-binary 0/1]
	         [-thread THREAD] [-epochs EPOCHS] [-nbatches NBATCHES]
	         [-alpha ALPHA] [-margin MARGIN]
	         [-note NOTE]
	
	optional arguments:
	-size SIZE           dimension of entity embeddings
	-sizeR SIZER         dimension of relation embeddings
	-input INPUT         folder of training data
	-output OUTPUT       folder of outputing results
	-load LOAD           folder of pretrained data
	-load-binary [0/1]   [1] pretrained data need to load in is in the binary form
	-out-binary [0/1]    [1] results will be outputed in the binary form
	-thread THREAD       number of worker threads
	-epochs EPOCHS       number of epochs
	-nbatches NBATCHES   number of batches for each epoch
	-alpha ALPHA         learning rate
	-margin MARGIN       margin in max-margin loss for pairwise training
	-note NOTE           information you want to add to the filename

# Test
	./test_transX [-size SIZE] [-sizeR SIZER]
	         [-input INPUT] [-init INIT]
	         [-binary 0/1] [-thread THREAD]
	         [-note NOTE]
	
	optional arguments:
	-size SIZE           dimension of entity embeddings
	-sizeR SIZER         dimension of relation embeddings
	-input INPUT         folder of testing data
	-init INIT           folder of embeddings
	-binary [0/1]        [1] embeddings are in the binary form
	-thread THREAD       number of worker threads
	-note NOTE           information you want to add to the filename

# Citation

If you use the code, please kindly cite the following paper and other papers listed in our reference:

Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, Xuan Zhu. Learning Entity and Relation Embeddings for Knowledge Graph Completion. The 29th AAAI Conference on Artificial Intelligence (AAAI'15). [[pdf]](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/aaai2015_transr.pdf)

# Reference

[1] Bordes, Antoine, et al. Translating embeddings for modeling multi-relational data. Proceedings of NIPS, 2013.

[2]	Zhen Wang, Jianwen Zhang, et al. Knowledge Graph Embedding by Translating on Hyperplanes. Proceedings of AAAI, 2014.

[3] Yankai Lin, Zhiyuan Liu, et al. Learning Entity and Relation Embeddings for Knowledge Graph Completion. Proceedings of AAAI, 2015.

[4] Guoliang Ji, Shizhu He, et al. Knowledge Graph Embedding via Dynamic Mapping Matrix. Proceedings of ACL, 2015.

[5] Guoliang Ji, Kang Liu, et al. Knowledge Graph Completion with Adaptive Sparse Transfer Matrix. Proceedings of AAAI, 2016.
