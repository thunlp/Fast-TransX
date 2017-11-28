#define REAL float
#define INT int
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>
#include <pthread.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

using namespace std;

const REAL pi = 3.141592653589793238462643383;

INT threads = 8;
INT bernFlag = 0;
INT loadBinaryFlag = 0;
INT outBinaryFlag = 0;
INT trainTimes = 1000;
INT nbatches = 100;
INT dimension = 100;
INT dimensionR = 100;
REAL alpha = 0.001;
REAL margin = 1;

string inPath = "./";
string outPath = "";
string loadPath = "";
string note = "";

INT *lefHead, *rigHead;
INT *lefTail, *rigTail;

struct Triple {
	INT h, r, t;
};

struct cmp_head {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};

struct cmp_tail {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}
};

struct cmp_list {
	INT minimal(INT a,INT b) {
		if (a < b) return b;
		return a;
	}
	bool operator()(const Triple &a, const Triple &b) {
		return (minimal(a.h, a.t) < minimal(b.h, b.t));
	}
};

Triple *trainHead, *trainTail, *trainList;

/*
	There are some math functions for the program initialization.
*/

unsigned long long *next_random;

unsigned long long randd(INT id) {
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];
}

INT rand_max(INT id, INT x) {
	INT res = randd(id) % x;
	while (res<0)
		res+=x;
	return res;
}

REAL rand(REAL min, REAL max) {
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

REAL normal(REAL x, REAL miu,REAL sigma) {
	return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

REAL randn(REAL miu,REAL sigma, REAL min ,REAL max) {
	REAL x, y, dScope;
	do {
		x = rand(min,max);
		y = normal(x,miu,sigma);
		dScope=rand(0.0,normal(miu,miu,sigma));
	} while (dScope > y);
	return x;
}

void norm(REAL *con, INT dimension) {
	REAL x = 0;
	for (INT  ii = 0; ii < dimension; ii++)
		x += (*(con + ii)) * (*(con + ii));
	x = sqrt(x);
	if (x>1)
		for (INT ii=0; ii < dimension; ii++)
			*(con + ii) /= x;
}

void norm(REAL *con, REAL *conTrans, REAL *relTrans) {
	REAL x = 0, ee = 0, tmp1 = 0;
	for (INT i = 0; i < dimension; i++)
		ee += con[i] * conTrans[i];
	for (INT i = 0; i < dimensionR; i++) {
		REAL tmp = ee * relTrans[i];
		if (i < dimension)
			tmp = tmp + con[i];
		x += tmp * tmp;
		tmp1 += tmp * 2 * relTrans[i];
	}
	if (x > 1) {
		REAL lambda = 1;
		for (INT i = 0; i < dimensionR; i++) {
			REAL tmp = ee * relTrans[i];
			if (i < dimension)
				tmp = tmp + con[i];
			tmp = tmp + tmp;
			relTrans[i] -= alpha * lambda * tmp * ee;
			if (i < dimension)
				con[i] -= alpha * lambda * tmp;
		}
		for (INT i = 0; i < dimension; i++) {
			con[i] -= alpha * lambda * tmp1 * conTrans[i];
			conTrans[i] -= alpha * lambda * tmp1 * con[i];
		}
	}
}

INT relationTotal, entityTotal, tripleTotal;
INT *freqRel, *freqEnt;
REAL *left_mean, *right_mean;
REAL *relationVec, *entityVec, *matrix;
REAL *relationVecDao, *entityVecDao, *matrixDao;
REAL *entityTransVec, *entityTransVecDao, *relationTransVec, *relationTransVecDao;
REAL *tmpValue;

void norm(INT h, INT t, INT r, INT j) {
		norm(relationVecDao + dimensionR * r, dimensionR);
		norm(entityVecDao + dimension * h, dimension);
		norm(entityVecDao + dimension * t, dimension);
		norm(entityVecDao + dimension * j, dimension);
		norm(entityVecDao + dimension * h, entityTransVecDao + dimension * h, relationTransVecDao + dimensionR * r);
		norm(entityVecDao + dimension * t, entityTransVecDao + dimension * t, relationTransVecDao + dimensionR * r);
		norm(entityVecDao + dimension * j, entityTransVecDao + dimension * j, relationTransVecDao + dimensionR * r);
}

/*
	Read triples from the training file.
*/

void init() {

	FILE *fin;
	INT tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &relationTotal);
	fclose(fin);

	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &entityTotal);
	fclose(fin);

	relationVec = (REAL *)calloc(relationTotal * dimensionR * 2 + entityTotal * dimension * 2 + relationTotal * dimensionR * 2 + entityTotal * dimension * 2, sizeof(REAL));
	relationVecDao = relationVec + relationTotal * dimensionR;
	entityVec = relationVecDao + relationTotal * dimensionR;
	entityVecDao = entityVec + entityTotal * dimension;
	relationTransVec = entityVecDao + entityTotal * dimension;
	relationTransVecDao = relationTransVec + relationTotal * dimensionR;
	entityTransVec = relationTransVecDao + relationTotal * dimensionR;
	entityTransVecDao = entityTransVec + entityTotal * dimension;

	freqRel = (INT *)calloc(relationTotal + entityTotal, sizeof(INT));
	freqEnt = freqRel + relationTotal;

	for (INT i = 0; i < relationTotal; i++) {
		for (INT ii=0; ii < dimensionR; ii++) {
			relationVec[i * dimensionR + ii] = randn(0, 1.0 / dimensionR, -6 / sqrt(dimensionR), 6 / sqrt(dimensionR));
			relationTransVec[i * dimensionR + ii] = randn(0, 1.0 / dimensionR, -6 / sqrt(dimensionR), 6 / sqrt(dimensionR));
		}
		norm(relationTransVec + i * dimensionR, dimensionR);
	}
	for (INT i = 0; i < entityTotal; i++) {
		for (INT ii=0; ii < dimension; ii++) {
			entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
			entityTransVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
		}
		norm(entityVec + i * dimension, dimension);
		norm(entityTransVec + i * dimension, dimension);
	}

	fin = fopen((inPath + "train2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &tripleTotal);
	trainHead = (Triple *)calloc(tripleTotal * 3, sizeof(Triple));
	trainTail = trainHead + tripleTotal;
	trainList = trainTail + tripleTotal;
	for (INT i = 0; i < tripleTotal; i++) {
		tmp = fscanf(fin, "%d", &trainList[i].h);
		tmp = fscanf(fin, "%d", &trainList[i].t);
		tmp = fscanf(fin, "%d", &trainList[i].r);
		freqEnt[trainList[i].t]++;
		freqEnt[trainList[i].h]++;
		freqRel[trainList[i].r]++;
		trainHead[i] = trainList[i];
		trainTail[i] = trainList[i];
	}
	fclose(fin);

	sort(trainHead, trainHead + tripleTotal, cmp_head());
	sort(trainTail, trainTail + tripleTotal, cmp_tail());
	sort(trainList, trainList + tripleTotal, cmp_list());

	lefHead = (INT *)calloc(entityTotal * 4, sizeof(INT));
	rigHead = lefHead + entityTotal;
	lefTail = rigHead + entityTotal;
	rigTail = lefTail + entityTotal;
	memset(rigHead, -1, sizeof(INT)*entityTotal);
	memset(rigTail, -1, sizeof(INT)*entityTotal);
	for (INT i = 1; i < tripleTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) {
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
	}
	rigHead[trainHead[tripleTotal - 1].h] = tripleTotal - 1;
	rigTail[trainTail[tripleTotal - 1].t] = tripleTotal - 1;

	left_mean = (REAL *)calloc(relationTotal * 2,sizeof(REAL));
	right_mean = left_mean + relationTotal;
	for (INT i = 0; i < entityTotal; i++) {
		for (INT j = lefHead[i] + 1; j <= rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (INT j = lefTail[i] + 1; j <= rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	for (INT i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];
		right_mean[i] = freqRel[i] / right_mean[i];
	}
}

void load_binary() {
    struct stat statbuf1;
    if (stat((loadPath + "entity2vec" + note + ".bin").c_str(), &statbuf1) != -1) {  
        INT fd = open((loadPath + "entity2vec" + note + ".bin").c_str(), O_RDONLY);
        REAL* entityVecTmp = (REAL*)mmap(NULL, statbuf1.st_size, PROT_READ, MAP_PRIVATE, fd, 0); 
        memcpy(entityVec, entityVecTmp, statbuf1.st_size);
        munmap(entityVecTmp, statbuf1.st_size);
        close(fd);
    }  
    struct stat statbuf2;
    if (stat((loadPath + "relation2vec" + note + ".bin").c_str(), &statbuf2) != -1) {  
        INT fd = open((loadPath + "relation2vec" + note + ".bin").c_str(), O_RDONLY);
        REAL* relationVecTmp =(REAL*)mmap(NULL, statbuf2.st_size, PROT_READ, MAP_PRIVATE, fd, 0); 
        memcpy(relationVec, relationVecTmp, statbuf2.st_size);
        munmap(relationVecTmp, statbuf2.st_size);
        close(fd);
    }
    struct stat statbuf3;
    if (stat((loadPath + "A" + note + ".bin").c_str(), &statbuf3) != -1) {  
        INT fd = open((loadPath + "A" + note + ".bin").c_str(), O_RDONLY);
        REAL* ATmp =(REAL*)mmap(NULL, statbuf3.st_size, PROT_READ, MAP_PRIVATE, fd, 0); 
        memcpy(relationTransVec, ATmp, relationTotal * dimensionR * sizeof(REAL));
        memcpy(entityTransVec, ATmp + relationTotal * dimensionR, entityTotal * dimension * sizeof(REAL));
        munmap(ATmp, statbuf3.st_size);
        close(fd);
    }
}

void load() {
    if (loadBinaryFlag) {
        load_binary();
        return;
    }
    FILE *fin;
    INT tmp;
    fin = fopen((loadPath + "entity2vec" + note + ".vec").c_str(), "r");
    for (INT i = 0; i < entityTotal; i++) {
        INT last = i * dimension;
        for (INT j = 0; j < dimension; j++)
            tmp = fscanf(fin, "%f", &entityVec[last + j]);
    }
    fclose(fin);
    fin = fopen((loadPath + "relation2vec" + note + ".vec").c_str(), "r");
    for (INT i = 0; i < relationTotal; i++) {
        INT last = i * dimensionR;
        for (INT j = 0; j < dimensionR; j++)
            tmp = fscanf(fin, "%f", &relationVec[last + j]);
    }
    fclose(fin);

    fin = fopen((loadPath + "A" + note + ".vec").c_str(), "r");
    for (INT i=0; i < relationTotal; i++) {
        INT last = dimensionR * i;
        for (INT ii = 0; ii < dimensionR; ii++)
            tmp = fscanf(fin, "%f", &relationTransVec[last + ii]);
    }
    for (INT  i = 0; i < entityTotal; i++) {
        INT last = i * dimension;
        for (INT ii = 0; ii < dimension; ii++)
            tmp = fscanf(fin, "%f", &entityTransVec[last + ii] );
    }
    fclose(fin);
}

/*
	Training process of transD.
*/

INT transDLen;
INT transDBatch;
REAL res;

REAL calc_sum(INT e1, INT e2, INT rel, REAL *tmp1, REAL *tmp2, REAL &ee1, REAL &ee2) {
	INT last1 = e1 * dimension;
	INT last2 = e2 * dimension;
	INT lastr = rel * dimensionR;
	REAL sum = 0;
	ee1 = 0, ee2 = 0;
	for (INT ii = 0; ii < dimension; ii++) {
		ee1 += entityTransVec[last1 + ii] * entityVec[last1 + ii];
		ee2 += entityTransVec[last2 + ii] * entityVec[last2 + ii];
	}
	for (INT ii = 0; ii < dimensionR; ii++) {
		tmp1[ii] = ee1 * relationTransVec[lastr + ii];
		tmp2[ii] = ee2 * relationTransVec[lastr + ii];
		if (ii < dimension) {
			tmp1[ii] += entityVec[last1 + ii];
			tmp2[ii] += entityVec[last2 + ii];
		}
		sum += fabs(tmp1[ii] + relationVec[lastr + ii] - tmp2[ii]);
	}
	return sum;
}

void gradient(INT e1_a, INT e2_a, INT rel_a, INT belta, INT same, REAL *tmp1, REAL *tmp2, REAL &e1, REAL &e2) {
	INT lasta1 = e1_a * dimension;
	INT lasta2 = e2_a * dimension;
	INT lastar = rel_a * dimensionR;
	INT lastM = rel_a * dimensionR * dimension;
	REAL x, s = 0;
	for (INT ii = 0; ii < dimensionR; ii++) {
		x = tmp2[ii] - tmp1[ii] - relationVec[lastar + ii];
		if (x > 0)
			x = belta * alpha;
		else
			x = -belta * alpha;
		s += x * relationTransVec[lastar + ii];
		relationTransVecDao[lastar + ii] -= same * x * e1;
		relationTransVecDao[lastar + ii] += same * x * e2;
		relationVecDao[lastar + ii] -= same * x;
		if (ii < dimension) {
			entityVecDao[lasta1 + ii] -= same * x;
			entityVecDao[lasta2 + ii] += same * x;
		}
	}
	s = s * same;
	for (INT ii = 0; ii < dimension; ii++) {
		entityVecDao[lasta1 + ii] -= s * entityTransVec[lasta1 + ii];
		entityTransVecDao[lasta1 + ii] -= s * entityVec[lasta1 + ii];
		entityVecDao[lasta2 + ii] += s * entityTransVec[lasta2 + ii];
		entityTransVecDao[lasta2 + ii] += s * entityVec[lasta2 + ii];
	}
}

void train_kb(INT e1_a, INT e2_a, INT rel_a, INT e1_b, INT e2_b, INT rel_b, REAL *tmp) {
	REAL e1, e2, e3, e4;
	REAL sum1 = calc_sum(e1_a, e2_a, rel_a, tmp, tmp + dimensionR, e1, e2);
	REAL sum2 = calc_sum(e1_b, e2_b, rel_b, tmp + dimensionR * 2, tmp + dimensionR * 3, e3, e4);
	if (sum1 + margin > sum2) {
		res += margin + sum1 - sum2;
		gradient(e1_a, e2_a, rel_a, -1, 1, tmp, tmp + dimensionR, e1, e2);
    	gradient(e1_b, e2_b, rel_b, 1, 1, tmp + dimensionR * 2, tmp + dimensionR * 3, e3, e4);
	}
}

INT corrupt_head(INT id, INT h, INT r) {
	INT lef, rig, mid, ll, rr;
	lef = lefHead[h] - 1;
	rig = rigHead[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainHead[ll].t) return tmp;
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

INT corrupt_tail(INT id, INT t, INT r) {
	INT lef, rig, mid, ll, rr;
	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainTail[ll].h) return tmp;
	if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

void* transDtrainMode(void *con) {
	INT id, i, j, pr;
	id = (unsigned long long)(con);
	next_random[id] = rand();
	REAL *tmp = tmpValue + id * dimensionR * 4;
	for (INT k = transDBatch / threads; k >= 0; k--) {
		i = rand_max(id, transDLen);	
		if (bernFlag)
			pr = 1000*right_mean[trainList[i].r]/(right_mean[trainList[i].r]+left_mean[trainList[i].r]);
		else
			pr = 500;
		if (randd(id) % 1000 < pr) {
			j = corrupt_head(id, trainList[i].h, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, trainList[i].h, j, trainList[i].r, tmp);
		} else {
			j = corrupt_tail(id, trainList[i].t, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, j, trainList[i].t, trainList[i].r, tmp);
		}
		norm(trainList[i].h, trainList[i].t, trainList[i].r, j);
	}
	pthread_exit(NULL);
}

void* train(void *con) {
	transDLen = tripleTotal;
	transDBatch = transDLen / nbatches;
	next_random = (unsigned long long *)calloc(threads, sizeof(unsigned long long));
	tmpValue = (REAL *)calloc(threads * dimensionR * 4, sizeof(REAL));
	memcpy(relationVecDao, relationVec, dimensionR * relationTotal * sizeof(REAL));
	memcpy(entityVecDao, entityVec, dimension * entityTotal * sizeof(REAL));
	memcpy(relationTransVecDao, relationTransVec, dimensionR * relationTotal * sizeof(REAL));
	memcpy(entityTransVecDao, entityTransVec, dimension * entityTotal * sizeof(REAL));
	for (INT epoch = 0; epoch < trainTimes; epoch++) {
		res = 0;
		for (INT batch = 0; batch < nbatches; batch++) {
			pthread_t *pt = (pthread_t *)malloc(threads * sizeof(pthread_t));
			for (long a = 0; a < threads; a++)
				pthread_create(&pt[a], NULL, transDtrainMode,  (void*)a);
			for (long a = 0; a < threads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
			memcpy(relationVec, relationVecDao, dimensionR * relationTotal * sizeof(REAL));
			memcpy(entityVec, entityVecDao, dimension * entityTotal * sizeof(REAL));
			memcpy(relationTransVec, relationTransVecDao, dimensionR * relationTotal * sizeof(REAL));
			memcpy(entityTransVec, entityTransVecDao, dimension * entityTotal * sizeof(REAL));
		}
		printf("epoch %d %f\n", epoch, res);
	}
}

/*
	Get the results of transD.
*/

void out_binary() {
		INT len, tot;
		REAL *head;		
		FILE* f2 = fopen((outPath + "relation2vec" + note + ".bin").c_str(), "wb");
		FILE* f3 = fopen((outPath + "entity2vec" + note + ".bin").c_str(), "wb");
		len = relationTotal * dimension; tot = 0;
		head = relationVec;
		while (tot < len) {
			INT sum = fwrite(head + tot, sizeof(REAL), len - tot, f2);
			tot = tot + sum;
		}
		len = entityTotal * dimension; tot = 0;
		head = entityVec;
		while (tot < len) {
			INT sum = fwrite(head + tot, sizeof(REAL), len - tot, f3);
			tot = tot + sum;
		}	
		fclose(f2);
		fclose(f3);
		FILE* f1 = fopen((outPath + "A" + note + ".bin").c_str(), "wb");
		len = relationTotal * dimensionR; tot = 0;
		head = relationTransVec;
		while (tot < len) {
			INT sum = fwrite(head + tot, sizeof(REAL), len - tot, f1);
			tot = tot + sum;
		}
		len = entityTotal * dimension; tot = 0;
		head = entityTransVec;
		while (tot < len) {
			INT sum = fwrite(head + tot, sizeof(REAL), len - tot, f1);
			tot = tot + sum;
		}
		fclose(f1);
}

void out() {
		if (outBinaryFlag) {
			out_binary(); 
			return;
		}
		FILE* f2 = fopen((outPath + "relation2vec" + note + ".vec").c_str(), "w");
		FILE* f3 = fopen((outPath + "entity2vec" + note + ".vec").c_str(), "w");
		for (INT i=0; i < relationTotal; i++) {
			INT last = dimension * i;
			for (INT ii = 0; ii < dimension; ii++)
				fprintf(f2, "%.6f\t", relationVec[last + ii]);
			fprintf(f2,"\n");
		}
		for (INT  i = 0; i < entityTotal; i++) {
			INT last = i * dimension;
			for (INT ii = 0; ii < dimension; ii++)
				fprintf(f3, "%.6f\t", entityVec[last + ii] );
			fprintf(f3,"\n");
		}
		fclose(f2);
		fclose(f3);
		FILE* f1 = fopen((outPath + "A" + note + ".vec").c_str(),"w");
		for (INT i = 0; i < relationTotal; i++) {
			INT last = dimensionR * i;
			for (INT ii = 0; ii < dimensionR; ii++)
				fprintf(f1, "%.6f\t", relationTransVec[last + ii]);
			fprintf(f1,"\n");
		}		
		for (INT  i = 0; i < entityTotal; i++) {
			INT last = i * dimension;
			for (INT ii = 0; ii < dimension; ii++)
				fprintf(f1, "%.6f\t", entityTransVec[last + ii] );
			fprintf(f1,"\n");
		}
		fclose(f1);
}

/*
	Main function
*/

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}


void setparameters(int argc, char **argv) {
	int i;
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dimension = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-sizeR", argc, argv)) > 0) dimensionR = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-input", argc, argv)) > 0) inPath = argv[i + 1];
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) outPath = argv[i + 1];
	if ((i = ArgPos((char *)"-load", argc, argv)) > 0) loadPath = argv[i + 1];
	if ((i = ArgPos((char *)"-thread", argc, argv)) > 0) threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-epochs", argc, argv)) > 0) trainTimes = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-nbatches", argc, argv)) > 0) nbatches = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-load-binary", argc, argv)) > 0) loadBinaryFlag = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-out-binary", argc, argv)) > 0) outBinaryFlag = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-note", argc, argv)) > 0) note = argv[i + 1];
}

int main(int argc, char **argv) {
	setparameters(argc, argv);
	init();
	if (loadPath != "") load();
	train(NULL);
	if (outPath != "") out();
	return 0;
}
