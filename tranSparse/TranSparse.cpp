#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>
#include <pthread.h>

using namespace std;

const float pi = 3.141592653589793238462643383;
int bern = 0;
int tranSparseThreads = 8;
int tranSparseTrainTimes = 1000;
int nbatches = 100;
int dimension = 50;
int dimensionR = 50;
float tranSparseAlpha = 0.001;
float margin = 1;

string inPath = "./data/";
string outPath = "./out/";

int *lefHead, *rigHead;
int *lefTail, *rigTail;

struct Triple {
	int h, r, t;
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
	int minimal(int a,int b) {
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

unsigned long long randd(int id) {
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];
}

int rand_max(int id, int x) {
	int res = randd(id) % x;
	while (res < 0)
		res += x;
	return res;
}

float rand(float min, float max) {
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

float normal(float x, float miu,float sigma) {
	return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

float randn(float miu,float sigma, float min ,float max) {
	float x, y, dScope;
	do {
		x = rand(min,max);
		y = normal(x,miu,sigma);
		dScope=rand(0.0,normal(miu,miu,sigma));
	} while (dScope > y);
	return x;
}

void norm(float *con, int dimension) {
	float x = 0;
	for (int  ii = 0; ii < dimension; ii++)
		x += (*(con + ii)) * (*(con + ii));
	x = sqrt(x);
	if (x>1)
		for (int ii=0; ii < dimension; ii++)
			*(con + ii) /= x;
}

void norm(float *con, float *matrix, int *sparse) {
	float tmp, x = 0;
	int last = 0, lastM = 0;
	for (int ii = 0; ii < dimensionR; ii++) {
		tmp = 0;
		for (int i = sparse[last]; i >= 1; i--)
			tmp += matrix[lastM + sparse[last + i]] * con[sparse[last + i]];
		x += tmp * tmp;
		last += sparse[last] + 1;
		lastM += dimension;
	}
	if (x > 1) {
		float lambda = 1;
		last = 0; lastM = 0;
		for (int ii = 0; ii < dimensionR; ii++) {
			tmp = 0;
			for (int jj = sparse[last]; jj >= 1; jj--)
				tmp += matrix[lastM + sparse[last + jj]] * con[sparse[last + jj]];
			tmp = tmp + tmp;
			for (int jj = sparse[last]; jj >= 1; jj--) {
				matrix[lastM + sparse[last + jj]] -= tranSparseAlpha * lambda * tmp * con[sparse[last + jj]];
				con[sparse[last + jj]] -= tranSparseAlpha * lambda * tmp * matrix[lastM + sparse[last + jj]];
			}
			last += sparse[last] + 1;
			lastM += dimension;
		}
	}
}

int relationTotal, entityTotal, tripleTotal;
int *freqRel, *freqEnt;
float *left_mean, *right_mean;
float *relationVec, *entityVec, *matrixHead, *matrixTail;
float *relationVecDao, *entityVecDao, *matrixHeadDao, *matrixTailDao;
float *tmpValue;
int *sparse_id_l, *sparse_id_r, *sparse_pos_l, *sparse_pos_r;

void norm(int h, int t, int r, int j, int tip) {
		norm(relationVecDao + dimensionR * r, dimensionR);
		norm(entityVecDao + dimension * h, dimension);
		norm(entityVecDao + dimension * t, dimension);
		norm(entityVecDao + dimension * j, dimension);
		norm(entityVecDao + dimension * h, matrixHeadDao + dimension * dimensionR * r, sparse_id_l + sparse_pos_l[r]);
		norm(entityVecDao + dimension * t, matrixTailDao + dimension * dimensionR * r, sparse_id_r + sparse_pos_r[r]);
		if (tip == 1)
			norm(entityVecDao + dimension * j, matrixHeadDao + dimension * dimensionR * r, sparse_id_l + sparse_pos_l[r]);
		else
			norm(entityVecDao + dimension * j, matrixTailDao + dimension * dimensionR * r, sparse_id_r + sparse_pos_r[r]);
}

/*
	Read triples from the training file.
*/

void init() {

	FILE *fin;
	int tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &relationTotal);
	fclose(fin);

	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &entityTotal);
	fclose(fin);

	relationVec = (float *)calloc(relationTotal * dimensionR * 2 + entityTotal * dimension * 2 + relationTotal * dimension * dimensionR * 4, sizeof(float));
	relationVecDao = relationVec + relationTotal * dimensionR;
	entityVec = relationVecDao + relationTotal * dimensionR;
	entityVecDao = entityVec + entityTotal * dimension;
	matrixHead = entityVecDao + entityTotal * dimension;
	matrixHeadDao = matrixHead + dimension * dimensionR * relationTotal;
	matrixTail = matrixHeadDao + dimension * dimensionR * relationTotal;
	matrixTailDao = matrixTail + dimension * dimensionR * relationTotal;

	freqRel = (int *)calloc(relationTotal + entityTotal, sizeof(int));
	freqEnt = freqRel + relationTotal;

	for (int i = 0; i < relationTotal; i++) {
		for (int ii=0; ii < dimensionR; ii++)
			relationVec[i * dimensionR + ii] = randn(0, 1.0 / dimensionR, -6 / sqrt(dimensionR), 6 / sqrt(dimensionR));
	}
	for (int i = 0; i < entityTotal; i++) {
		for (int ii=0; ii < dimension; ii++)
			entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
		norm(entityVec + i * dimension, dimension);
	}

	for (int i = 0; i < relationTotal; i++)
		for (int j = 0; j < dimensionR; j++)
			for (int k = 0; k < dimension; k++) {
				matrixHead[i * dimension * dimensionR + j * dimension + k] =  randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
				matrixTail[i * dimension * dimensionR + j * dimension + k] =  randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
			}

	fin = fopen((inPath + "triple2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &tripleTotal);
	trainHead = (Triple *)calloc(tripleTotal * 3, sizeof(Triple));
	trainTail = trainHead + tripleTotal;
	trainList = trainTail + tripleTotal;
	for (int i = 0; i < tripleTotal; i++) {
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

	lefHead = (int *)calloc(entityTotal * 4, sizeof(int));
	rigHead = lefHead + entityTotal;
	lefTail = rigHead + entityTotal;
	rigTail = lefTail + entityTotal;
	memset(rigHead, -1, sizeof(int)*entityTotal);
	memset(rigTail, -1, sizeof(int)*entityTotal);
	for (int i = 1; i < tripleTotal; i++) {
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

	left_mean = (float *)calloc(relationTotal * 2,sizeof(float));
	right_mean = left_mean + relationTotal;
	for (int i = 0; i < entityTotal; i++) {
		for (int j = lefHead[i] + 1; j < rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (int j = lefTail[i] + 1; j < rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	for (int i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];
		right_mean[i] = freqRel[i] / right_mean[i];
	}
	for (int i = 0; i < relationTotal; i++)
		for (int j = 0; j < dimensionR; j++)
			for (int k = 0; k < dimension; k++)
				if (j == k) {
					matrixHead[i * dimension * dimensionR + j * dimension + k] = 1;
					matrixTail[i * dimension * dimensionR + j * dimension + k] = 1;
				}
				else {
					matrixHead[i * dimension * dimensionR + j * dimension + k] = 0;
					matrixTail[i * dimension * dimensionR + j * dimension + k] = 0;
				}
	
	FILE* f1 = fopen((inPath + "tranSparsedata/entity2vec.bern").c_str(),"r");
	for (int i = 0; i < entityTotal; i++) {
		for (int ii = 0; ii < dimension; ii++)
			tmp = fscanf(f1, "%f", &entityVec[i * dimension + ii]);
		norm(entityVec + i * dimension, dimension);
	}
	fclose(f1);
	FILE* f2 = fopen((inPath + "tranSparsedata/relation2vec.bern").c_str(),"r");
	for (int i=0; i < relationTotal; i++) {
		for (int ii=0; ii < dimension; ii++)
			tmp = fscanf(f2, "%f", &relationVec[i * dimensionR + ii]);
	}
	fclose(f2);

	int numLef, numRig;

	FILE* f_d_l = fopen((inPath + "set_num_l.txt").c_str(), "r");
	fscanf(f_d_l, "%d", &numLef);
	sparse_id_l = (int *)calloc(numLef, sizeof(int));
	sparse_pos_l = (int *)calloc(relationTotal, sizeof(int));
	for (int i = 0; i < numLef; i++)
		tmp = fscanf(f_d_l, "%d", &sparse_id_l[i]);
	for (int i = 0, last = 0; i < relationTotal; i++) {
		sparse_pos_l[i] = last;
		for (int j = 0; j < dimensionR; j++)
			last = last + sparse_id_l[last] + 1;
	}
	fclose(f_d_l);

	FILE* f_d_r = fopen((inPath + "set_num_r.txt").c_str(), "r");
	fscanf(f_d_r, "%d", &numRig);
	sparse_id_r = (int *)calloc(numRig, sizeof(int));
	sparse_pos_r = (int *)calloc(relationTotal, sizeof(int));
	for (int i = 0; i < numRig; i++)
		tmp = fscanf(f_d_r, "%d", &sparse_id_r[i]);
	for (int i = 0, last = 0; i < relationTotal; i++) {
		sparse_pos_r[i] = last;
		for (int j = 0; j < dimensionR; j++)
			last = last + sparse_id_r[last] + 1;
	}
	fclose(f_d_r);
}

/*
	Training process of tranSparse.
*/

int tranSparseLen;
int tranSparseBatch;
float res;

double calc_sum(int e1, int e2, int rel, float *tmp1, float *tmp2) {
	int lastM = rel * dimensionR * dimension;
	int last1 = e1 * dimension;
	int last2 = e2 * dimension;
	int lastR = rel * dimensionR;
	int lastl = sparse_pos_l[rel], lastr = sparse_pos_r[rel];
	float sum = 0;
	for (int i = 0; i < dimensionR; i++) {
		tmp1[i] = 0;
		for (int jj = sparse_id_l[lastl]; jj >= 1; jj--) {
			int j = sparse_id_l[lastl+jj];
			tmp1[i] += matrixHead[lastM + j] * entityVec[last1 + j];
		}
		tmp2[i] = 0;
		for (int jj = sparse_id_l[lastr]; jj >= 1; jj--) {
			int j = sparse_id_r[lastr+jj];
			tmp2[i] += matrixTail[lastM + j] * entityVec[last2 + j];
		}
		lastM += dimension;
		lastl += sparse_id_l[lastl] + 1;
		lastr += sparse_id_r[lastr] + 1;
		sum += fabs(tmp1[i] + relationVec[lastR + i] - tmp2[i]);
	}
	return sum;
}

void gradient(int e1_a, int e2_a, int rel_a, int belta, int same, float *tmp1, float *tmp2) {
	int lasta1 = e1_a * dimension;
	int lasta2 = e2_a * dimension;
	int lastar = rel_a * dimensionR;
	int lastM = rel_a * dimensionR * dimension;
	int lastl = sparse_pos_l[rel_a], lastr = sparse_pos_r[rel_a];
	float x;
	for (int ii=0; ii < dimensionR; ii++) {
		x = tmp2[ii] - tmp1[ii] - relationVec[lastar + ii];
		if (x > 0)
			x = belta * tranSparseAlpha;
		else
			x = -belta * tranSparseAlpha;
		for (int j = sparse_id_l[lastl]; j >= 1; j--) {
			int jj = sparse_id_l[lastl + j];
			matrixHeadDao[lastM + jj] -=  x * (entityVec[lasta1 + jj]);
			entityVecDao[lasta1 + jj] -= x * matrixHead[lastM + jj];
		}
		for (int j = sparse_id_r[lastr]; j >= 1; j--) {
			int jj = sparse_id_r[lastr + j];
			matrixTailDao[lastM + jj] -=  x * (-entityVec[lasta2 + jj]);
			entityVecDao[lasta2 + jj] += x * matrixTail[lastM + jj];
		}
		relationVecDao[lastar + ii] -= same * x;
		lastM = lastM + dimension;
		lastl += sparse_id_l[lastl] + 1;
		lastr += sparse_id_r[lastr] + 1;
	}
}

void train_kb(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b, float *tmp) {
	float sum1 = calc_sum(e1_a, e2_a, rel_a, tmp, tmp + dimensionR);
	float sum2 = calc_sum(e1_b, e2_b, rel_b, tmp + dimensionR * 2, tmp + dimensionR * 3);
	if (sum1 + margin > sum2) {
		res += margin + sum1 - sum2;
		gradient(e1_a, e2_a, rel_a, -1, 1, tmp, tmp + dimensionR);
    	gradient(e1_b, e2_b, rel_b, 1, 1, tmp + dimensionR * 2, tmp + dimensionR * 3);
	}
}

int corrupt_head(int id, int h, int r) {
	int lef, rig, mid, ll, rr;
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
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
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

int corrupt_tail(int id, int t, int r) {
	int lef, rig, mid, ll, rr;
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
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
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

void* tranSparsetrainMode(void *con) {
	int id, i, j, pr, tip;
	id = (unsigned long long)(con);
	next_random[id] = rand();
	float *tmp = tmpValue + id * dimensionR * 4;
	for (int k = tranSparseBatch / tranSparseThreads; k >= 0; k--) {
		i = rand_max(id, tranSparseLen);	
		if (bern)
			pr = 1000*right_mean[trainList[i].r]/(right_mean[trainList[i].r]+left_mean[trainList[i].r]);
		else
			pr = 500;
		if (randd(id) % 1000 < pr) {
			j = corrupt_head(id, trainList[i].h, trainList[i].r);
			tip = 0;
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, trainList[i].h, j, trainList[i].r, tmp);
		} else {
			j = corrupt_tail(id, trainList[i].t, trainList[i].r);
			tip = 1;
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, j, trainList[i].t, trainList[i].r, tmp);
		}
		norm(trainList[i].h, trainList[i].t, trainList[i].r, j, tip);
	}
	pthread_exit(NULL);
}

void* train_tranSparse(void *con) {
	tranSparseLen = tripleTotal;
	tranSparseBatch = tranSparseLen / nbatches;
	next_random = (unsigned long long *)calloc(tranSparseThreads, sizeof(unsigned long long));
	tmpValue = (float *)calloc(tranSparseThreads * dimensionR * 4, sizeof(float));
	memcpy(relationVecDao, relationVec, dimensionR * relationTotal * sizeof(float));
	memcpy(entityVecDao, entityVec, dimension * entityTotal * sizeof(float));
	memcpy(matrixHeadDao, matrixHead, dimension * relationTotal * dimensionR * sizeof(float));
	memcpy(matrixTailDao, matrixTail, dimension * relationTotal * dimensionR * sizeof(float));
	for (int epoch = 0; epoch < tranSparseTrainTimes; epoch++) {
		res = 0;
		for (int batch = 0; batch < nbatches; batch++) {
			pthread_t *pt = (pthread_t *)malloc(tranSparseThreads * sizeof(pthread_t));
			for (long a = 0; a < tranSparseThreads; a++)
				pthread_create(&pt[a], NULL, tranSparsetrainMode,  (void*)a);
			for (long a = 0; a < tranSparseThreads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
			memcpy(relationVec, relationVecDao, dimensionR * relationTotal * sizeof(float));
			memcpy(entityVec, entityVecDao, dimension * entityTotal * sizeof(float));
			memcpy(matrixHead, matrixHeadDao, dimension * relationTotal * dimensionR * sizeof(float));
			memcpy(matrixTail, matrixTailDao, dimension * relationTotal * dimensionR * sizeof(float));
		}
		printf("epoch %d %f\n", epoch, res);
	}
	pthread_exit(NULL);
}

/*
	Get the results of tranSparse.
*/

void out_tranSparse() {
		FILE* f2 = fopen((outPath + "relation2vec.vec").c_str(), "w");
		FILE* f3 = fopen((outPath + "entity2vec.vec").c_str(), "w");
		for (int i = 0; i < relationTotal; i++) {
			int last = dimension * i;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f2, "%.6f\t", relationVec[last + ii]);
			fprintf(f2,"\n");
		}
		for (int  i = 0; i < entityTotal; i++) {
			int last = i * dimension;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f3, "%.6f\t", entityVec[last + ii] );
			fprintf(f3,"\n");
		}
		fclose(f2);
		fclose(f3);
		FILE* f1 = fopen((outPath + "A.vec").c_str(),"w");
		for (int i = 0; i < relationTotal; i++)
			for (int jj = 0; jj < dimension; jj++) {
				for (int ii = 0; ii < dimensionR; ii++)
					fprintf(f1, "%f\t", matrixHead[i * dimensionR * dimension + jj + ii * dimension]);
				fprintf(f1,"\n");
			}
		for (int i = 0; i < relationTotal; i++)
			for (int jj = 0; jj < dimension; jj++) {
				for (int ii = 0; ii < dimensionR; ii++)
					fprintf(f1, "%f\t", matrixTail[i * dimensionR * dimension + jj + ii * dimension]);
				fprintf(f1,"\n");
			}
		fclose(f1);
}

/*
	Main function
*/

int main() {
	init();
	train_tranSparse(NULL);
	out_tranSparse();
	return 0;
}