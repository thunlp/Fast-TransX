#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>
#include <pthread.h>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <vector>
#include <tuple>
#include "hashing.h"
#include <boost/functional/hash.hpp>
using namespace std;

const double pi = 3.141592653589793238462643383;

int ptranseThreads = 1;
int ptranseTrainTimes = 3000;
int nbatches = 1;
int dimension = 50;
double ptranseAlpha = 0.001;
double margin = 1;

string inPath = "data/";
string outPath = "res/";

int *lefHead, *rigHead;
int *lefTail, *rigTail;

struct Triple {
    int h, r, t;
    list<pair<vector<int>, double> > pathList;
};

Triple **trainHead, **trainTail, ** trainList;

// restore (h, r, t)s
unordered_set<tuple<int, int, int>> isInTrain;
unordered_map<pair<vector<int>, int>, double, pair_hash> pathConfidence;


struct cmp_head {
    bool operator()(const Triple* const &a, const Triple* const &b) {
        return (a->h < b->h)||(a->h == b->h && a->r < b->r)||(a->h == b->h && a->r == b->r && a->t < b->t);
    }
};

struct cmp_tail {
    bool operator()(const Triple* const &a, const Triple* const &b) {
        return (a->t < b->t)||(a->t == b->t && a->r < b->r)||(a->t == b->t && a->r == b->r && a->h < b->h);
    }
};

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
    while (res<0)
        res+=x;
    return res;
}

double rand(double min, double max) {
    return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

double normal(double x, double miu,double sigma) {
    return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

double randn(double miu,double sigma, double min ,double max) {
    double x, y, dScope;
    do {
        x = rand(min,max);
        y = normal(x,miu,sigma);
        dScope=rand(0.0,normal(miu,miu,sigma));
    } while (dScope > y);
    return x;
}

void norm(double * con) {
    double x = 0;
    for (int  ii = 0; ii < dimension; ii++)
        x += (*(con + ii)) * (*(con + ii));
    x = sqrt(x);
    if (x>1)
        for (int ii=0; ii < dimension; ii++)
            *(con + ii) /= x;
}

double sqr(double x){
    return x * x;
}

/*
    Read triples from the training file.
*/

int relationTotal, entityTotal, tripleTotal;
double *relationVec, *entityVec;

int ptranseBatch, ptranseLen;
double res;

void gradient(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
    int lasta1 = e1_a * dimension;
    int lasta2 = e2_a * dimension;
    int lastar = rel_a * dimension;
    int lastb1 = e1_b * dimension;
    int lastb2 = e2_b * dimension;
    int lastbr = rel_b * dimension;
    for (int ii=0; ii  < dimension; ii++) {
        double x;
        x = (entityVec[lasta2 + ii] - entityVec[lasta1 + ii] - relationVec[lastar + ii]);
        if (x > 0)
            x = -ptranseAlpha;
        else
            x = ptranseAlpha;
        relationVec[lastar + ii] -= x;
        entityVec[lasta1 + ii] -= x;
        entityVec[lasta2 + ii] += x;
        x = (entityVec[lastb2 + ii] - entityVec[lastb1 + ii] - relationVec[lastbr + ii]);
        if (x > 0)
            x = ptranseAlpha;
        else
            x = -ptranseAlpha;
        relationVec[lastbr + ii] -=  x;
        entityVec[lastb1 + ii] -= x;
        entityVec[lastb2 + ii] += x;
    }
}

double calc_sum(int e1, int e2, int rel) {
    double sum=0;
    int last1 = e1 * dimension;
    int last2 = e2 * dimension;
    int lastr = rel * dimension;
    for (int ii=0; ii < dimension; ii++) {
                    sum += fabs(entityVec[last2 + ii] - entityVec[last1 + ii] - relationVec[lastr + ii]);
                }
    return sum;
}

void train_kb(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
    double sum1 = calc_sum(e1_a, e2_a, rel_a);
    double sum2 = calc_sum(e1_b, e2_b, rel_b);
    if (sum1 + margin > sum2) {
        res += margin + sum1 - sum2;
        gradient(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
    }
}

int corrupt_head(int id, int h, int r) {
    int lef, rig, mid, ll, rr;
    lef = lefHead[h] - 1;
    rig = rigHead[h];
    while (lef + 1 < rig) {
        mid = (lef + rig) >> 1;
        if (trainHead[mid]->r >= r) rig = mid; else
        lef = mid;
    }
    ll = rig;
    lef = lefHead[h];
    rig = rigHead[h] + 1;
    while (lef + 1 < rig) {
        mid = (lef + rig) >> 1;
        if (trainHead[mid]->r <= r) lef = mid; else
        rig = mid;
    }
    rr = lef;
    int tmp = rand_max(id, entityTotal - (rr - ll + 1));
    if (tmp < trainHead[ll]->t) return tmp;
    if (tmp > trainHead[rr]->t - rr + ll - 1) return tmp + rr - ll + 1;
    lef = ll, rig = rr + 1;
    while (lef + 1 < rig) {
        mid = (lef + rig) >> 1;
        if (trainHead[mid]->t - mid + ll - 1 < tmp)
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
        if (trainTail[mid]->r >= r) rig = mid; else
        lef = mid;
    }
    ll = rig;
    lef = lefTail[t];
    rig = rigTail[t] + 1;
    while (lef + 1 < rig) {
        mid = (lef + rig) >> 1;
        if (trainTail[mid]->r <= r) lef = mid; else
        rig = mid;
    }
    rr = lef;
    int tmp = rand_max(id, entityTotal - (rr - ll + 1));
    if (tmp < trainTail[ll]->h) return tmp;
    if (tmp > trainTail[rr]->h - rr + ll - 1) return tmp + rr - ll + 1;
    lef = ll, rig = rr + 1;
    while (lef + 1 < rig) {
        mid = (lef + rig) >> 1;
        if (trainTail[mid]->h - mid + ll - 1 < tmp)
            lef = mid;
        else
            rig = mid;
    }
    return tmp + lef - ll + 1;
}

double calc_path(int r1, const vector<int>& relPath) {
    double sum = 0;
    for (int ii = 0; ii < dimension; ii++) {
        double tmp = relationVec[r1 * dimension + ii];
        for (auto &j : relPath)
            tmp -= relationVec[j * dimension + ii];
            // L1 norm as default
            sum+=fabs(tmp);
    }
    return sum;
}

void gradient_path(int r1, const vector<int>& relPath, double belta) {
    for (int ii=0; ii < dimension; ii++) {
        double x = relationVec[r1 * dimension + ii];
        for (auto &j : relPath)
            x -= relationVec[j * dimension + ii];
        if (x>0) x=1;
            else x=-1;
        relationVec[r1 * dimension + ii]+=belta*ptranseAlpha*x;
        for (auto &j : relPath)
            relationVec[j * dimension + ii]-=belta*ptranseAlpha*x;
    }
}

void train_path(int rel, int relNeg, const vector<int>& relPath, double margin,double x) {
    double sum1 = calc_path(rel,relPath);
    double sum2 = calc_path(relNeg,relPath);
    double lambda = 1;
    if (sum1+margin>sum2) {
        res+=x*lambda*(margin+sum1-sum2);
        gradient_path(rel,relPath, -x*lambda);
        gradient_path(relNeg,relPath, x*lambda);
    }
}

void* ptransetrainMode(void *con) {
    int id;
    id = (unsigned long long)(con);
    next_random[id] = rand();
    for (int k = ptranseBatch / ptranseThreads; k >= 0; k--) {
        int i = rand_max(id, ptranseLen);

        // transe part
        int j; // corrupted part
        int pr = 500;
        if (randd(id) % 1000 < pr){
            j = corrupt_head(id, trainList[i]->h, trainList[i]->r);
            train_kb(trainList[i]->h, trainList[i]->t, trainList[i]->r, trainList[i]->h, j, trainList[i]->r);
        }
        else {
            j = corrupt_tail(id, trainList[i]->t, trainList[i]->r);
            train_kb(trainList[i]->h, trainList[i]->t, trainList[i]->r, j, trainList[i]->t, trainList[i]->r);
        }

        // ptranse part
        j = rand_max(id, relationTotal);
        while (isInTrain.find(make_tuple(trainList[i]->h, j, trainList[i]->t)) != isInTrain.end())
            j = rand_max(id, relationTotal);
        for (auto & pathList : trainList[i]->pathList){
            vector<int> relPath = pathList.first;
            double pr = pathList.second;
            double pr_path = 0;
            if (pathConfidence.count(make_pair(relPath, trainList[i]->r))>0)
                pr_path = pathConfidence[make_pair(relPath, trainList[i]->r)];
            pr_path = 0.99*pr_path + 0.01;
            train_path(trainList[i]->r, j, relPath, 2*margin, pr*pr_path);
        }

        //norm
        norm(relationVec + dimension * trainList[i]->r);
        norm(entityVec + dimension * trainList[i]->h);
        norm(entityVec + dimension * trainList[i]->t);
        norm(entityVec + dimension * j);
    }
}

void* train_ptranse(void *con) {
    ptranseLen = tripleTotal;
    ptranseBatch = ptranseLen / nbatches;
    next_random = (unsigned long long *)calloc(ptranseThreads, sizeof(unsigned long long));
    for (int epoch = 0; epoch < ptranseTrainTimes; epoch++) {
        printf("epoch %d started.\n", epoch);
        res = 0;
        for (int batch = 0; batch < nbatches; batch++) {
            pthread_t *pt = (pthread_t *)malloc(ptranseThreads * sizeof(pthread_t));
            for (int a = 0; a < ptranseThreads; a++)
                pthread_create(&pt[a], NULL, ptransetrainMode,  (void*)a);
            for (int a = 0; a < ptranseThreads; a++)
                pthread_join(pt[a], NULL);
            free(pt);
        }
        printf("epoch %d %f\n", epoch, res);
    }
}

void init() {

    FILE *fin;
    int tmp;

    // fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    // tmp = fscanf(fin, "%d", &relationTotal);
    // fclose(fin);
    relationTotal = 1345;
    printf("Relations:\t%d\n", relationTotal);

    relationTotal <<= 1;
    relationVec = (double *)calloc(relationTotal * dimension, sizeof(double));
    for (int i = 0; i < relationTotal; i++) {
        for (int ii=0; ii<dimension; ii++)
            relationVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
    }

    // fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    // tmp = fscanf(fin, "%d", &entityTotal);
    // fclose(fin);
    entityTotal = 14951;
    printf("Entities:\t%d\n", entityTotal);

    entityVec = (double *)calloc(entityTotal * dimension, sizeof(double));
    for (int i = 0; i < entityTotal; i++) {
        for (int ii=0; ii<dimension; ii++)
            entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
        norm(entityVec+i*dimension);
    }
    printf("Entities' vectors initialized.\n");

    fin = fopen((inPath + "train_pra.txt").c_str(), "r");
    tmp = fscanf(fin, "%d", &tripleTotal);
    printf("Triples:\t%d\n", tripleTotal);
    trainHead = (Triple **)calloc(tripleTotal, sizeof(Triple*));
    trainTail = (Triple **)calloc(tripleTotal, sizeof(Triple*));
    trainList = (Triple **)calloc(tripleTotal, sizeof(Triple*));
    tripleTotal = 0;
    // establish new triple in advance
    trainHead[tripleTotal] = new Triple();
    trainTail[tripleTotal] = new Triple();
    trainList[tripleTotal] = new Triple();
    while (fscanf(fin, "%d", &trainList[tripleTotal]->h) == 1) {

        //input (h, t, r)
        tmp = fscanf(fin, "%d", &trainList[tripleTotal]->t);
        tmp = fscanf(fin, "%d", &trainList[tripleTotal]->r);


        //input paths
        //input path_amount
        int pathsAmount;
        tmp = fscanf(fin, "%d", &pathsAmount);
        //input each path
        for(int i = 0;i<pathsAmount;i++){
            int pathLength, pathElement;
            double pathProbability;
            vector<int> relPath;
            relPath.clear();
            tmp = fscanf(fin, "%d", &pathLength);
            for(int j = 0;j<pathLength;j++){
                tmp = fscanf(fin, "%d", &pathElement);
                relPath.push_back(pathElement);
            }
            tmp = fscanf(fin, "%lf", &pathProbability);
            trainList[tripleTotal]->pathList.push_back(make_pair(relPath, pathProbability));
        }

        //put (h, r, t) into training set
        isInTrain.insert(make_tuple(trainList[tripleTotal]->h, trainList[tripleTotal]->r, trainList[tripleTotal]->t));

        //copy trainList to trainHead trainTail
        (*trainHead[tripleTotal]) = (*trainTail[tripleTotal]) = (*trainList[tripleTotal]);
        tripleTotal++;
        trainHead[tripleTotal] = new Triple();
        trainTail[tripleTotal] = new Triple();
        trainList[tripleTotal] = new Triple();

    }
    fclose(fin);

    sort(trainHead, trainHead + tripleTotal, cmp_head());
    sort(trainTail, trainTail + tripleTotal, cmp_tail());

    lefHead = (int *)calloc(entityTotal, sizeof(int));
    rigHead = (int *)calloc(entityTotal, sizeof(int));
    lefTail = (int *)calloc(entityTotal, sizeof(int));
    rigTail = (int *)calloc(entityTotal, sizeof(int));
    memset(rigHead, -1, entityTotal * sizeof(int));
    memset(rigTail, -1, entityTotal * sizeof(int));
    memset(lefHead, -1, entityTotal * sizeof(int));
    memset(lefTail, -1, entityTotal * sizeof(int));
    // tripleTotal should be larger than 0, otherwise could cause errors.
    lefTail[trainTail[0]->t] = 0;
    lefHead[trainHead[0]->h] = 0;
    for (int i = 1; i < tripleTotal; i++) {
        if (trainTail[i]->t != trainTail[i - 1]->t) {
            rigTail[trainTail[i - 1]->t] = i - 1;
            lefTail[trainTail[i]->t] = i;
        }
        if (trainHead[i]->h != trainHead[i - 1]->h) {
            rigHead[trainHead[i - 1]->h] = i - 1;
            lefHead[trainHead[i]->h] = i;
        }
    }
    rigHead[trainHead[tripleTotal - 1]->h] = tripleTotal - 1;
    rigTail[trainTail[tripleTotal - 1]->t] = tripleTotal - 1;

    // input confidence file
    fin = fopen((inPath + "confidence.txt").c_str(), "r");
    int pathLength;
    while (fscanf(fin, "%d", &pathLength)==1){
        int pathElement;
        vector<int> relPath;
        relPath.clear();
        for (int i=0; i<pathLength; i++)
        {
            tmp = fscanf(fin, "%d", &pathElement);
            relPath.push_back(pathElement);
        }
        int relationsAmount;
        fscanf(fin, "%d", &relationsAmount);
        for (int i=0; i<relationsAmount; i++)
        {
            int relation;
            double pr;
            tmp = fscanf(fin, "%d%lf", &relation, &pr);
            pathConfidence[make_pair(relPath, relation)] = pr;
        }
    }
    fclose(fin);
    printf("Initialization completed.\n");
}

void destruct(){
    for(int i = 0;i<=tripleTotal;i++){
        delete(trainHead[i]);
        delete(trainTail[i]);
        delete(trainList[i]);
    }
}

void out_ptranse() {
		FILE* f2 = fopen((outPath + "relation2vec.bern").c_str(), "w");
		FILE* f3 = fopen((outPath + "entity2vec.bern").c_str(), "w");
		for (int i=0; i < relationTotal; i++) {
			int last = dimension * i;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f2, "%.6lf\t", relationVec[last + ii]);
			fprintf(f2,"\n");
		}
		for (int  i = 0; i < entityTotal; i++) {
			int last = i * dimension;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f3, "%.6lf\t", entityVec[last + ii] );
			fprintf(f3,"\n");
		}
		fclose(f2);
		fclose(f3);
}

int main() {
    init();
    train_ptranse(NULL);
    out_ptranse();
    destruct();
    return 0;
}
