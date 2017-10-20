#include <iostream>
#include <cstring>
#include <cstdio>
#include <map>
#include <vector>
#include <string>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <cstdlib>

using namespace std;

int relationTotal;
int entityTotal;
int Threads = 8;
int dimensionR = 100;
int dimension = 100;

float *entityVec, *relationVec, *entityRelVec, *entityTransVec, *relationTransVec, *norm_vec;
int testTotal, tripleTotal, trainTotal, validTotal;

struct Triple {
    int h, r, t;
};

struct cmp_head {
    bool operator()(const Triple &a, const Triple &b) {
        return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
    }
};

Triple *testList, *tripleList;

void init() {
    FILE *fin;
    int tmp, h, r, t;

    fin = fopen("data/relation2id.txt", "r");
    tmp = fscanf(fin, "%d", &relationTotal);
    fclose(fin);
    relationVec = (float *)calloc(relationTotal * dimensionR, sizeof(float));
    relationTransVec = (float *)calloc(relationTotal * dimensionR, sizeof(float));
    
    fin = fopen("data/entity2id.txt", "r");
    tmp = fscanf(fin, "%d", &entityTotal);
    fclose(fin);
    entityVec = (float *)calloc(entityTotal * dimension, sizeof(float));
    entityTransVec = (float *)calloc(entityTotal * dimension, sizeof(float));
    norm_vec = (float *)calloc(relationTotal * dimensionR, sizeof(float));
    

    FILE* f_kb1 = fopen("data/test2id.txt","r");
    FILE* f_kb2 = fopen("data/triple2id.txt","r");
    FILE* f_kb3 = fopen("data/valid2id.txt","r");
    tmp = fscanf(f_kb1, "%d", &testTotal);
    tmp = fscanf(f_kb2, "%d", &trainTotal);
    tmp = fscanf(f_kb3, "%d", &validTotal);
    tripleTotal = testTotal + trainTotal + validTotal;
    testList = (Triple *)calloc(testTotal, sizeof(Triple));
    tripleList = (Triple *)calloc(tripleTotal, sizeof(Triple));

    for (int i = 0; i < testTotal; i++) {
        tmp = fscanf(f_kb1, "%d", &h);
        tmp = fscanf(f_kb1, "%d", &t);
        tmp = fscanf(f_kb1, "%d", &r);
        testList[i].h = h;
        testList[i].t = t;
        testList[i].r = r;
        tripleList[i].h = h;
        tripleList[i].t = t;
        tripleList[i].r = r;
    }

    for (int i = 0; i < trainTotal; i++) {
        tmp = fscanf(f_kb2, "%d", &h);
        tmp = fscanf(f_kb2, "%d", &t);
        tmp = fscanf(f_kb2, "%d", &r);
        tripleList[i + testTotal].h = h;
        tripleList[i + testTotal].t = t;
        tripleList[i + testTotal].r = r;
    }

    for (int i = 0; i < validTotal; i++) {
        tmp = fscanf(f_kb3, "%d", &h);
        tmp = fscanf(f_kb3, "%d", &t);
        tmp = fscanf(f_kb3, "%d", &r);
        tripleList[i + testTotal + trainTotal].h = h;
        tripleList[i + testTotal + trainTotal].t = t;
        tripleList[i + testTotal + trainTotal].r = r;
    }
    
    fclose(f_kb1);
    fclose(f_kb2);
    fclose(f_kb3);

    sort(tripleList, tripleList + tripleTotal, cmp_head());
}

void prepare() {
    FILE *fin;
    int tmp;
    fin = fopen("entity2vec.txt", "r");
    for (int i = 0; i < entityTotal; i++) {
        int last = i * dimension;
        for (int j = 0; j < dimension; j++)
            tmp = fscanf(fin, "%f", &entityVec[last + j]);
    }
    fclose(fin);
    fin = fopen("relation2vec.txt", "r");
    for (int i = 0; i < relationTotal; i++) {
        int last = i * dimensionR;
        for (int j = 0; j < dimensionR; j++)
            tmp = fscanf(fin, "%f", &relationVec[last + j]);
    }
    fclose(fin);

    fin = fopen("norm2vec.txt", "r");
    for (int i=0; i < relationTotal; i++) {
        int last = dimensionR * i;
        for (int ii = 0; ii < dimensionR; ii++)
            tmp = fscanf(fin, "%f", &norm_vec[last + ii]);
    }
    fclose(fin);
}

float calc_sum(int e1, int e2, int rel) {
    int lasta1 = e1 * dimension;
    int lasta2 = e2 * dimension;
    int lastRel = rel * dimension;
    float tmp1 = 0, tmp2 = 0;
    for (int i = 0; i < dimension; i++) {
        tmp1 += norm_vec[lastRel + i] * entityVec[lasta1 + i];
        tmp2 += norm_vec[lastRel + i] * entityVec[lasta2 + i];
    }
    float sum = 0;
    for (int i = 0; i < dimension; i++) {
        float gg = (entityVec[lasta2 + i] - tmp2 * norm_vec[lastRel + i]) - (entityVec[lasta1 + i] - tmp1 * norm_vec[lastRel + i]) - relationVec[lastRel + i];
        sum += fabs(gg);
    }
    return sum;
}

bool find(int h, int t, int r) {
    int lef = 0;
    int rig = tripleTotal - 1;
    int mid;
    while (lef + 1 < rig) {
        int mid = (lef + rig) >> 1;
        if ((tripleList[mid]. h < h) || (tripleList[mid]. h == h && tripleList[mid]. r < r) || (tripleList[mid]. h == h && tripleList[mid]. r == r && tripleList[mid]. t < t)) lef = mid; else rig = mid;
    }
    if (tripleList[lef].h == h && tripleList[lef].r == r && tripleList[lef].t == t) return true;
    if (tripleList[rig].h == h && tripleList[rig].r == r && tripleList[rig].t == t) return true;
    return false;
}

float *l_filter_tot, *r_filter_tot, *l_tot, *r_tot;
float *l_filter_rank, *r_filter_rank, *l_rank, *r_rank;

void* testMode(void *con) {
    int id;
    id = (unsigned long long)(con);
    int lef = testTotal / (Threads) * id;
    int rig = testTotal / (Threads) * (id + 1) - 1;
    if (id == Threads - 1) rig = testTotal - 1;
    for (int i = lef; i <= rig; i++) {
        int h = testList[i].h;
        int t = testList[i].t;
        int r = testList[i].r;
        float minimal = calc_sum(h, t, r);
        int l_filter_s = 0;
        int l_s = 0;
        int r_filter_s = 0;
        int r_s = 0;
        for (int j = 0; j <= entityTotal; j++) {
            if (j != h) {
                float value = calc_sum(j, t, r);
                if (value < minimal) {
                    l_s += 1;
                    if (not find(j, t, r))
                        l_filter_s += 1;
                }
            }
            if (j != t) {
                float value = calc_sum(h, j, r);
                if (value < minimal) {
                    r_s += 1;
                    if (not find(h, j, r))
                        r_filter_s += 1;
                }
            }
        }
        if (l_filter_s < 10) l_filter_tot[id] += 1;
        if (l_s < 10) l_tot[id] += 1;
        if (r_filter_s < 10) r_filter_tot[id] += 1;
        if (r_s < 10) r_tot[id] += 1;

        l_filter_rank[id] += l_filter_s;
        r_filter_rank[id] += r_filter_s;
        l_rank[id] += l_s;
        r_rank[id] += r_s;
    }
}

void* test(void *con) {
    l_filter_tot = (float *)calloc(Threads, sizeof(float));
    r_filter_tot = (float *)calloc(Threads, sizeof(float));
    l_tot = (float *)calloc(Threads, sizeof(float));
    r_tot = (float *)calloc(Threads, sizeof(float));

    l_filter_rank = (float *)calloc(Threads, sizeof(float));
    r_filter_rank = (float *)calloc(Threads, sizeof(float));
    l_rank = (float *)calloc(Threads, sizeof(float));
    r_rank = (float *)calloc(Threads, sizeof(float));

    pthread_t *pt = (pthread_t *)malloc(Threads * sizeof(pthread_t));
    for (int a = 0; a < Threads; a++)
        pthread_create(&pt[a], NULL, testMode,  (void*)a);
    for (int a = 0; a < Threads; a++)
        pthread_join(pt[a], NULL);
    free(pt);
    for (int a = 1; a < Threads; a++) {
        l_filter_tot[a] += l_filter_tot[a - 1];
        r_filter_tot[a] += r_filter_tot[a - 1];
        l_tot[a] += l_tot[a - 1];
        r_tot[a] += r_tot[a - 1];

        l_filter_rank[a] += l_filter_rank[a - 1];
        r_filter_rank[a] += r_filter_rank[a - 1];
        l_rank[a] += l_rank[a - 1];
        r_rank[a] += r_rank[a - 1];
    }

    printf("left %f %f\n", l_rank[Threads - 1] / testTotal, l_tot[Threads - 1] / testTotal);
    printf("left(filter) %f %f\n", l_filter_rank[Threads - 1] / testTotal, l_filter_tot[Threads - 1] / testTotal);
    printf("right %f %f\n", r_rank[Threads - 1] / testTotal, r_tot[Threads - 1] / testTotal);
    printf("right(filter) %f %f\n", r_filter_rank[Threads - 1] / testTotal, r_filter_tot[Threads - 1] / testTotal);
}


int main() {
    init();
    prepare();
    test(NULL);
    return 0;
}
