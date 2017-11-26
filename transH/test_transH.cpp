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
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

using namespace std;
long l1_flag = 1;
long relationTotal;
long entityTotal;
long Threads = 8;
long dimensionR = 100;
long dimension = 100;
long binaryFlag = 0;

float *entityVec, *relationVec, *entityTransVec, *relationTransVec, *norm_vec;
long testTotal, tripleTotal, tralongotal, validTotal;

struct Triple {
    long h, r, t, label;
};

struct cmp_head {
    bool operator()(const Triple &a, const Triple &b) {
        return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
    }
};

Triple *testList, *tripleList;
string initPath = "";
string inPath = "";
string note = "";
int nntotal[5];
int head_lef[10000];
int head_rig[10000];
int tail_lef[10000];
int tail_rig[10000];
int head_type[1000000];
int tail_type[1000000];

void init() {
    FILE *fin;
    long tmp, h, r, t, label;

    fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    tmp = fscanf(fin, "%ld", &relationTotal);
    fclose(fin);
    relationVec = (float *)calloc(relationTotal * dimensionR, sizeof(float));
    relationTransVec = (float *)calloc(relationTotal * dimensionR, sizeof(float));
    
    fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    tmp = fscanf(fin, "%ld", &entityTotal);
    fclose(fin);
    entityVec = (float *)calloc(entityTotal * dimension, sizeof(float));
    entityTransVec = (float *)calloc(entityTotal * dimension, sizeof(float));
    norm_vec = (float *)calloc(relationTotal * dimensionR, sizeof(float));
    

    FILE* f_kb1 = fopen((inPath + "test2id_all.txt").c_str(),"r");
    FILE* f_kb2 = fopen((inPath + "train2id.txt").c_str(),"r");
    FILE* f_kb3 = fopen((inPath + "valid2id.txt").c_str(),"r");
    tmp = fscanf(f_kb1, "%ld", &testTotal);
    tmp = fscanf(f_kb2, "%ld", &tralongotal);
    tmp = fscanf(f_kb3, "%ld", &validTotal);
    tripleTotal = testTotal + tralongotal + validTotal;
    testList = (Triple *)calloc(testTotal, sizeof(Triple));
    tripleList = (Triple *)calloc(tripleTotal, sizeof(Triple));
    memset(nntotal, 0, sizeof(nntotal));
    for (long i = 0; i < testTotal; i++) {
        tmp = fscanf(f_kb1, "%ld", &label);
        tmp = fscanf(f_kb1, "%ld", &h);
        tmp = fscanf(f_kb1, "%ld", &t);
        tmp = fscanf(f_kb1, "%ld", &r);
        label++;
        nntotal[label]++;
        testList[i].label = label;
        testList[i].h = h;
        testList[i].t = t;
        testList[i].r = r;
        tripleList[i].h = h;
        tripleList[i].t = t;
        tripleList[i].r = r;
    }

    for (long i = 0; i < tralongotal; i++) {
        tmp = fscanf(f_kb2, "%ld", &h);
        tmp = fscanf(f_kb2, "%ld", &t);
        tmp = fscanf(f_kb2, "%ld", &r);
        tripleList[i + testTotal].h = h;
        tripleList[i + testTotal].t = t;
        tripleList[i + testTotal].r = r;
    }

    for (long i = 0; i < validTotal; i++) {
        tmp = fscanf(f_kb3, "%ld", &h);
        tmp = fscanf(f_kb3, "%ld", &t);
        tmp = fscanf(f_kb3, "%ld", &r);
        tripleList[i + testTotal + tralongotal].h = h;
        tripleList[i + testTotal + tralongotal].t = t;
        tripleList[i + testTotal + tralongotal].r = r;
    }
    
    fclose(f_kb1);
    fclose(f_kb2);
    fclose(f_kb3);

    sort(tripleList, tripleList + tripleTotal, cmp_head());

    long total_lef = 0;
    long total_rig = 0;
    FILE* f_type = fopen((inPath + "type_constrain.txt").c_str(),"r");
    tmp = fscanf(f_type, "%ld", &tmp);
    for (int i = 0; i < relationTotal; i++) {
        int rel, tot;
        tmp = fscanf(f_type, "%d%d", &rel, &tot);
        head_lef[rel] = total_lef;
        for (int j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%d", &head_type[total_lef]);
            total_lef++;
        }
        head_rig[rel] = total_lef;
        sort(head_type + head_lef[rel], head_type + head_rig[rel]);
        tmp = fscanf(f_type, "%d%d", &rel, &tot);
        tail_lef[rel] = total_rig;
        for (int j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%d", &tail_type[total_rig]);
            total_rig++;
        }
        tail_rig[rel] = total_rig;
        sort(tail_type + tail_lef[rel], tail_type + tail_rig[rel]);
    }
    fclose(f_type);
}

void prepre_binary() {
    struct stat statbuf1;
    if (stat((initPath + "entity2vec" + note + ".bin").c_str(), &statbuf1) != -1) {  
        int fd = open((initPath + "entity2vec" + note + ".bin").c_str(), O_RDONLY);
        float* entityVecTmp = (float*)mmap(NULL, statbuf1.st_size, PROT_READ, MAP_PRIVATE, fd, 0); 
        memcpy(entityVec, entityVecTmp, statbuf1.st_size);
        munmap(entityVecTmp, statbuf1.st_size);
        close(fd);
    }  
    struct stat statbuf2;
    if (stat((initPath + "relation2vec" + note + ".bin").c_str(), &statbuf2) != -1) {  
        int fd = open((initPath + "relation2vec" + note + ".bin").c_str(), O_RDONLY);
        float* relationVecTmp =(float*)mmap(NULL, statbuf2.st_size, PROT_READ, MAP_PRIVATE, fd, 0); 
        memcpy(relationVec, relationVecTmp, statbuf2.st_size);
        munmap(relationVecTmp, statbuf2.st_size);
        close(fd);
    }
    struct stat statbuf3;
    if (stat((initPath + "A" + note + ".bin").c_str(), &statbuf3) != -1) {  
        int fd = open((initPath + "A" + note + ".bin").c_str(), O_RDONLY);
        float* ATmp =(float*)mmap(NULL, statbuf3.st_size, PROT_READ, MAP_PRIVATE, fd, 0); 
        memcpy(norm_vec, ATmp, statbuf3.st_size);
        munmap(ATmp, statbuf3.st_size);
        close(fd);
    }
}

void prepare() {
    if (binaryFlag) {
        prepre_binary();
        return;
    }
    FILE *fin;
    long tmp;
    fin = fopen((initPath + "entity2vec" + note + ".vec").c_str(), "r");
    for (long i = 0; i < entityTotal; i++) {
        long last = i * dimension;
        for (long j = 0; j < dimension; j++)
            tmp = fscanf(fin, "%f", &entityVec[last + j]);
    }
    fclose(fin);
    fin = fopen((initPath + "relation2vec" + note + ".vec").c_str(), "r");
    for (long i = 0; i < relationTotal; i++) {
        long last = i * dimensionR;
        for (long j = 0; j < dimensionR; j++)
            tmp = fscanf(fin, "%f", &relationVec[last + j]);
    }
    fclose(fin);

    fin = fopen((initPath + "A" + note + ".vec").c_str(), "r");
    for (long i=0; i < relationTotal; i++) {
        long last = dimensionR * i;
        for (long ii = 0; ii < dimensionR; ii++)
            tmp = fscanf(fin, "%f", &norm_vec[last + ii]);
    }
    fclose(fin);
}

float calc_sum(long e1, long e2, long rel) {
    long lasta1 = e1 * dimension;
    long lasta2 = e2 * dimension;
    long lastRel = rel * dimension;
    float tmp1 = 0, tmp2 = 0;
    for (long i = 0; i < dimension; i++) {
        tmp1 += norm_vec[lastRel + i] * entityVec[lasta1 + i];
        tmp2 += norm_vec[lastRel + i] * entityVec[lasta2 + i];
    }
    float sum = 0;
    for (long i = 0; i < dimension; i++) {
        float gg = (entityVec[lasta2 + i] - tmp2 * norm_vec[lastRel + i]) - (entityVec[lasta1 + i] - tmp1 * norm_vec[lastRel + i]) - relationVec[lastRel + i];
        sum += fabs(gg);
    }
    return sum;
}

bool find(long h, long t, long r) {
    long lef = 0;
    long rig = tripleTotal - 1;
    long mid;
    while (lef + 1 < rig) {
        long mid = (lef + rig) >> 1;
        if ((tripleList[mid]. h < h) || (tripleList[mid]. h == h && tripleList[mid]. r < r) || (tripleList[mid]. h == h && tripleList[mid]. r == r && tripleList[mid]. t < t)) lef = mid; else rig = mid;
    }
    if (tripleList[lef].h == h && tripleList[lef].r == r && tripleList[lef].t == t) return true;
    if (tripleList[rig].h == h && tripleList[rig].r == r && tripleList[rig].t == t) return true;
    return false;
}

float *l_filter_tot[6], *r_filter_tot[6], *l_tot[6], *r_tot[6];
float *l_filter_rank[6], *r_filter_rank[6], *l_rank[6], *r_rank[6];


void* testMode(void *con) {
    long id;
    id = (unsigned long long)(con);
    long lef = testTotal / (Threads) * id;
    long rig = testTotal / (Threads) * (id + 1) - 1;
    if (id == Threads - 1) rig = testTotal - 1;
    for (long i = lef; i <= rig; i++) {
        long h = testList[i].h;
        long t = testList[i].t;
        long r = testList[i].r;
        long label = testList[i].label;
        float minimal = calc_sum(h, t, r);
        long l_filter_s = 0;
        long l_s = 0;
        long r_filter_s = 0;
        long r_s = 0;
        long l_filter_s_constrain = 0;
        long l_s_constrain = 0;
        long r_filter_s_constrain = 0;
        long r_s_constrain = 0;
        long type_head = head_lef[r], type_tail = tail_lef[r];
        for (long j = 0; j <= entityTotal; j++) {
            if (j != h) {
                float value = calc_sum(j, t, r);
                if (value < minimal) {
                    l_s += 1;
                    if (not find(j, t, r))
                        l_filter_s += 1;
                }
                while (type_head < head_rig[r] && head_type[type_head] < j) type_head++;
                if (type_head < head_rig[r] && head_type[type_head] == j) {
                    if (value < minimal) {
                        l_s_constrain += 1;
                        if (not find(j, t, r))
                            l_filter_s_constrain += 1;
                    }
                }
            }
            if (j != t) {
                float value = calc_sum(h, j, r);
                if (value < minimal) {
                    r_s += 1;
                    if (not find(h, j, r))
                        r_filter_s += 1;
                }
                while (type_tail < tail_rig[r] && tail_type[type_tail] < j) type_tail++;
                if (type_tail < tail_rig[r] && tail_type[type_tail] == j) {
                    if (value < minimal) {
                        r_s_constrain += 1;
                        if (not find(h, j, r))
                            r_filter_s_constrain += 1;
                    }
                }
            }
        }
        if (l_filter_s < 10) l_filter_tot[0][id] += 1;
        if (l_s < 10) l_tot[0][id] += 1;
        if (r_filter_s < 10) r_filter_tot[0][id] += 1;
        if (r_s < 10) r_tot[0][id] += 1;

        l_filter_rank[0][id] += l_filter_s;
        r_filter_rank[0][id] += r_filter_s;
        l_rank[0][id] += l_s;
        r_rank[0][id] += r_s;

        if (l_filter_s < 10) l_filter_tot[label][id] += 1;
        if (l_s < 10) l_tot[label][id] += 1;
        if (r_filter_s < 10) r_filter_tot[label][id] += 1;
        if (r_s < 10) r_tot[label][id] += 1;

        l_filter_rank[label][id] += l_filter_s;
        r_filter_rank[label][id] += r_filter_s;
        l_rank[label][id] += l_s;
        r_rank[label][id] += r_s;



        if (l_filter_s_constrain < 10) l_filter_tot[5][id] += 1;
        if (l_s_constrain < 10) l_tot[5][id] += 1;
        if (r_filter_s_constrain < 10) r_filter_tot[5][id] += 1;
        if (r_s_constrain < 10) r_tot[5][id] += 1;

        l_filter_rank[5][id] += l_filter_s_constrain;
        r_filter_rank[5][id] += r_filter_s_constrain;
        l_rank[5][id] += l_s_constrain;
        r_rank[5][id] += r_s_constrain;

    }
    pthread_exit(NULL);
}

void* test(void *con) {
    for (int i = 0; i <= 5; i++) {
        l_filter_tot[i] = (float *)calloc(Threads, sizeof(float));
        r_filter_tot[i] = (float *)calloc(Threads, sizeof(float));
        l_tot[i] = (float *)calloc(Threads, sizeof(float));
        r_tot[i] = (float *)calloc(Threads, sizeof(float));

        l_filter_rank[i] = (float *)calloc(Threads, sizeof(float));
        r_filter_rank[i] = (float *)calloc(Threads, sizeof(float));
        l_rank[i] = (float *)calloc(Threads, sizeof(float));
        r_rank[i] = (float *)calloc(Threads, sizeof(float));
    }

    pthread_t *pt = (pthread_t *)malloc(Threads * sizeof(pthread_t));
    for (long a = 0; a < Threads; a++)
        pthread_create(&pt[a], NULL, testMode,  (void*)a);
    for (long a = 0; a < Threads; a++)
        pthread_join(pt[a], NULL);
    free(pt);
    for (int i = 0; i <= 5; i++)
    for (long a = 1; a < Threads; a++) {
        l_filter_tot[i][a] += l_filter_tot[i][a - 1];
        r_filter_tot[i][a] += r_filter_tot[i][a - 1];
        l_tot[i][a] += l_tot[i][a - 1];
        r_tot[i][a] += r_tot[i][a - 1];

        l_filter_rank[i][a] += l_filter_rank[i][a - 1];
        r_filter_rank[i][a] += r_filter_rank[i][a - 1];
        l_rank[i][a] += l_rank[i][a - 1];
        r_rank[i][a] += r_rank[i][a - 1];
    }
    for (int i = 0; i <= 0; i++) {
        printf("left %f %f\n", l_rank[i][Threads - 1] / testTotal, l_tot[i][Threads - 1] / testTotal);
        printf("left(filter) %f %f\n", l_filter_rank[i][Threads - 1] / testTotal, l_filter_tot[i][Threads - 1] / testTotal);
        printf("right %f %f\n", r_rank[i][Threads - 1] / testTotal, r_tot[i][Threads - 1] / testTotal);
        printf("right(filter) %f %f\n", r_filter_rank[i][Threads - 1] / testTotal, r_filter_tot[i][Threads - 1] / testTotal);
    }
    for (int i = 5; i <= 5; i++) {
        printf("left %f %f\n", l_rank[i][Threads - 1] / testTotal, l_tot[i][Threads - 1] / testTotal);
        printf("left(filter) %f %f\n", l_filter_rank[i][Threads - 1] / testTotal, l_filter_tot[i][Threads - 1] / testTotal);
        printf("right %f %f\n", r_rank[i][Threads - 1] / testTotal, r_tot[i][Threads - 1] / testTotal);
        printf("right(filter) %f %f\n", r_filter_rank[i][Threads - 1] / testTotal, r_filter_tot[i][Threads - 1] / testTotal);
    }
    for (int i = 1; i <= 4; i++) {
        printf("left %f %f\n", l_rank[i][Threads - 1] / nntotal[i], l_tot[i][Threads - 1] / nntotal[i]);
        printf("left(filter) %f %f\n", l_filter_rank[i][Threads - 1] / nntotal[i], l_filter_tot[i][Threads - 1] / nntotal[i]);
        printf("right %f %f\n", r_rank[i][Threads - 1] / nntotal[i], r_tot[i][Threads - 1] / nntotal[i]);
        printf("right(filter) %f %f\n", r_filter_rank[i][Threads - 1] / nntotal[i], r_filter_tot[i][Threads - 1] / nntotal[i]);
    }
}

long ArgPos(char *str, long argc, char **argv) {
    long a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

void setparameters(long argc, char **argv) {
    long i;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dimension = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sizeR", argc, argv)) > 0) dimensionR = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-input", argc, argv)) > 0) inPath = argv[i + 1];
    if ((i = ArgPos((char *)"-init", argc, argv)) > 0) initPath = argv[i + 1];
    if ((i = ArgPos((char *)"-thread", argc, argv)) > 0) Threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binaryFlag = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-note", argc, argv)) > 0) note = argv[i + 1];
} 

int main(int argc, char **argv) {
    setparameters(argc, argv);
    printf("%ld %ld\n", dimension, dimensionR);
    init();
    prepare();
    test(NULL);
    return 0;
}
