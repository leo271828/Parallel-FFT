#include <bits/stdc++.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define OMP_NUM 4
#define TRIALS 1

using namespace std;

typedef complex<double> dcomp;
typedef struct Data {  // Just store the required data
    int p; 
    int q; 
    int r; 
    int P;
    int Q;
    int R;
    int val;
    stack<int> arr;
} Data;

dcomp F[11][11];
const dcomp I = {0, 1};

void data_init(Data&, dcomp*, dcomp*, int);
void FFT(dcomp*, dcomp*, int);
void iter_FFT(dcomp*, dcomp*, int, Data);
void position(vector<int>&, dcomp*, int);

int main(){
    omp_set_num_threads(OMP_NUM);

    int L = 20, N = 1 << L;
    time_t T;
    vector<int> pos;
    Data datas;
    dcomp *z, *u;
    z = (dcomp*)malloc( N*sizeof(dcomp) ); 
    u = (dcomp*)malloc( N*sizeof(dcomp) );

    data_init(datas, z, u, N);
    printf("-------------------------------------\n");

    // Recurseive FFT
    printf("Start Recursive FFT with N : %d ...\n", N);
    T = clock();
    for(int i=0;i<TRIALS;i++)
        FFT(z, u, N);

	printf("Total time with clock  : %d\n", (clock() - T) / TRIALS);
	printf("Trivals time           : %d\n", TRIALS);
	printf("N = 2^%d               : %d\n\n", L, N);

    // Parallel FFT
    printf("Start Iterative PFFT with N : %d ...\n", N);
    T = clock();
    for(int i=0;i<TRIALS;i++)
        iter_FFT(z, u, N, datas);

	printf("Total time with clock  : %d\n", (clock() - T) / TRIALS);
	printf("Trivals time           : %d\n", TRIALS);
	printf("N = 2^%d               : %d\n\n", L, N);

    free(z); free(u); 
    return 0;
}

void data_init(Data &datas, dcomp *z, dcomp *u, int N){
    
    // z, u initialize
    for(int k=0;k<N;k++){
        u[k] = (dcomp)k;
        z[k] = 0;
    }
    
    // Compute F matrix
    F[2][0] = {1, 0}, F[2][1] = {-1, 0};
    for(int k=0;k<3;k++) F[3][k] = exp(-I*(dcomp)(2*k*M_PI/3));
    for(int k=0;k<5;k++) F[5][k] = exp(-I*(dcomp)(2*k*M_PI/5));

    // Compute prime factorization of N
    datas = { .p = 0, .q = 0, .r = 0, .P = 1, .Q = 1, .R = 1};
    while( N % 2 ==0 && N > 0){
        N/=2;
        datas.p++;
        datas.P*=2;
        datas.arr.push(2);
    }
    while( N % 3 ==0 && N > 0){
        N/=3;
        datas.q++;
        datas.Q*=3;
        datas.arr.push(3);
    }
    while( N % 5 ==0 && N > 0){
        N/=5;
        datas.r++;
        datas.R*=5;
        datas.arr.push(5);
    }
}

/* ------------------------- */
/* --- Recursive version --- */
/* ------------------------- */ 
void FFT(dcomp* y, dcomp *x, int N){
    // Part 1
    int base;
    if( N % 2 == 0 ) base = 2;
    else if ( N % 3 == 0 ) base = 3;
    else if ( N % 5 == 0 ) base = 5;
    else {
        base = N;
    }

    if( N == base ){
        for(int i=0;i<base;i++){
            y[i] = 0;
            for(int j=0;j<base;j++)
                y[i] += x[j] * F[base][(i*j)%base];
        }
        return ;
    }
    dcomp *Nx = (dcomp*)malloc( N * sizeof(dcomp) );
    dcomp *Ny = (dcomp*)malloc( N * sizeof(dcomp) );
    if( Nx == NULL || Ny == NULL ) printf("None");
    for(int k=0;k<N/base;k++){
        for(int j=0;j<base;j++)
            Nx[k + j*N/base] = x[k*base + j];
    }

    // Part 2
    for(int k=0;k<base;k++)
        FFT(Ny + k*N/base, Nx + k*N/base, N/base);

    // Part 3
    dcomp wn = 1;
    dcomp w = exp(-I*(dcomp)(2*M_PI/N));
    for(int k=0;k<N/base;k++){
        for(int j=0;j<base;j++)
            Ny[k + j*N/base] *= pow(wn, j);

        for(int j=0;j<base;j++){
            y[k+j*N/base] = 0;
            for(int i=0;i<base;i++)
                y[k+j*N/base] += F[base][i*j%base]*Ny[k+i*N/base];
        }
        wn *= w;
    }
    free(Ny); free(Nx);
}

/* ------------------------------------------------------ */
/* ------- Iterative version ---------------------------- */
/* --- If you want to see none parallel iterative FFT --- */
/* --- Plz comment line 170, 184 ------------------------ */
/* ------------------------------------------------------ */ 
void iter_FFT(dcomp *y, dcomp *x, int N, Data datas){
    int base = 1;
    dcomp wn = 1;
    dcomp w = exp(-I*(dcomp)(2*M_PI/N));
    dcomp *ans = (dcomp*)malloc( N*sizeof(dcomp) ); 

    // First run the smallest base
    base = datas.arr.top(); 
    datas.arr.pop();
    datas.val = base;

    #pragma omp parallel for schedule(static)
    for(int k=0;k<N;k+=base){
        for(int i=0;i<base;i++){
            y[i + k] = 0;
            for(int j=0;j<base;j++){
                y[i + k] += x[j + k] * F[base][(i*j)%base];
            }
        }
    }

    while( !datas.arr.empty() ){
        base = datas.arr.top(); datas.arr.pop();
        datas.val *= base;

        #pragma omp parallel for schedule(static)
        for(int idx=0; idx<N; idx+=datas.val){
            wn = 1;
            
            // Cannot be parallel
            for(int k=0; k<datas.val/base; k++){
                w = exp(-I*(dcomp)(2*M_PI/datas.val));
                for(int j=0;j<base;j++){
                    y[k + j*datas.val/base + idx] *= pow(wn, j);
                } 

                for(int j=0;j<base;j++){
                    ans[k+j*datas.val/base + idx] = 0;
                    for(int i=0;i<base;i++){
                        ans[k+j*datas.val/base + idx] += F[base][i*j%base]*y[k+i*datas.val/base + idx];
                    }
                }
                wn *= w;
            }
        }
        // y = ans;
    }
    // cout << "\nans:\n" ;
    // for(int i=0;i<3;i++) cout << ans[i] << endl; 
    // cout << "Done!" << endl;
    return ;
}


/* ------------------------------------------- */
/* --- Locate the position of all elements --- */
/* ------------------------------------------- */ 
void position(vector<int> &pos, dcomp *x, int N){ 
    int base;
    if( N % 2 == 0 ) base = 2;
    else if ( N % 3 == 0 ) base = 3;
    else if ( N % 5 == 0 ) base = 5;

    if( N == base ){
        for(int i=0;i<base;i++){
            pos.push_back(x[i].real());
        }
        return ;
    }

    dcomp *Nx = (dcomp*)malloc( N * sizeof(dcomp) );
    if( Nx == NULL ) printf("None");

    for(int k=0;k<N/base;k++)
        for(int j=0;j<base;j++)
            Nx[k + j*N/base] = x[k*base + j];

    for(int k=0;k<base;k++)
        position(pos, Nx + k*N/base, N/base);

    free(Nx);
}


// Execute command:
// g++ Parallel_FFT.cpp -fopenmp -O4