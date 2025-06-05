#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>
#include <fstream>
#include <immintrin.h>  // SIMD AVX2

using namespace std;
using namespace std::chrono;


inline double& get(vector<double>& A, int i, int j, int n) {
    return A[i * n + j];
}

inline const double& get(const vector<double>& A, int i, int j, int n) {
    return A[i * n + j];
}

void generate_matrix(vector<double>& A, vector<double>& b, int n) {
    srand(0);
    for (int i = 0; i < n; i++) {
        b[i] = rand() % 10 + 1;
        for (int j = 0; j < n; j++) {
            get(A, i, j, n) = rand() % 10 + 1;
        }
    }
}

// SIMD wersja eliminacji Gaussa (bez pivotingu)
void gaussian_elimination(vector<double>& A, vector<double>& b, int n) {
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            double factor = get(A, i, k, n) / get(A, k, k, n);
            int j = k;

            __m256d f = _mm256_set1_pd(factor);
            for (; j <= n - 4; j += 4) {
                __m256d aikj = _mm256_loadu_pd(&get(A, i, j, n));
                __m256d akkj = _mm256_loadu_pd(&get(A, k, j, n));
                __m256d prod = _mm256_mul_pd(f, akkj);
                __m256d res = _mm256_sub_pd(aikj, prod);
                _mm256_storeu_pd(&get(A, i, j, n), res);
            }

            // Zwykła pętla dla "ogonka""
            for (; j < n; j++) {
                get(A, i, j, n) -= factor * get(A, k, j, n);
            }

            b[i] -= factor * b[k];
        }
    }
}

vector<double> back_substitution(const vector<double>& A, const vector<double>& b, int n) {
    vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        __m256d sum = _mm256_setzero_pd();

        int j = i + 1;
        for (; j <= n - 4; j += 4) {
            __m256d a = _mm256_loadu_pd(&get(A, i, j, n));
            __m256d xj = _mm256_loadu_pd(&x[j]);
            __m256d prod = _mm256_mul_pd(a, xj);
            sum = _mm256_add_pd(sum, prod);
        }

        // Zsumuj 4 wartości z rejestru sum
        double temp[4];
        _mm256_storeu_pd(temp, sum);
        double s = temp[0] + temp[1] + temp[2] + temp[3];

        // Dodaj ogonek
        for (; j < n; j++) {
            s += get(A, i, j, n) * x[j];
        }

        x[i] = (b[i] - s) / get(A, i, i, n);
    }
    return x;
}


double compute_flops(int n) {
    return (2.0 / 3.0) * n * n * n;
}

int main() {
    ofstream timeFile("czasy3.csv");
    ofstream gflopsFile("gflopsy3.csv");

    timeFile << "n,czas\n";
    gflopsFile << "n,gflops\n";

    for (int n = 100; n <= 2000; n += 100) {
        vector<double> A(n * n);
        vector<double> b(n);

        generate_matrix(A, b, n);

        auto A_copy = A;
        auto b_copy = b;

        auto start = high_resolution_clock::now();
        gaussian_elimination(A, b, n);
        auto x = back_substitution(A, b, n);
        auto end = high_resolution_clock::now();

        duration<double> elapsed = end - start;
        double gflops = compute_flops(n) / elapsed.count() / 1e9;

        timeFile << n << "," << elapsed.count() << "\n";
        gflopsFile << n << "," << gflops << "\n";

        //sprawdzenie poprawności
        double error = 0.0;
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++)
                sum += get(A_copy, i, j, n) * x[j];
            error += pow(sum - b_copy[i], 2);
        }

        cout << "n = " << n << ", czas = " << elapsed.count()
             << " s, GFLOPS = " << gflops
             << ", residuum = " << error << endl;
    }

    timeFile.close();
    gflopsFile.close();

    return 0;
}