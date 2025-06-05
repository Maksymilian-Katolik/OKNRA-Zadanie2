#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>
#include <fstream>


using namespace std;
using namespace std::chrono;

// Przed	                                        Po
// vector<vector<double>> A(n, vector<double>(n));	vector<double> A(n * n);
// Dostęp: A[i][j]	                                Dostęp: A[i * n + j]

// Pomocniczy dostęp do A[i][j] w formacie 1D
inline double& get(vector<double>& A, int i, int j, int n) {
    return A[i * n + j];
}

inline const double& get(const vector<double>& A, int i, int j, int n) {
    return A[i * n + j];
}

// Funkcja pomocnicza do generowania macierzy układu Ax = b
void generate_matrix(vector<double>& A, vector<double>& b, int n) {
    srand(0);
    for (int i = 0; i < n; i++) {
        b[i] = rand() % 10 + 1;
        for (int j = 0; j < n; j++) {
            get(A, i, j, n) = rand() % 10 + 1;
        }
    }
}

// Eliminacja Gaussa (bez pivotingu) dla spłaszczonej macierzy
void gaussian_elimination(vector<double>& A, vector<double>& b, int n) {
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            double factor = get(A, i, k, n) / get(A, k, k, n);
            for (int j = k; j < n; j++) {
                get(A, i, j, n) -= factor * get(A, k, j, n);
            }
            b[i] -= factor * b[k];
        }
    }
}

// Podstawianie wsteczne dla spłaszczonej macierzy
vector<double> back_substitution(const vector<double>& A, const vector<double>& b, int n) {
    vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= get(A, i, j, n) * x[j];
        }
        x[i] /= get(A, i, i, n);
    }
    return x;
}

double compute_flops(int n) {
    return (2.0 / 3.0) * n * n * n;
}

int main() {
    ofstream timeFile("czasy2.csv");
    ofstream gflopsFile("gflopsy2.csv");

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