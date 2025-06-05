#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>
#include <fstream>


using namespace std;
using namespace std::chrono;

// Funkcja pomocnicza do generowania macierzy układu Ax = b
void generate_matrix(vector<vector<double>> &A, vector<double> &b, int n) {
    srand(0);
    for (int i = 0; i < n; i++) {
        b[i] = rand() % 10 + 1;
        for (int j = 0; j < n; j++) {
            A[i][j] = rand() % 10 + 1;
        }
    }
}

// Referencyjna wersja eliminacji Gaussa (bez pivotingu)
void gaussian_elimination(vector<vector<double>> &A, vector<double> &b, int n) {
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
}

// Rozwiązywanie układu Ax = b po eliminacji - metodą podstawiania wstecznego
vector<double> back_substitution(const vector<vector<double>> &A, const vector<double> &b, int n) {
    vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }
    return x;
}

// Oblicz liczbę FLOP dla eliminacji Gaussa (około: 2/3 * n^3)
double compute_flops(int n) {
    return (2.0 / 3.0) * n * n * n;
}

int main() {
    ofstream timeFile("czasy1.csv");
    ofstream gflopsFile("gflopsy1.csv");

    timeFile << "n,czas\n";
    gflopsFile << "n,gflops\n";

    for (int n = 100; n <= 2000; n += 100) {
        vector<vector<double>> A(n, vector<double>(n));
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
                sum += A_copy[i][j] * x[j];
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