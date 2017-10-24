
//GFA (Group Factor Analysis)
//Translation into C++ of the Python implementation of the file ./R/CCAGFA.R in the R package CCAGFA
//

#include <stdlib.h>     /* srand, rand */

#include <limits>
#include <iomanip>
#include <string>
#include <algorithm>
#include <random>
#include <sys/time.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <cmath>
#include <cerrno>
#include <cstring>
#include <cfenv>
#include <bitset>
#include <list>
#include <cfloat>
#include <chrono>

#include "counters.h"

#define EIGEN_DONT_PARALLELIZE 1
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>

const unsigned K = 32;

typedef Eigen::SparseMatrix<double> SparseMatrixD;
typedef Eigen::Matrix<double, K, K> MatrixNNd;
typedef Eigen::Matrix<double, K, Eigen::Dynamic> MatrixNXd;
typedef Eigen::Matrix<double, K, 1> VectorNd;
typedef Eigen::Array<double, K, K> ArrayNNd;
typedef Eigen::Array<double, K, Eigen::Dynamic> ArrayNXd;
typedef Eigen::Array<double, K, 1> ArrayNd;


#ifdef _OPENMP
#pragma omp declare reduction (VectorPlus : VectorNd : omp_out += omp_in) initializer(omp_priv = VectorNd::Zero())
#pragma omp declare reduction (MatrixPlus : MatrixNNd : omp_out += omp_in) initializer(omp_priv = MatrixNNd::Zero())
#endif

using namespace std;
using namespace Eigen;

const double pi = 3.1415926;

double randn(double = .0) {
    static thread_local std::mt19937 rng;
    static thread_local normal_distribution<> nd;
    return nd(rng); 
}
auto nrandn(int n) -> decltype( ArrayXd::NullaryExpr(n, ptr_fun(randn)) ) { return ArrayXd::NullaryExpr(n, ptr_fun(randn)); }
auto nrandn(int n, int m) -> decltype( ArrayXXd::NullaryExpr(n, m, ptr_fun(randn)) ) { return ArrayXXd::NullaryExpr(n, m, ptr_fun(randn)); }

double init_tau = 1000.;
double prior_alpha_0t = 1.;
double prior_beta_0t = 1.;
double prior_alpha_0 = 1.;
double prior_beta_0 = 1.;
double prior_alpha_0X = 1.;
double prior_beta_0X = 1.;
double prior_beta = 1;
double prior_beta_X = 1;

MatrixXd random_Y(int N, int D, int K)
{
    MatrixXd X_tmp = nrandn(N,3);
    MatrixXd W_tmp = nrandn(D,3);
    return X_tmp * W_tmp.transpose() + nrandn(N,D).matrix();
}


MatrixXd ones_Y(int N, int D, int K)
{
    MatrixXd X_tmp = MatrixXd::Ones(N,K);
    MatrixXd W_tmp = MatrixXd::Ones(D,K);
    return X_tmp * W_tmp.transpose();
}

// scale and center rows
template<typename Matrix>
void scale_rows(Matrix &X) {
    Matrix T = X.colwise() - X.rowwise().mean();
    VectorXd standardDeviation = T.array().square().rowwise().sum().sqrt() / T.rows();
    for (int i = 0 ; i < X.rows(); i++) X.row(i) = T.row(i) / standardDeviation[i];
}

int main(int argc, char *argv[])
{

    assert(argc>3 && "Usage GFA N D iter");

    int N = atoi(argv[1]);
    int D = atoi(argv[2]);
    int iter_max = atoi(argv[3]);

    assert(D>0 && N>0 && iter_max > 0 && "Usage GFA N D iter_max");

    srand (12345);
    //MatrixXd Y = random_Y(N,D,3);
    MatrixXd Y = ones_Y(N,D,2);
    MatrixXd Yt = Y.transpose();

    MatrixNXd W(K,D); W.setZero(); //# the loading/weights matrix
    MatrixNXd X = nrandn(K,N);
    scale_rows(X);
    MatrixNNd XX = X * X.transpose();

    ArrayNd alpha = ArrayNd::Ones(K);
    VectorNd r = VectorNd::Zero(K).array() + 0.5;
    VectorXd cost(VectorXd::Constant(iter_max, -N * D * log(2 * pi) / 2));
    VectorXd tau(VectorXd::Constant(D, init_tau)); //# The mean noise precisions

    const double alpha_0t = 1.;
    const double a_tau = alpha_0t + N / 2;


    // init W
    MatrixNNd covW = (MatrixNNd::Identity(K, K) + init_tau * XX).inverse();
    MatrixNNd Sx = covW.llt().matrixU();
    W = covW * (X * Y) * init_tau + Sx * nrandn(K,D).matrix();
    MatrixNXd Wtau = W * init_tau;
    MatrixNNd WtauW = Wtau *  W.transpose();
    VectorNd Zcol = VectorNd::Constant(D);
    VectorNd Zkeep = Zcol;
    VectorNd W2col = W.array().square().rowwise().sum();;

    auto start = tick();
    auto prev = start;

    double err = 0.0;
    for (int iter = 1 ; iter < iter_max; iter++)
    {
        GFA_COUNTER("main");

        //## Sample the projections W.
        {
            std::default_random_engine generator;
            std::uniform_real_distribution<double> udist(0,1);
            ArrayNd log_alpha = alpha.log();
            ArrayNd log_r = - r.array().log() + (VectorNd::Ones(K) - r).array().log();

            double log_t = .0, b_tau_sq = .0;
            Zcol.setZero();
            W2col.setZero();
            WtauW.setZero();

            #pragma omp parallel for \
            reduction(+:log_t,b_tau_sq,err) \
            reduction(VectorPlus:Zcol,W2col) \
            reduction(MatrixPlus:WtauW) \
            schedule(dynamic, 8)
            for(int d = 0; d<D; d++) {
                VectorNd Wcol = W.col(d);
                VectorNd yX(VectorNd::Zero());
                double b_tau = .0;
                for (int n=0; n<N; ++n) {
                    double y = Y(n,d);
                    auto Xcol = X.col(n);
                    double e = y - (Wcol.transpose() * Xcol);
                    yX.noalias() += y * Xcol; 
                    b_tau += e * e;
                }

                std::gamma_distribution<double> gdist(a_tau, 1/(prior_beta_0t + b_tau/2) );
                double t = tau[d] = gdist(generator);

                for(unsigned k=0;k<K;++k) {
                    double lambda = t * XX(k,k) + alpha(k);
                    double mu = t / lambda * (yX(k) - Wcol.transpose() * XX.col(k) + Wcol(k) * XX(k,k));
                    double z1 = log_r(k) -  0.5 * (lambda * mu * mu - log(lambda) + log_alpha(k));
                    double z = 1 / (1 + exp(z1));
                    double r = udist(generator);
                    if (Zkeep(k) > 0 && r < z) {
                        Zcol(k)++;
                        Wcol(k) = mu + randn() / sqrt(lambda);
                    } else {
                        Wcol(k) = .0;
                    }
                }

                log_t += log(t);
                b_tau_sq += b_tau/2 * t;
                err += b_tau/2;
                W.col(d) = Wcol;
                W2col += Wcol.array().square().matrix();
                Wtau.col(d) = Wcol * t;
                WtauW += Wcol.matrix() * (Wcol * t).matrix().transpose();
            }

            cost[iter] = cost[iter] + N * log_t / 2 - b_tau_sq;
            r = ( Zcol.array() + prior_beta ) / ( D + prior_beta * D ) ;
            Zkeep = Zcol.array();

            //-- updata alpha K samples from Gamma
            auto ww = W2col.array() / 2 + prior_beta_0;
            auto tmpz = Zcol.array() / 2 + prior_alpha_0 ;
            alpha = tmpz.binaryExpr(ww, [](double a, double b)->double {
                    std::default_random_engine generator;
                    std::gamma_distribution<double> distribution(a, 1/b);
                    return distribution(generator) + 1e-7;
            });
        }

        //## Update the latent variables X
        {
            MatrixNNd covZ = (MatrixNNd::Identity() + WtauW).inverse();
            auto Sx = covZ.llt().matrixL();

            XX.setZero();
            #pragma omp parallel for \
            reduction(MatrixPlus:XX) \
            schedule(dynamic, 8)
            for(int n = 0; n<N; n++) {
                VectorNd x = covZ * (Wtau * Yt.col(n)) + Sx * nrandn(K).matrix();
                X.col(n) = x;
                XX += x * x.transpose();
            }
        }

        auto end = tick();
        const bool verbose = false;
        if (verbose || end - prev > 1.0 || iter == iter_max - 1)  {
            int eta = (int)((end-start)/iter*(iter_max-iter));
            int keep = (Zkeep.array() > 0).count();
            printf("iter: %d/%d\tN/D/K/k: %d/%d/%d/%d\terr: %.4f\tMsamples/sec: %.0f\tETA: %02d:%02d:%02d\n",
                    iter, iter_max, N, D, K, keep, (err / N / D / iter),
                    (1e-6) * N * D * K * iter / (end-start),
                    eta/(60*60), (eta%(60*60))/60, eta%60);
            prev = end;
        }
    } //## The main loop of the algorithm ends.

#ifdef GFA_PROFILING
    perf_data.print();
#endif

    return 0;
}
