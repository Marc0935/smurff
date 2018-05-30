#include "MacauOnePrior.h"

#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/IO/GenericIO.h>

#include <SmurffCpp/Utils/linop.h>

using namespace smurff;

MacauOnePrior::MacauOnePrior(std::shared_ptr<BaseSession> session, uint32_t mode)
   : NormalOnePrior(session, mode, "MacauOnePrior")
{
   bp0 = SideInfoConfig::BETA_PRECISION_DEFAULT_VALUE;

   enable_beta_precision_sampling = Config::ENABLE_BETA_PRECISION_SAMPLING_DEFAULT_VALUE;
}

void MacauOnePrior::init()
{
   NormalOnePrior::init();

   // init SideInfo related
   Uhat = Eigen::MatrixXd::Constant(num_latent(), Features->num_item(), 0.0);
   beta = Eigen::MatrixXd::Constant(num_latent(), Features->num_feat(), 0.0);

   // initial value (should be determined automatically)
   // Hyper-prior for beta_precision (mean 1.0):
   beta_precision = Eigen::VectorXd::Constant(num_latent(), bp0);
   beta_precision_a0 = 0.1;
   beta_precision_b0 = 0.1;
}

void MacauOnePrior::update_prior()
{
   sample_mu_lambda(U());
   sample_beta(U());
   //Features->compute_uhat(Uhat, beta);

   if (enable_beta_precision_sampling)
      sample_beta_precision();
}

const Eigen::VectorXd MacauOnePrior::getMu(int n) const
{
   return this->mu + Uhat.col(n);
}

void MacauOnePrior::addSideInfo(const SideInfoConfig &)
{
   //F_colsq = Features->col_square_sum();
}

void MacauOnePrior::sample_beta(const Eigen::MatrixXd &U)
{
   // updating beta and beta_var
   const int nfeat = beta.cols();
   const int N = U.cols();
   const int blocksize = 4;

   Eigen::MatrixXd Z;

   #pragma omp parallel for private(Z) schedule(static, 1)
   for (int dstart = 0; dstart < num_latent(); dstart += blocksize)
   {
      const int dcount = std::min(blocksize, num_latent() - dstart);
      Z.resize(dcount, U.cols());

      for (int i = 0; i < N; i++)
      {
         for (int d = 0; d < dcount; d++)
         {
            int dx = d + dstart;
            Z(d, i) = U(dx, i) - mu(dx) - Uhat(dx, i);
         }
      }

      for (int f = 0; f < nfeat; f++)
      {
         Eigen::VectorXd zx(dcount), delta_beta(dcount), randvals(dcount);
         // zx = Z[dstart : dstart + dcount, :] * F[:, f]
         ///Features->At_mul_Bt(zx, f, Z);
         // TODO: check if sampling randvals for whole [nfeat x dcount] matrix works faster
         bmrandn_single_thread(randvals);

         for (int d = 0; d < dcount; d++)
         {
            int dx = d + dstart;
            double A_df = beta_precision(dx) + Lambda(dx, dx) * F_colsq(f);
            double B_df = Lambda(dx, dx) * (zx(d) + beta(dx, f) * F_colsq(f));
            double A_inv = 1.0 / A_df;
            double beta_new = B_df * A_inv + std::sqrt(A_inv) * randvals(d);
            delta_beta(d) = beta(dx, f) - beta_new;

            beta(dx, f) = beta_new;
         }
         // Z[dstart : dstart + dcount, :] += F[:, f] * delta_beta'
         //Features->add_Acol_mul_bt(Z, f, delta_beta);
      }
   }
}

void MacauOnePrior::sample_mu_lambda(const Eigen::MatrixXd &U)
{
   Eigen::MatrixXd WI(num_latent(), num_latent());
   WI.setIdentity();
   int N = U.cols();

   Eigen::MatrixXd Udelta(num_latent(), N);
   #pragma omp parallel for schedule(static)
   for (int i = 0; i < N; i++)
   {
      for (int d = 0; d < num_latent(); d++)
      {
         Udelta(d, i) = U(d, i) - Uhat(d, i);
      }
   }
   std::tie(mu, Lambda) = CondNormalWishart(Udelta, Eigen::VectorXd::Constant(num_latent(), 0.0), 2.0, WI, num_latent());
}

void MacauOnePrior::sample_beta_precision()
{
   double beta_precision_a = beta_precision_a0 + beta.cols() / 2.0;
   Eigen::VectorXd beta_precision_b = Eigen::VectorXd::Constant(beta.rows(), beta_precision_b0);
   const int D = beta.rows();
   const int F = beta.cols();
   #pragma omp parallel
   {
      Eigen::VectorXd tmp(D);
      tmp.setZero();
      #pragma omp for schedule(static)
      for (int f = 0; f < F; f++)
      {
         for (int d = 0; d < D; d++)
         {
            tmp(d) += std::pow(beta(d, f), 2);
         }
      }
      #pragma omp critical
      {
         beta_precision_b += tmp / 2;
      }
   }
   for (int d = 0; d < D; d++)
   {
      beta_precision(d) = rgamma(beta_precision_a, 1.0 / beta_precision_b(d));
   }
}

void MacauOnePrior::save(std::shared_ptr<const StepFile> sf) const
{
   NormalOnePrior::save(sf);

   std::string path = sf->getLinkMatrixFileName(m_mode);
   smurff::matrix_io::eigen::write_matrix(path, beta);
}

void MacauOnePrior::restore(std::shared_ptr<const StepFile> sf)
{
   NormalOnePrior::restore(sf);

   std::string path = sf->getLinkMatrixFileName(m_mode);

   THROWERROR_FILE_NOT_EXIST(path);

   smurff::matrix_io::eigen::read_matrix(path, beta);
}

std::ostream& MacauOnePrior::status(std::ostream &os, std::string indent) const
{
   os << indent << "  " << m_name << ": Beta = " << beta.norm() << std::endl;
   return os;
}
