#include "ISideInfo.h"

#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/IO/GenericIO.h>

#include <SmurffCpp/Utils/linop.h>
#include <SmurffCpp/Utils/TensorUtils.h>

#include <SmurffCpp/Configs/SideInfoConfig.h>

#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/Error.h>

#include <SmurffCpp/Priors/ILatentPrior.h>

#include <ios>

using namespace smurff;
using namespace Eigen;

ISideInfo::ISideInfo(const SideInfoConfig &c, const ILatentPrior &prior)
    : m_config(c), m_prior(prior)
{
}

ISideInfo::~ISideInfo()
{
}

int ISideInfo::num_latent() const
{
   return m_prior.num_latent();
}

int ISideInfo::num_feat() const
{
   return m_config.getSideInfo()->getNCol();
}

int ISideInfo::num_item() const
{
   return m_config.getSideInfo()->getNRow();
}

uint64_t ISideInfo::nnz() const
{
   return m_config.getSideInfo()->getNNZ();
}

const Eigen::MatrixXd &ISideInfo::getUhat() const
{
   return Uhat;
}

const Eigen::MatrixXd &ISideInfo::getBeta() const
{
   return beta;
}

const Eigen::MatrixXd &ISideInfo::getBBt() const
{
   return BBt;
}


template <typename FeaturesType>
class SideInfoTempl : public ISideInfo
{
 public:
   SideInfoTempl(const SideInfoConfig &c, const ILatentPrior &prior)
       : ISideInfo(c, prior)
   {
   }

 private:
   FeaturesType F, Ft;

   void init() override
   {
      F = tensor_utils::config_to_eigen<FeaturesType>(*m_config.getSideInfo());
      Ft = F.transpose();
      //F_colsq = F.cwiseSquare().colwise().sum();

      // initial value (should be determined automatically)
      // Hyper-prior for beta_precision (mean 1.0):
      beta_precision = m_config.getSideInfo()->getNoiseConfig().getPrecision();
      beta_precision_one = Eigen::VectorXd::Constant(num_latent(), beta_precision);

      if (m_config.getDirect())
      {
         FtF_plus_beta = Ft * F;
         FtF_plus_beta.diagonal().array() += beta_precision;
      }

      Uhat = MatrixXd::Zero(num_latent(), num_item());
      beta = MatrixXd::Zero(num_latent(), num_feat());
      BBt = MatrixXd::Zero(num_latent(), num_latent());


   }

   void sample_beta() override
   {
      const MatrixXd &U = m_prior.U();
      // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + sqrt(lambda_beta) * Normal(0, Lambda^-1)
      // Ft_y is [ D x F ] matrix
      auto tmp = (U + MvNormal_prec(m_prior.getLambda(), num_item())).colwise() - m_prior.getMu();
      MatrixXd Ft_y = tmp * F + sqrt(beta_precision) * MvNormal_prec(m_prior.getLambda(), num_feat());

      if (m_config.getDirect())
         beta = FtF_plus_beta.llt().solve(Ft_y.transpose()).transpose();
      else
      {
         double tol = m_config.getTol();
         double max_iter = m_config.getMaxIter();
         double throw_on_chol = m_config.getThrowOnCholeskyError();
         //block cg operator
         auto FtFOp = [this](MatrixXd &X) -> MatrixXd
         {
            return (this->Ft * (this->F * X.transpose())).transpose() + this->beta_precision * X;
         };

         linop::solve_blockcg(beta, FtFOp, Ft_y, tol, max_iter, 32, 8, throw_on_chol);
      }
      
      //-- compute Uhat
      // uhat       - [D x N] dense matrix
      // sparseFeat - [N x F] sparse matrix (features)
      // beta       - [D x F] dense matrix
      // computes:
      //   uhat = beta * sparseFeat'
      Uhat = beta * Ft;

      BBt = beta * beta.transpose();

      if (m_config.getSampleBetaPrecision())
      {
         double old_beta = beta_precision;
         beta_precision = sample_beta_precision();
         if (m_config.getDirect())
         {
            FtF_plus_beta.diagonal().array() += beta_precision - old_beta;
         }
      }
   }

   // univariate version
   void sample_beta_one() override
   {
      // updating beta and beta_var
      const int blocksize = 4;
      const MatrixXd &U = m_prior.U();
      const VectorXd &mu = m_prior.getMu();
      const MatrixXd &Lambda = m_prior.getLambda();

      Eigen::MatrixXd Z;

#pragma omp parallel for private(Z) schedule(static, 1)
      for (int dstart = 0; dstart < num_latent(); dstart += blocksize)
      {
         const int dcount = std::min(blocksize, num_latent() - dstart);
         Z.resize(dcount, num_item());

         for (int i = 0; i < num_item(); i++)
         {
            for (int d = 0; d < dcount; d++)
            {
               int dx = d + dstart;
               Z(d, i) = U(dx, i) - mu(dx) - Uhat(dx, i);
            }
         }

         for (int f = 0; f < num_feat(); f++)
         {
            Eigen::VectorXd zx(dcount), delta_beta(dcount), randvals(dcount);
            // zx = Z[dstart : dstart + dcount, :] * F[:, f]
            zx = F.col(f).transpose() * Z.transpose();
            // TODO: check if sampling randvals for whole [nfeat x dcount] matrix works faster
            bmrandn_single_thread(randvals);

            for (int d = 0; d < dcount; d++)
            {
               int dx = d + dstart;
               double A_df = beta_precision_one(dx) + Lambda(dx, dx) * F_colsq(f);
               double B_df = Lambda(dx, dx) * (zx(d) + beta(dx, f) * F_colsq(f));
               double A_inv = 1.0 / A_df;
               double beta_new = B_df * A_inv + std::sqrt(A_inv) * randvals(d);
               delta_beta(d) = beta(dx, f) - beta_new;

               beta(dx, f) = beta_new;
            }
            // Z[dstart : dstart + dcount, :] += F[:, f] * delta_beta'
            Z += F.col(f) * delta_beta.transpose();
         }
      }

      Uhat = beta * Ft;

      if (m_config.getSampleBetaPrecision())
      {
         sample_beta_precision_one();
      }
   } // end sample_beta_one
};

typedef SideInfoTempl<Eigen::MatrixXd> DenseSideInfo;

std::shared_ptr<ISideInfo> ISideInfo::create_side_info(const SideInfoConfig &c, const ILatentPrior &p)
{
   return std::shared_ptr<ISideInfo>(new DenseSideInfo(c, p));
}

void ISideInfo::save(std::shared_ptr<const StepFile> sf) const
{
   std::string path = sf->getLinkMatrixFileName(m_prior.getMode());
   smurff::matrix_io::eigen::write_matrix(path, this->beta);
}

void ISideInfo::restore(std::shared_ptr<const StepFile> sf)
{
   std::string path = sf->getLinkMatrixFileName(m_prior.m_mode);
   THROWERROR_FILE_NOT_EXIST(path);
   smurff::matrix_io::eigen::read_matrix(path, this->beta);
}

std::ostream &ISideInfo::info(std::ostream &os, std::string indent)
{
   os << indent << "Method: ";
   if (m_config.getDirect())
   {
      os << "Cholesky Decomposition";
      double needs_gb = (double)num_feat() / 1024. * (double)num_feat() / 1024. / 1024.;
      if (needs_gb > 1.0)
         os << " (needing " << needs_gb << " GB of memory)";
      os << std::endl;
   }
   else
   {
      os << "CG Solver" << std::endl;
      os << indent << "  with tolerance: " << std::scientific << m_config.getTol() << std::fixed << std::endl;
      os << indent << "  with max iter: " << m_config.getMaxIter() << std::fixed << std::endl;
   }
   os << indent << " BetaPrecision: " << beta_precision << std::endl;
   return os;
}

std::ostream &ISideInfo::status(std::ostream &os, std::string indent) const
{
   os << indent << "FtF_plus_beta = " << FtF_plus_beta.norm() << std::endl;
   os << indent << "Beta          = " << beta.norm() << std::endl;
   os << indent << "beta_precision= " << beta_precision << std::endl;
   return os;
}

double ISideInfo::sample_beta_precision()
{
   const double nu = beta_precision_nu0;
   const double mu =  beta_precision_mu0;
   double nux = nu + beta.rows() * beta.cols();
   SHOW(getBBt());
   SHOW(m_prior.getLambda());
   double mux = mu * nux / (nu + mu * (getBBt().selfadjointView<Eigen::Lower>() * m_prior.getLambda()).trace());
   double b = nux / 2;
   double c = 2 * mux / nux;
   return rgamma(b, c);
}

void ISideInfo::sample_beta_precision_one()
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
      beta_precision_one(d) = rgamma(beta_precision_a, 1.0 / beta_precision_b(d));
   }
}