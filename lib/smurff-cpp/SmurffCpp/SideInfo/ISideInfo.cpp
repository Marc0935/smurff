#include "ISideInfo.h"

#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/IO/GenericIO.h>

#include <SmurffCpp/Configs/SideInfoConfig.h>

#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/Error.h>

#include <SmurffCpp/Priors/MacauPrior.h>

#include <ios>

using namespace smurff;
using namespace Eigen;

ISideInfo::ISideInfo(const SideInfoConfig &c, const MacauPrior &prior)
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

void ISideInfo::init()
{
   if (m_config.getDirect())
   {
      compute_FtF(); // sparse or dense
      FtF_plus_beta.diagonal().array() += beta_precision;
   }

   Uhat = MatrixXd::Zero(num_latent(), num_item());
   beta = MatrixXd::Zero(num_latent(), num_feat());
}

template <typename FeaturesType>
class SideInfoTempl : public ISideInfo
{
 public:
   SideInfoTempl(const SideInfoConfig &c, const MacauPrior &prior)
       : ISideInfo(c, prior)
   {
   }

   int num_feat() const override { return F.cols(); }
   int num_item() const override { return F.rows(); }
   uint64_t nnz() const override { return F.nonZeros(); }

 private:
   FeaturesType F, Ft;

   void sample_beta() override
   {
      const MatrixXd &U = m_prior.U();
      // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + sqrt(lambda_beta) * Normal(0, Lambda^-1)
      // Ft_y is [ D x F ] matrix
      auto tmp = (U + MvNormal_pre(m_prior.getLambda(), num_item())).colwise() - m_prior.getMu();
      MatrixXd Ft_y = tmp * F + sqrt(beta_precision) * MvNormal_prec(m_prior.getLambda(), num_feat());

      if (m_config.getDirect())
         beta = FtF_plus_beta.llt().solve(Ft_y.transpose()).transpose();
      else
      {
         double tol = m_config.getTol();
         double max_iter = m_config.getMaxIter();
         double throw_on_chol = m_config.getThrowOnCholeskyError();
         solve_blockcg(beta, beta_precision, Ft_y, tol, max_iter, 32, 8, throw_on_chol);
      }

      //-- compute Uhat
      // uhat       - [D x N] dense matrix
      // sparseFeat - [N x F] sparse matrix (features)
      // beta       - [D x F] dense matrix
      // computes:
      //   uhat = beta * sparseFeat'
      Uhat = beta * Ft;

      if (m_config.getSampleBetaPrecision())
      {
         double old_beta = beta_precision;
         beta_precision = sample_beta_precision();
         FtF_plus_beta.diagonal().array() += beta_precision - old_beta;
      }
   }
};

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
   os << indent << "SideInfo: ";
   print(os);
   os << indent << " Method: ";
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
   Eigen::MatrixXd BB = beta * beta.transpose();
   double nux = nu + beta.rows() * beta.cols();
   double mux = mu * nux / (nu + mu * (BB.selfadjointView<Eigen::Lower>() * m_prior.getLambda()).trace());
   double b = nux / 2;
   double c = 2 * mux / nux;
   return rgamma(b, c);
}