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

void ISideInfo::init()
{
   if (m_config.getDirect())
   {
      compute_FtF(); // sparse or dense
      FtF_plus_beta.diagonal().array() += beta_precision;
   }

   Uhat = MatrixXd::Zero(num_latent(), rows());
   beta = MatrixXd::Zero(num_latent(), cols());
}

template <typename FType>
class SideInfo : public ISideInfo
{
 public:
   SideInfo::SideInfo(const SideInfoConfig &c, const MacauPrior &prior)
       : ISideInfo(c, prior)
   {
   }

 private:
   Ftype F, Ft;

   void sample_beta() override
   {
      const int num_feat = beta.cols();
      // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + sqrt(lambda_beta) * Normal(0, Lambda^-1)
      // Ft_y is [ D x F ] matrix
      MatrixXd tmp = (U() + MvNormal_pre(m_prior.getLambda(), U().cols())).colwise() - m_prior.getMu();
      MatrixXd Ft_y = tmp * F + sqrt(beta_precision) * MvNormal_prec(m_prior.getLambda(), num_feat());

      if (m_config.getDirect())
         beta = FtF_plus_beta.llt().solve(Ft_y.transpose()).transpose();
      else
         solve_blockcg(beta, beta_precision, Ft_y, m_config.tol, m_config.max_iter, 32, 8, m_config.throw_on_cholesky_error);

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
         beta_precision = sample_beta_precision(beta, this->Lambda, beta_precision_nu0, beta_precision_mu0);
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
      double needs_gb = (double)cols() / 1024. * (double)cols() / 1024. / 1024.;
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
   os << indent << "HyperU        = " << HyperU.norm() << std::endl;
   os << indent << "HyperU2       = " << HyperU2.norm() << std::endl;
   os << indent << "Beta          = " << beta.norm() << std::endl;
   os << indent << "beta_precision= " << beta_precision << std::endl;
   os << indent << "Ft_y          = " << Ft_y.norm() << std::endl;
   return os;
}

std::pair<double, double> ISideInfo::posterior_beta_precision(Eigen::MatrixXd &beta, Eigen::MatrixXd &Lambda_u, double nu, double mu)
{
   Eigen::MatrixXd BB = beta * beta.transpose();
   double nux = nu + beta.rows() * beta.cols();
   double mux = mu * nux / (nu + mu * (BB.selfadjointView<Eigen::Lower>() * Lambda_u).trace());
   double b = nux / 2;
   double c = 2 * mux / nux;
   return std::make_pair(b, c);
}

double ISideInfo::sample_beta_precision(Eigen::MatrixXd &beta, Eigen::MatrixXd &Lambda_u, double nu, double mu)
{
   auto gamma_post = posterior_beta_precision(beta, Lambda_u, nu, mu);
   return rgamma(gamma_post.first, gamma_post.second);
}