#include "ISideInfo.h"

#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/IO/GenericIO.h>

#include <SmurffCpp/Utils/linop.h>
#include <SmurffCpp/Utils/TensorUtils.h>

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
   SideInfoTempl(const SideInfoConfig &c, const MacauPrior &prior)
       : ISideInfo(c, prior)
   {
   }

 private:
   FeaturesType F, Ft;

   void init() override
   {
      F = tensor_utils::config_to_eigen<FeaturesType>(*m_config.getSideInfo());
      Ft = F.transpose();

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
      auto tmp = (U + MvNormal_prec(m_prior.getLambda(), num_item())).colwise() - m_prior.getMuuuu();
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
};

typedef SideInfoTempl<Eigen::MatrixXd> DenseSideInfo;

std::shared_ptr<ISideInfo> ISideInfo::create_side_info(const SideInfoConfig &c, const MacauPrior &p)
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
   double mux = mu * nux / (nu + mu * (getBBt().selfadjointView<Eigen::Lower>() * m_prior.getLambda()).trace());
   double b = nux / 2;
   double c = 2 * mux / nux;
   return rgamma(b, c);
}