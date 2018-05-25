#include "MacauPrior.h"

#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/IO/GenericIO.h>

#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/Error.h>

#include <SmurffCpp/Utils/linop.h>

#include <ios>

using namespace smurff;

MacauPrior::MacauPrior()
   : NormalPrior() 
{
}

MacauPrior::MacauPrior(std::shared_ptr<BaseSession> session, uint32_t mode)
   : NormalPrior(session, mode, "MacauPrior")
{
   beta_precision = SideInfoConfig::BETA_PRECISION_DEFAULT_VALUE;
   tol = SideInfoConfig::TOL_DEFAULT_VALUE;
   max_iter = SideInfoConfig::MAX_ITER_DEFAULT_VALUE;


   enable_beta_precision_sampling = Config::ENABLE_BETA_PRECISION_SAMPLING_DEFAULT_VALUE;
}

MacauPrior::~MacauPrior()
{
}

void MacauPrior::init()
{
   NormalPrior::init();

   THROWERROR_ASSERT_MSG(Features->rows() == num_cols(), "Number of rows in train must be equal to number of rows in features");

   if (use_FtF)
   {
      std::uint64_t dim = Features->cols();
      FtF_plus_beta.resize(dim, dim);
      Features->At_mul_A(FtF_plus_beta);
      FtF_plus_beta.diagonal().array() += beta_precision;
   }

   Uhat.resize(this->num_latent(), Features->rows());
   Uhat.setZero();

   beta.resize(this->num_latent(), Features->cols());
   beta.setZero();
}

void MacauPrior::update_prior()
{
   // residual (Uhat is later overwritten):
   Uhat.noalias() = U() - Uhat;
   Eigen::MatrixXd BBt = smurff::linop::A_mul_At_combo(beta);

   // sampling Gaussian
   std::tie(this->mu, this->Lambda) = CondNormalWishart(Uhat, this->mu0, this->b0, this->WI + beta_precision * BBt, this->df + beta.cols());
   sample_beta();
   Features->compute_uhat(Uhat, beta);

   if (enable_beta_precision_sampling)
   {
      double old_beta = beta_precision;
      beta_precision = sample_beta_precision(beta, this->Lambda, beta_precision_nu0, beta_precision_mu0);
      FtF_plus_beta.diagonal().array() += beta_precision - old_beta;
   }
}

const Eigen::VectorXd MacauPrior::getMu(int n) const
{
   return this->mu + Uhat.col(n);
}

void MacauPrior::compute_Ft_y_omp(Eigen::MatrixXd& Ft_y)
{
   const int num_feat = beta.cols();

   // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + std::sqrt(beta_precision) * Normal(0, Lambda^-1)
   // Ft_y is [ D x F ] matrix
   HyperU = (U() + MvNormal_prec(Lambda, num_cols())).colwise() - mu;

   Ft_y = Features->A_mul_B(HyperU);
   HyperU2 = MvNormal_prec(Lambda, num_feat);

   #pragma omp parallel for schedule(static)
   for (int f = 0; f < num_feat; f++)
   {
      for (int d = 0; d < num_latent(); d++)
      {
         Ft_y(d, f) += std::sqrt(beta_precision) * HyperU2(d, f);
      }
   }
}

void MacauPrior::sample_beta()
{
   if (use_FtF)
      sample_beta_direct();
   else
      sample_beta_cg();
}

void MacauPrior::addSideInfo(const std::shared_ptr<ISideInfo>& side_info_a, double beta_precision_a, double tolerance_a, int max_iter_a, bool direct_a, bool enable_beta_precision_sampling_a, bool throw_on_cholesky_error_a)
{
   //FIXME: remove old code

   // old code

   // side information
   Features = side_info_a;
   beta_precision = beta_precision_a;
   tol = tolerance_a;
   max_iter = max_iter_a;
   use_FtF = direct_a;
   enable_beta_precision_sampling = enable_beta_precision_sampling_a;
   throw_on_cholesky_error = throw_on_cholesky_error_a;

   // new code

   // side information
   side_info_values.push_back(side_info_a);
   beta_precision_values.push_back(beta_precision_a);
   tol_values.push_back(tolerance_a);
   max_iter_values.push_back(max_iter_a);
   direct_values.push_back(direct_a);
   enable_beta_precision_sampling_values.push_back(enable_beta_precision_sampling_a);
   throw_on_cholesky_error_values.push_back(throw_on_cholesky_error_a);

   // other code

   // Hyper-prior for beta_precision (mean 1.0, var of 1e+3):
   beta_precision_mu0 = 1.0;
   beta_precision_nu0 = 1e-3;
}

void MacauPrior::save(std::shared_ptr<const StepFile> sf) const
{
   NormalPrior::save(sf);

   std::string path = sf->getLinkMatrixFileName(m_mode);
   smurff::matrix_io::eigen::write_matrix(path, this->beta);
}

void MacauPrior::restore(std::shared_ptr<const StepFile> sf)
{
   NormalPrior::restore(sf);

   std::string path = sf->getLinkMatrixFileName(m_mode);

   THROWERROR_FILE_NOT_EXIST(path);

   smurff::matrix_io::eigen::read_matrix(path, this->beta);
}

std::ostream& MacauPrior::info(std::ostream &os, std::string indent)
{
   NormalPrior::info(os, indent);
   os << indent << " SideInfo: ";
   Features->print(os);
   os << indent << " Method: ";
   if (use_FtF)
   {
      os << "Cholesky Decomposition";
      double needs_gb = (double)Features->cols() / 1024. * (double)Features->cols() / 1024. / 1024.;
      if (needs_gb > 1.0) os << " (needing " << needs_gb << " GB of memory)";
      os << std::endl;
   } else {
      os << "CG Solver" << std::endl;
      os << indent << "  with tolerance: " << std::scientific << tol << std::fixed << std::endl;
      os << indent << "  with max iter: " << max_iter << std::fixed << std::endl;
   }
   os << indent << " BetaPrecision: " << beta_precision << std::endl;
   return os;
}

std::ostream& MacauPrior::status(std::ostream &os, std::string indent) const
{
   os << indent << m_name << ": " << std::endl;
   indent += "  ";
   os << indent << "FtF_plus_beta          = " << FtF_plus_beta.norm() << std::endl;
   os << indent << "HyperU       = " << HyperU.norm() << std::endl;
   os << indent << "HyperU2      = " << HyperU2.norm() << std::endl;
   os << indent << "Beta         = " << beta.norm() << std::endl;
   os << indent << "beta_precision  = " << beta_precision << std::endl;
   os << indent << "Ft_y         = " << Ft_y.norm() << std::endl;
   return os;
}

// direct method
void MacauPrior::sample_beta_direct()
{
    this->compute_Ft_y_omp(Ft_y);
    beta = FtF_plus_beta.llt().solve(Ft_y.transpose()).transpose();
}

std::pair<double, double> MacauPrior::posterior_beta_precision(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu)
{
   const int D = beta.rows();
   Eigen::MatrixXd BB(D, D);
   smurff::linop::A_mul_At_combo(BB, beta);
   double nux = nu + beta.rows() * beta.cols();
   double mux = mu * nux / (nu + mu * (BB.selfadjointView<Eigen::Lower>() * Lambda_u).trace());
   double b = nux / 2;
   double c = 2 * mux / nux;
   return std::make_pair(b, c);
}

double MacauPrior::sample_beta_precision(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu)
{
   auto gamma_post = posterior_beta_precision(beta, Lambda_u, nu, mu);
   return rgamma(gamma_post.first, gamma_post.second);
}

void MacauPrior::sample_beta_cg()
{
    Eigen::MatrixXd Ft_y;
    this->compute_Ft_y_omp(Ft_y);

    Features->solve_blockcg(beta, beta_precision, Ft_y, tol, max_iter, 32, 8, throw_on_cholesky_error);
}
