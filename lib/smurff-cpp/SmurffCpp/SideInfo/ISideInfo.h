#pragma once

#include <iostream>

#include <Eigen/Dense>

namespace smurff
{

class SideInfoConfig;
class MacauPrior;

class ISideInfo
{
 public:
   ISideInfo(const SideInfoConfig &, const MacauPrior &);
   virtual ~ISideInfo() {}

   int num_latent() const;
   virtual int cols() const = 0;
   virtual int rows() const = 0;
   virtual uint64_t nnz() const = 0;

   Eigen::MatrixXd::ConstColXpr getUhat(int n);
   const Eigen::MatrixXd &getUhat() const;
   const Eigen::MatrixXd &getBeta() const;
   const Eigen::MatrixXd &getBBt() const;

 private:
   Eigen::MatrixXd Uhat;          // delta to m_prior.U()
   Eigen::MatrixXd FtF_plus_beta; // F'F + I*beta_precision
   Eigen::MatrixXd beta;          // link matrix
   Eigen::MatrixXd BBt;           // beta * beta' * precision
   Eigen::MatrixXd HyperU, HyperU2;
   Eigen::MatrixXd Ft_y;

   // Hyper-prior for beta_precision (mean 1.0, var of 1e+3):
   const double beta_precision_mu0 = 1.0;
   const double beta_precision_nu0 = 1e-3;

   double beta_precision;

public:
   void sample_beta();
   void init();

 private:
   SideInfoConfig m_config;
   const MacauPrior &m_prior;

 public:
   void save(std::shared_ptr<const StepFile> sf) const;
   void restore(std::shared_ptr<const StepFile> sf);

 public:
   std::ostream &info(std::ostream &os, std::string indent);
   std::ostream &status(std::ostream &os, std::string indent) const;

 private:
   virtual void compute_Ft_y(const Eigen::MatrixXd &);
   virtual void compute_FtF();
   virtual void compute_uhat();

 public:
   static std::pair<double, double> posterior_beta_precision(Eigen::MatrixXd &beta, Eigen::MatrixXd &Lambda_u, double nu, double mu);

   static double sample_beta_precision(Eigen::MatrixXd &beta, Eigen::MatrixXd &Lambda_u, double nu, double mu);
   virtual std::ostream &print(std::ostream &os) const = 0;

   virtual bool is_dense() const = 0;
};

} // namespace smurff
