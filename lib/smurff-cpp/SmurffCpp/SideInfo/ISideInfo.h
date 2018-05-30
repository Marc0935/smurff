#pragma once

#include <iostream>

#include <Eigen/Dense>

#include <SmurffCpp/Configs/SideInfoConfig.h>
#include <SmurffCpp/Utils/StepFile.h>

namespace smurff
{

class MacauPrior;

class ISideInfo
{
 public:
   ISideInfo(const SideInfoConfig &, const MacauPrior &);
   virtual ~ISideInfo();

           int num_latent() const;
   virtual int num_feat() const = 0;
   virtual int num_item() const = 0;
   virtual uint64_t nnz() const = 0;
   virtual bool is_dense() const = 0;

   // access from prior
   const Eigen::MatrixXd &getUhat() const;
   const Eigen::MatrixXd &getBeta() const;
   const Eigen::MatrixXd &getBBt() const;

 protected:
   Eigen::MatrixXd Uhat;          // delta to m_prior.U()
   Eigen::MatrixXd FtF_plus_beta; // F'F + I*beta_precision
   Eigen::MatrixXd beta;          // link matrix
   Eigen::MatrixXd BBt;           // beta * beta' * precision

   // Hyper-prior for beta_precision (mean 1.0, var of 1e+3):
   const double beta_precision_mu0 = 1.0;
   const double beta_precision_nu0 = 1e-3;

   // sampled or fixed
   double beta_precision;

public:
   void init();

   virtual void sample_beta() = 0;
   virtual void compute_FtF() = 0;

protected:
   double sample_beta_precision();

protected:
   SideInfoConfig m_config;
   const MacauPrior &m_prior;

 public:
   void save(std::shared_ptr<const StepFile> sf) const;
   void restore(std::shared_ptr<const StepFile> sf);

   std::ostream &info(std::ostream &os, std::string indent);
   std::ostream &status(std::ostream &os, std::string indent) const;
   virtual std::ostream &print(std::ostream &os) const = 0;

};

} // namespace smurff
