#pragma once

#include <iostream>

#include <Eigen/Dense>

#include <SmurffCpp/Configs/SideInfoConfig.h>
#include <SmurffCpp/Utils/StepFile.h>

namespace smurff
{

class ILatentPrior;

class ISideInfo
{
 public:
   ISideInfo(const SideInfoConfig &, const ILatentPrior &);
   virtual ~ISideInfo();

   static std::shared_ptr<ISideInfo> create_side_info(const SideInfoConfig &, const ILatentPrior &);

   int num_latent() const;
   int num_feat() const;
   int num_item() const;
   uint64_t nnz() const;

   // access from prior
   const Eigen::MatrixXd &getUhat() const;
   const Eigen::MatrixXd &getBeta() const;
   const Eigen::MatrixXd &getBBt() const;

 protected:
   Eigen::MatrixXd Uhat;          // delta to m_prior.U()
   Eigen::MatrixXd FtF_plus_beta; // F'F + I*beta_precision
   Eigen::MatrixXd F_colsq;       // columnwise sum of squared values
   Eigen::MatrixXd beta;          // link matrix
   Eigen::MatrixXd BBt;           // beta * beta' * precision

   // Hyper-prior for beta_precision (mean 1.0, var of 1e+3):
   const double beta_precision_mu0 = 1.0;
   const double beta_precision_nu0 = 1e-3;

   const double beta_precision_a0 = 0.1;
   const double beta_precision_b0 = 0.1;

   // sampled or fixed
   double beta_precision;
   Eigen::VectorXd beta_precision_one;

public:
   virtual void init() = 0;
   virtual void sample_beta() = 0;
   virtual void sample_beta_one() = 0;

protected:
   double sample_beta_precision();
   void sample_beta_precision_one();

protected:
   SideInfoConfig m_config;
   const ILatentPrior &m_prior;

 public:
   void save(std::shared_ptr<const StepFile> sf) const;
   void restore(std::shared_ptr<const StepFile> sf);

   std::ostream &info(std::ostream &os, std::string indent);
   std::ostream &status(std::ostream &os, std::string indent) const;
};

} // namespace smurff
