#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Priors/NormalOnePrior.h>

#include <SmurffCpp/SideInfo/ISideInfo.h>

namespace smurff {


class MacauOnePrior : public NormalOnePrior
{
public:
   std::vector<std::shared_ptr<ISideInfo>> features;  // side information

public:
   MacauOnePrior(std::shared_ptr<BaseSession> session, uint32_t mode);

   void init() override;

   void update_prior() override;
    
   const Eigen::VectorXd getMu(int n) const override;

public:
   void addSideInfo(const SideInfoConfig &) override;

public:

   //used in update_prior
   void sample_mu_lambda();

public:

   void save(std::shared_ptr<const StepFile> sf) const override;
   void restore(std::shared_ptr<const StepFile> sf) override;
   std::ostream& status(std::ostream &os, std::string indent) const override;
};

}
