#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Priors/NormalPrior.h>

#include <SmurffCpp/SideInfo/ISideInfo.h>

namespace smurff {

/// Prior with side information
class MacauPrior : public NormalPrior
{
private:
   std::vector<std::shared_ptr<ISideInfo>> features;  // side information

private:
   MacauPrior();

public:
   MacauPrior(std::shared_ptr<BaseSession> session, uint32_t mode);

   virtual ~MacauPrior();

   void init() override;

   void update_prior() override;

   const Eigen::VectorXd getMu(int n) const override;

public:
   void addSideInfo(const SideInfoConfig &) override;

public:

   void save(std::shared_ptr<const StepFile> sf) const override;

   void restore(std::shared_ptr<const StepFile> sf) override;

public:

   std::ostream& info(std::ostream &os, std::string indent) override;

   std::ostream& status(std::ostream &os, std::string indent) const override;
};

}
