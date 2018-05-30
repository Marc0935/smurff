#pragma once

#include <memory>

#include <mpi.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Priors/MacauPrior.h>

namespace smurff 
{

//why not use init method ?

// Prior with side information
class MPIMacauPrior : public MacauPrior
{
public:
   int world_rank;
   int world_size;

private:
   int* rhs_for_rank = NULL;
   double* rec     = NULL;
   int* sendcounts = NULL;
   int* displs     = NULL;
   Eigen::MatrixXd Ft_y;

public:
   MPIMacauPrior(std::shared_ptr<BaseSession> session, int mode);

   virtual ~MPIMacauPrior();

   void init() override;

   std::ostream &info(std::ostream &os, std::string indent) override;
       

   bool run_slave() override;

   int rhs() const;

   void split_work_mpi(int num_latent, int num_nodes, int* work);
};

}
