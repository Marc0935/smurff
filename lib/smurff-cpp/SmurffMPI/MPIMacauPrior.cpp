#include "MPIMacauPrior.h"

#include <SmurffCpp/Utils/Distribution.h>

using namespace smurff;

MPIMacauPrior::MPIMacauPrior(std::shared_ptr<BaseSession> session, int mode)
   : MacauPrior(session, mode)
{
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}

MPIMacauPrior::~MPIMacauPrior()
{
}

void MPIMacauPrior::init()
{
   MacauPrior::init();

   rhs_for_rank = new int[world_size];
   split_work_mpi(this->num_latent(), world_size, rhs_for_rank);

   sendcounts = new int[world_size];
   displs = new int[world_size];
   int sum = 0;
   for (int n = 0; n < world_size; n++) {
      sendcounts[n] = rhs_for_rank[n] * this->Features->cols();
      displs[n] = sum;
      sum += sendcounts[n];
   }
   rec = new double[sendcounts[world_rank]];
}

//TODO: missing implementation
std::ostream& MPIMacauPrior::info(std::ostream &os, std::string indent)
{
   if (world_rank == 0) {
      MacauPrior::info(os, indent);
      os << indent << " MPI version with " << world_size << " ranks\n";
   }
   return os;
}

void MPIMacauPrior::sample_beta()
{
   const int num_latent = this->beta.rows();
   const int num_feat = this->beta.cols();

   if (world_rank == 0) {
      this->compute_Ft_y_omp(this->Ft_y);
      this->Ft_y.transposeInPlace();
   }

   MPI_Bcast(&this->beta_precision, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(&this->tol, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(&this->max_iter, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   // sending Ft_y
   MPI_Scatterv(this->Ft_y.data(), sendcounts, displs, MPI_DOUBLE, rec, sendcounts[world_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
   int nrhs = rhs_for_rank[world_rank];
   Eigen::MatrixXd RHS(nrhs, num_feat), result(nrhs, num_feat);

#pragma omp parallel for schedule(static)
   for (int f = 0; f < num_feat; f++)
   {
      for (int d = 0; d < nrhs; d++)
      {
         RHS(d, f) = rec[f + d * num_feat];
      }
   }
   // solving
   this->Features->solve_blockcg(result, this->beta_precision, RHS, this->tol, this->max_iter, 32, 8);
   result.transposeInPlace();
   MPI_Gatherv(result.data(), nrhs*num_feat, MPI_DOUBLE, this->Ft_y.data(), sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   if (world_rank == 0)
   {
      //this->beta = Ft_y.transpose();
#pragma omp parallel for schedule(static)
      for (int f = 0; f < num_feat; f++)
      {
         for (int d = 0; d < num_latent; d++)
         {
            this->beta(d, f) = this->Ft_y(f, d);
         }
      }
   }
}

bool MPIMacauPrior::run_slave()
{
   sample_beta();
   return true;
}

int MPIMacauPrior::rhs() const
{
   return rhs_for_rank[world_rank];
}

void MPIMacauPrior::split_work_mpi(int num_latent, int num_nodes, int* work)
{
   double avg_work = num_latent / (double)num_nodes;
   int work_unit;
   if (2 <= avg_work) work_unit = 2;
   else work_unit = 1;

   int min_work = work_unit * (int)floor(avg_work / work_unit);
   int work_left = num_latent;

   for (int i = 0; i < num_nodes; i++) {
      work[i] = min_work;
      work_left -= min_work;
   }
   int i = 0;
   while (work_left > 0) {
      int take = std::min(work_left, work_unit);
      work[i] += take;
      work_left -= take;
      i = (i + 1) % num_nodes;
   }
}