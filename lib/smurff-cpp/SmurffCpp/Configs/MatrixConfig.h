#pragma once

#include <vector>
#include <iostream>
#include <memory>

#include "TensorConfig.h"
#include "NoiseConfig.h"

namespace smurff
{
   class Data;

   class MatrixConfig : public TensorConfig
   {
   private:
      mutable std::shared_ptr<std::vector<std::uint32_t> > m_rows;
      mutable std::shared_ptr<std::vector<std::uint32_t> > m_cols;

   //
   // Dense double matrix constructos
   //
   public:
      MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
                   const std::vector<double>& values,
                   const NoiseConfig& noiseConfig);

      MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
                   std::vector<double>&& values,
                   const NoiseConfig& noiseConfig);

      MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
                   std::shared_ptr<std::vector<double> > values,
                   const NoiseConfig& noiseConfig);

   //
   // Sparse double matrix constructors
   //
   public:
      MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
                   const std::vector<std::uint32_t>& rows, const std::vector<std::uint32_t>& cols, const std::vector<double>& values,
                   const NoiseConfig& noiseConfig = NoiseConfig(), bool isScarce = false);

      MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
                   std::vector<std::uint32_t>&& rows, std::vector<std::uint32_t>&& cols, std::vector<double>&& values,
                   const NoiseConfig& noiseConfig = NoiseConfig(), bool isScarce = false);

      MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
                   std::shared_ptr<std::vector<std::uint32_t> > rows, std::shared_ptr<std::vector<std::uint32_t> > cols, std::shared_ptr<std::vector<double> > values,
                   const NoiseConfig& noiseConfig = NoiseConfig(), bool isScarce = false);

   //
   // Sparse binary matrix constructors
   //
   public:
      MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
                   const std::vector<std::uint32_t>& rows, const std::vector<std::uint32_t>& cols,
                   const NoiseConfig& noiseConfig = NoiseConfig(), bool isScarce = false);

      MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
                   std::vector<std::uint32_t>&& rows, std::vector<std::uint32_t>&& cols,
                   const NoiseConfig& noiseConfig = NoiseConfig(), bool isScarce = false);

      MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
                   std::shared_ptr<std::vector<std::uint32_t> > rows, std::shared_ptr<std::vector<std::uint32_t> > cols,
                   const NoiseConfig& noiseConfig = NoiseConfig(), bool isScarce = false);

   //
   // Constructors for constructing sparse matrix as a tensor
   //
   public:
      MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
                   const std::vector<std::uint32_t>& columns, const std::vector<double>& values,
                   const NoiseConfig& noiseConfig = NoiseConfig(), bool isScarce = false);

      MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
                   std::vector<std::uint32_t>&& columns, std::vector<double>&& values,
                   const NoiseConfig& noiseConfig = NoiseConfig(), bool isScarce = false);

      MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
                   std::shared_ptr<std::vector<std::uint32_t> > columns, std::shared_ptr<std::vector<double> > values,
                   const NoiseConfig& noiseConfig = NoiseConfig(), bool isScarce = false);

   //
   // Constructors for constructing sparse binary matrix as a tensor
   //
   public:
      MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
                   const std::vector<std::uint32_t>& columns,
                   const NoiseConfig& noiseConfig = NoiseConfig(), bool isScarce = false);

      MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
                   std::vector<std::uint32_t>&& columns,
                   const NoiseConfig& noiseConfig = NoiseConfig(), bool isScarce = false);

      MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
                   std::shared_ptr<std::vector<std::uint32_t> > columns,
                   const NoiseConfig& noiseConfig = NoiseConfig(), bool isScarce = false);

   public:
      MatrixConfig();

   public:
      std::uint64_t getNRow() const;
      std::uint64_t getNCol() const;

      const std::vector<std::uint32_t>& getRows() const;
      const std::vector<std::uint32_t>& getCols() const;

   public:
      std::shared_ptr<Data> create(std::shared_ptr<IDataCreator> creator) const override;

   public:
      void write(std::shared_ptr<IDataWriter> writer) const override;
   };
}
