#pragma once

#include <array>
#include <complex>
#include <deal.II/base/point.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/sparse_matrix.h>

using EFieldComponent = std::complex<double>;
using EFieldValue = std::array<EFieldComponent, 3>;
using DofCount = unsigned int;
using DofNumber = unsigned int;
using Position = dealii::Point<3,double>;
using NumericVectorLocal = dealii::Vector<EFieldComponent>;
using NumericVectorDistributed =  dealii::parallel::distributed::Vector<EFieldComponent>;
using SweepingLevel = unsigned int;
using HSIEElementOrder = unsigned int;
using NedelecElementOrder = unsigned int;
using BoundaryId = unsigned int;
using DofSortingData = std::pair<DofNumber, Position>;
using ComplexNumber = std::complex<double>;
using SparseComplexMatrix = dealii::SparseMatrix<EFieldComponent>;

