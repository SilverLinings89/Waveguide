// Tests for the bookkeeping of degrees of freedom in FEDomain and for the sectors of the shape description.
#include <gtest/gtest.h>
#include <climits>
#include <cmath>
#include <cstring>
#include <new>
#include "../Code/Core/FEDomain.h"
#include "../Code/Core/Sector.h"
#include "../Code/GlobalObjects/GlobalObjects.h"

namespace {

  // A domain with a fixed number of unknowns.
  class DomainStub final : public FEDomain {
   public:
    DofCount compute_n_locally_owned_dofs() override { return 0; }
    DofCount compute_n_locally_active_dofs() override { return 0; }
    void determine_non_owned_dofs() override {}
  };

}

TEST(FEDomainTests, NewDomainHasNoUnknowns) {
  DomainStub domain;
  EXPECT_EQ(domain.n_locally_active_dofs, 0u);
  EXPECT_EQ(domain.n_locally_owned_dofs, 0u);
  EXPECT_TRUE(domain.global_index_mapping.empty());
  EXPECT_TRUE(domain.is_dof_owned.empty());
}

TEST(FEDomainTests, UnknownsAreOwnedAndWithoutIndexAfterCounting) {
  DomainStub domain;
  domain.initialize_dof_counts(7, 5);
  EXPECT_EQ(domain.n_locally_active_dofs, 7u);
  EXPECT_EQ(domain.n_locally_owned_dofs, 5u);
  ASSERT_EQ(domain.global_index_mapping.size(), 7u);
  ASSERT_EQ(domain.is_dof_owned.size(), 7u);
  for (unsigned int i = 0; i < 7; i++) {
    EXPECT_EQ(domain.global_index_mapping[i], UINT_MAX);
    EXPECT_TRUE(domain.is_dof_owned[i]);
  }
}

// Owned unknowns are numbered consecutively from the first index of the domain; unknowns of other domains keep the
// index they got from their owner.
TEST(FEDomainTests, OwnedUnknownsAreNumberedConsecutively) {
  DomainStub domain;
  domain.initialize_dof_counts(6, 4);
  domain.mark_local_dofs_as_non_local({1, 4});
  domain.freeze_ownership();
  EXPECT_TRUE(domain.finish_initialization(10));
  domain.set_non_local_dof_indices({1, 4}, {55, 3});
  const DofIndexVector expected = {10, 55, 11, 12, 3, 13};
  EXPECT_EQ(domain.global_index_mapping, expected);
  EXPECT_FALSE(domain.is_dof_owned[1]);
  EXPECT_FALSE(domain.is_dof_owned[4]);
  EXPECT_TRUE(domain.is_dof_owned[0]);
  const DofIndexVector transformed = domain.transform_local_to_global_dofs({5, 0, 1});
  const DofIndexVector expected_transformed = {13, 10, 55};
  EXPECT_EQ(transformed, expected_transformed);
  EXPECT_TRUE(domain.transform_local_to_global_dofs({}).empty());
}

// Indices that were set before the numbering (as the surfaces do for the unknowns of the inner domain) survive it.
TEST(FEDomainTests, NumberingKeepsIndicesOfOtherDomains) {
  DomainStub domain;
  domain.initialize_dof_counts(4, 2);
  domain.mark_local_dofs_as_non_local({0, 3});
  domain.set_non_local_dof_indices({0, 3}, {7, 8});
  domain.freeze_ownership();
  EXPECT_TRUE(domain.finish_initialization(100));
  const DofIndexVector expected = {7, 100, 101, 8};
  EXPECT_EQ(domain.global_index_mapping, expected);
}

// FEDomain.cpp:4: the constructor does not initialize is_ownership_ready; in memory that held ones before, finish_initialization() numbers the dofs without freeze_ownership().
TEST(FEDomainTests, DISABLED_NumberingBeforeFreezingOwnershipIsRefused) {
  alignas(DomainStub) unsigned char memory[sizeof(DomainStub)];
  std::memset(memory, 1, sizeof(memory));
  // Default initialization, as for the domains of the solver, which have constructors of their own.
  DomainStub * domain = new (memory) DomainStub;
  domain->initialize_dof_counts(3, 3);
  EXPECT_FALSE(domain->finish_initialization(0));
  EXPECT_EQ(domain->global_index_mapping[0], UINT_MAX);
  domain->~DomainStub();
}

TEST(FEDomainTests, LocalVectorAndNormUseTheNumberingOfTheLevel) {
  DomainStub domain;
  domain.initialize_dof_counts(3, 2);
  domain.mark_local_dofs_as_non_local({1});
  domain.freeze_ownership();
  domain.finish_initialization(4);
  domain.set_non_local_dof_indices({1}, {0});
  dealii::IndexSet all_dofs(6);
  all_dofs.add_range(0, 6);
  NumericVectorDistributed vector(all_dofs, MPI_COMM_SELF);
  for (unsigned int i = 0; i < 6; i++) {
    vector[i] = ComplexNumber(i + 1.0, -1.0 * i);
  }
  vector.compress(dealii::VectorOperation::insert);
  // Local unknowns 0, 1, 2 are the unknowns 4, 0, 5 of the level.
  const NumericVectorLocal local = domain.get_local_vector_from_global(vector);
  ASSERT_EQ(local.size(), 3u);
  EXPECT_EQ(local[0], ComplexNumber(5, -4));
  EXPECT_EQ(local[1], ComplexNumber(1, 0));
  EXPECT_EQ(local[2], ComplexNumber(6, -5));
  // Only owned unknowns enter the norm, so that the norms of all domains add up to the norm of the vector.
  EXPECT_NEAR(domain.local_norm_of_vector(&vector), std::sqrt(25.0 + 16.0 + 36.0 + 25.0), 1e-13);
}

// Cubic interpolation between the two surfaces of the sector: the shift m takes the values m_0 and m_1 with the
// slopes v_0 and v_1; the tilt v takes the values v_0 and v_1.
TEST(SectorTests, InterpolatesShiftAndTiltBetweenSurfaces) {
  GlobalParams = Parameters();
  Sector<2> sector(false, false, 2.0, 3.0);
  const double m_0 = 0.3, m_1 = -0.5, v_0 = 0.2, v_1 = 0.7;
  sector.set_properties(m_0, m_1, v_0, v_1);
  EXPECT_NEAR(sector.get_m(0.0), m_0, 1e-14);
  EXPECT_NEAR(sector.get_m(1.0), m_1, 1e-14);
  EXPECT_NEAR(sector.get_v(0.0), v_0, 1e-14);
  EXPECT_NEAR(sector.get_v(1.0), v_1, 1e-14);
  const double h = 1e-6;
  EXPECT_NEAR((sector.get_m(h) - sector.get_m(0.0)) / h, v_0, 1e-5);
  EXPECT_NEAR((sector.get_m(1.0) - sector.get_m(1.0 - h)) / h, v_1, 1e-5);
  // Outside of the sector the values of the surfaces are used.
  EXPECT_NEAR(sector.get_m(-0.5), m_0, 1e-14);
  EXPECT_NEAR(sector.get_m(1.5), m_1, 1e-14);
  EXPECT_EQ(sector.z_0, 2.0);
  EXPECT_EQ(sector.z_1, 3.0);
  EXPECT_EQ(sector.getNDofs(), 2u);
}

// A sector at an end of the waveguide keeps the values of the connector, unless they are set by force.
TEST(SectorTests, EndsOfTheWaveguideAreOnlyChangedByForce) {
  GlobalParams = Parameters();
  Sector<2> first(true, false, 0.0, 1.0);
  first.set_properties(0.3, -0.5, 0.2, 0.7);
  EXPECT_NEAR(first.get_m(0.0), 0.0, 1e-14);
  EXPECT_NEAR(first.get_v(0.0), 0.0, 1e-14);
  EXPECT_NEAR(first.get_m(1.0), -0.5, 1e-14);
  EXPECT_NEAR(first.get_v(1.0), 0.7, 1e-14);
  Sector<2> last(false, true, 0.0, 1.0);
  last.set_properties(0.3, -0.5, 0.2, 0.7);
  EXPECT_NEAR(last.get_m(0.0), 0.3, 1e-14);
  EXPECT_NEAR(last.get_m(1.0), 0.0, 1e-14);
  EXPECT_NEAR(last.get_v(1.0), 0.0, 1e-14);
  first.set_properties_force(0.3, -0.5, 0.2, 0.7);
  EXPECT_NEAR(first.get_m(0.0), 0.3, 1e-14);
  EXPECT_NEAR(first.get_v(0.0), 0.2, 1e-14);
}

TEST(SectorTests, StraightSectorHasIdentityAsTransformationTensor) {
  GlobalParams = Parameters();
  Sector<2> sector(false, false, 0.0, 1.0);
  sector.set_properties(0, 0, 0, 0);
  for (double z : {0.0, 0.3, 1.0}) {
    const dealii::Tensor<2, 3, double> tensor = sector.TransformationTensorInternal(0.4, -0.2, z);
    for (unsigned int i = 0; i < 3; i++) {
      for (unsigned int j = 0; j < 3; j++) {
        EXPECT_NEAR(tensor[i][j], (i == j) ? 1.0 : 0.0, 1e-14);
      }
    }
  }
}

TEST(SectorTests, CountersAreStored) {
  GlobalParams = Parameters();
  Sector<3> sector(false, false, 0.0, 1.0);
  EXPECT_EQ(sector.getNDofs(), 3u);
  sector.setLowestDof(4);
  sector.setNDofs(9);
  sector.setNInternalBoundaryDofs(2);
  sector.setNActiveCells(100);
  EXPECT_EQ(sector.getLowestDof(), 4u);
  EXPECT_EQ(sector.getNDofs(), 9u);
  EXPECT_EQ(sector.getNInternalBoundaryDofs(), 2u);
  EXPECT_EQ(sector.getNActiveCells(), 100u);
}

// Sector.cpp:16 sets boundary = left && right, the documentation of the member says "either" (the member is not read).
TEST(SectorTests, DISABLED_BoundaryFlagIsSetAtEitherEnd) {
  EXPECT_TRUE(Sector<2>(true, false, 0.0, 1.0).boundary);
  EXPECT_TRUE(Sector<2>(false, true, 0.0, 1.0).boundary);
  EXPECT_TRUE(Sector<2>(true, true, 0.0, 1.0).boundary);
  EXPECT_FALSE(Sector<2>(false, false, 0.0, 1.0).boundary);
}
