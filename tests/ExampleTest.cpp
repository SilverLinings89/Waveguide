#include "../Code/HSIEPreconditioner/LaguerreFunction.h"
#include "gtest/gtest.h"
#include "../Code/HSIEPreconditioner/HSIESurface.h"
#include "../Code/HSIEPreconditioner/HSIEPolynomial.h"
#include "../Code/HSIEPreconditioner/HSIESurface.cpp"
#include <deal.II/grid/grid_generator.h>

Parameters GlobalParams;
GeometryManager Geometry;
MPICommunicator GlobalMPI;
ModeManager GlobalModeManager;
std::string solutionpath = "";
std::ofstream log_stream;
std::string constraints_filename = "constraints.log";
std::string assemble_filename = "assemble.log";
std::string precondition_filename = "precondition.log";
std::string solver_filename = "solver.log";
std::string total_filename = "total.log";
int StepsR = 10;
int StepsPhi = 10;
int alert_counter = 0;
std::string input_file_name;
SpaceTransformation *the_st;

static HSIEPolynomial random_poly(unsigned int Order, std::complex<double> k0) {
    std::vector<std::complex<double>> a;
    for(unsigned int i = 0; i < Order; i++) {
        a.emplace_back(rand()%10, rand()%10);
    }
    return HSIEPolynomial(a, k0);
}

static unsigned int dofs_per_edge(unsigned int hsie_Order, unsigned int inner_order) {
    switch(hsie_Order) {
        case 5:
            if(inner_order == 0) return 7;
            if(inner_order == 1) return 21;
            if(inner_order == 2) return 35;
            break;
        case 10:
            if(inner_order == 0) return 12;
            if(inner_order == 1) return 36;
            if(inner_order == 2) return 60;
            break;
        default:
            return 0;
    }
    return 0;
}

static unsigned int dofs_per_vertex(unsigned int hsie_Order) {
    return hsie_Order + 2;
}

static unsigned int dofs_per_face(unsigned int hsie_Order, unsigned int inner_order) {
    if(inner_order == 0) {
        return 0;
    }
    if(inner_order == 1) {
        return (hsie_Order + 2) * 3 * 4;
    }
    if(inner_order == 2) {
        return (hsie_Order + 2) * 3 * 12;
    }
    return 0;
}

class TestData {
public:
    unsigned int Inner_Element_Order;
    unsigned int Cells_Per_Direction;

    TestData( unsigned int in_inner_element, unsigned int in_cells)
    {
        Inner_Element_Order = in_inner_element;
        Cells_Per_Direction = in_cells;
    }

    TestData( const TestData &in_td) {
        Inner_Element_Order = in_td.Inner_Element_Order;
        Cells_Per_Direction = in_td.Cells_Per_Direction;
    }
};

class TestOrderFixture : public ::testing::TestWithParam<std::tuple<unsigned  int, unsigned int>> {
public:
    unsigned int Cells_Per_Direction;
    unsigned int InnerOrder;
    dealii::Triangulation<3> tria ;
    dealii::Triangulation<2,3> temp_triangulation;
    dealii::Triangulation<2> surf_tria;
    std::complex<double> k0;
protected:
    void SetUp() override {
        std::tuple<unsigned  int, unsigned int> Params = GetParam();
        Cells_Per_Direction = std::get<1>(Params);
        InnerOrder = std::get<0>(Params);
        k0= {0.0, -1.0};
        std::vector<unsigned int> repetitions;
        repetitions.push_back(Cells_Per_Direction);
        repetitions.push_back(Cells_Per_Direction);
        repetitions.push_back(Cells_Per_Direction);
        dealii::Point<3,double> left(-1, -1, -1);
        dealii::Point<3,double> right(1, 1, 1);
        dealii::GridGenerator::subdivided_hyper_rectangle(tria, repetitions, left, right, true);
        std::set<unsigned int> b_ids;
        b_ids.insert(4);
        dealii::GridTools::transform(Transform_4_to_5, tria);
    dealii::GridGenerator::extract_boundary_mesh(tria, temp_triangulation,
        b_ids);
        dealii::GridGenerator::flatten_triangulation(temp_triangulation, surf_tria);
    }
};


class TestDirectionFixture : public ::testing::TestWithParam<std::tuple<unsigned  int, unsigned int, unsigned int>> {
public:
    unsigned int Cells_Per_Direction;
    unsigned int InnerOrder;
    unsigned int boundary_id;
    dealii::Triangulation<3> tria ;
    dealii::Triangulation<2,3> temp_triangulation;
    dealii::Triangulation<2> surf_tria;
    std::complex<double> k0;
    std::set<unsigned int> b_ids;
protected:
    void SetUp() override {
        std::tuple<unsigned  int, unsigned int, unsigned int> Params = GetParam();
        Cells_Per_Direction = std::get<1>(Params);
        InnerOrder = std::get<0>(Params);
        boundary_id = std::get<2>(Params);
        k0= {0.0, -1.0};
        std::vector<unsigned int> repetitions;
        repetitions.push_back(Cells_Per_Direction);
        repetitions.push_back(Cells_Per_Direction);
        repetitions.push_back(Cells_Per_Direction);
        dealii::Point<3,double> left(-1, -1, -1);
        dealii::Point<3,double> right(1, 1, 1);
        dealii::GridGenerator::subdivided_hyper_rectangle(tria, repetitions, left, right, true);
        const unsigned int dest_cells = Cells_Per_Direction*Cells_Per_Direction*Cells_Per_Direction;
        ASSERT_EQ(tria.n_active_cells(), dest_cells);
        b_ids.insert(boundary_id);
        switch(boundary_id) {
            case 0:
                dealii::GridTools::transform(Transform_0_to_5, tria);
                break;
            case 1:
                dealii::GridTools::transform(Transform_1_to_5, tria);
                break;
            case 2:
                dealii::GridTools::transform(Transform_2_to_5, tria);
                break;
            case 3:
                dealii::GridTools::transform(Transform_3_to_5, tria);
                break;
            case 4:
                dealii::GridTools::transform(Transform_4_to_5, tria);
                break;
        }
    }
};

TEST_P(TestDirectionFixture, TestCellRequirements) {
  dealii::GridGenerator::extract_boundary_mesh(tria, temp_triangulation, b_ids);
    const unsigned int dest_surf_cells = Cells_Per_Direction*Cells_Per_Direction;
    ASSERT_EQ(temp_triangulation.n_active_cells(), dest_surf_cells);
    dealii::GridGenerator::flatten_triangulation(temp_triangulation, surf_tria);
    ASSERT_EQ(surf_tria.n_active_cells(), dest_surf_cells);
}

TEST_P(TestDirectionFixture, DofNumberingTest1) {
  dealii::GridGenerator::extract_boundary_mesh(tria, temp_triangulation, b_ids);
    dealii::GridGenerator::flatten_triangulation(temp_triangulation, surf_tria);
  const unsigned int component = 0;
  auto temp_it = temp_triangulation.begin();
  double additional_coorindate = temp_it->center()[component];
  HSIESurface surf(5, surf_tria, boundary_id, InnerOrder, k0,
      additional_coorindate);
    surf.initialize();
    std::vector< types::boundary_id > boundary_ids_of_flattened_mesh = surf.get_boundary_ids();
    ASSERT_EQ(boundary_ids_of_flattened_mesh.size(), 4);
    auto it = std::find(boundary_ids_of_flattened_mesh.begin(), boundary_ids_of_flattened_mesh.end(), boundary_id);
    ASSERT_EQ(boundary_ids_of_flattened_mesh.end(), it);
}


TEST_P(TestOrderFixture, AssemblationTestOrder5) {
  const unsigned int component = 0;
  auto temp_it = temp_triangulation.begin();
  double additional_coorindate = temp_it->center()[component];
  HSIESurface surf(5, surf_tria, 0, InnerOrder, k0,
      additional_coorindate);
    surf.initialize();
    ASSERT_EQ(surf.compute_dofs_per_vertex(), dofs_per_vertex(5) * 2 );
    ASSERT_EQ(surf.compute_dofs_per_edge(false), dofs_per_edge(5, InnerOrder) * 2 );
    ASSERT_EQ(surf.compute_dofs_per_face(false), dofs_per_face(5, InnerOrder) * 2 );
    ASSERT_EQ((Cells_Per_Direction+1) * (Cells_Per_Direction+1)  *surf.compute_dofs_per_vertex(), surf.vertex_dof_data.size());
    ASSERT_EQ(2 * Cells_Per_Direction * (Cells_Per_Direction+1) * surf.compute_dofs_per_edge(false), surf.edge_dof_data.size());
    ASSERT_EQ(Cells_Per_Direction * Cells_Per_Direction * surf.compute_dofs_per_face(false) / 2, surf.face_dof_data.size());
    ASSERT_TRUE(surf.check_number_of_dofs_for_cell_integrity());
    ASSERT_TRUE(surf.check_dof_assignment_integrity());
    unsigned int total_dof_count = surf.face_dof_data.size() + surf.edge_dof_data.size() + surf.vertex_dof_data.size();
    IndexSet hsie_dof_indices(total_dof_count);
    hsie_dof_indices.add_range(0,total_dof_count);
    dealii::DynamicSparsityPattern dsp(total_dof_count, total_dof_count);
    surf.fill_sparsity_pattern(&dsp);
    dealii::SparsityPattern sp;
    sp.copy_from(dsp);
    sp.compress();
  dealii::SparseMatrix<std::complex<double>> sys_matrix(sp);
    surf.fill_matrix(&sys_matrix, hsie_dof_indices);
    std::cout << "L1 Norm: " << sys_matrix.l1_norm() << std::endl;
    std::cout << "L infty Norm: " << sys_matrix.linfty_norm() << std::endl;
    ASSERT_NE(sys_matrix.linfty_norm(), 0);
    ASSERT_NE(sys_matrix.l1_norm(), 0);
}

TEST_P(TestOrderFixture, AssemblationTestOrder10) {
  const unsigned int component = 0;
  auto temp_it = temp_triangulation.begin();
  double additional_coorindate = temp_it->center()[component];
  HSIESurface surf(10, surf_tria, 0, InnerOrder, k0,
      additional_coorindate);
    surf.initialize();
    ASSERT_EQ(surf.compute_dofs_per_vertex(), dofs_per_vertex(10) * 2 );
    ASSERT_EQ(surf.compute_dofs_per_edge(false), dofs_per_edge(10, InnerOrder) * 2 );
    ASSERT_EQ(surf.compute_dofs_per_face(false), dofs_per_face(10, InnerOrder) * 2 );
    ASSERT_EQ((Cells_Per_Direction+1) * (Cells_Per_Direction+1)  *surf.compute_dofs_per_vertex(), surf.vertex_dof_data.size());
    ASSERT_EQ(2 * Cells_Per_Direction * (Cells_Per_Direction+1) * surf.compute_dofs_per_edge(false), surf.edge_dof_data.size());
    ASSERT_EQ(Cells_Per_Direction * Cells_Per_Direction * surf.compute_dofs_per_face(false) / 2, surf.face_dof_data.size());
    ASSERT_TRUE(surf.check_number_of_dofs_for_cell_integrity());
    ASSERT_TRUE(surf.check_dof_assignment_integrity());

    unsigned int total_dof_count = surf.face_dof_data.size() + surf.edge_dof_data.size() + surf.vertex_dof_data.size();
    IndexSet hsie_dof_indices(total_dof_count);
    hsie_dof_indices.add_range(0,total_dof_count);
    dealii::DynamicSparsityPattern dsp(total_dof_count, total_dof_count);
    surf.fill_sparsity_pattern(&dsp);
    dealii::SparsityPattern sp;
    sp.copy_from(dsp);
    sp.compress();
  dealii::SparseMatrix<std::complex<double>> sys_matrix(sp);
    surf.fill_matrix(&sys_matrix, hsie_dof_indices);
    std::cout << "L1 Norm: " << sys_matrix.l1_norm() << std::endl;
    std::cout << "L infty Norm: " << sys_matrix.linfty_norm() << std::endl;
    ASSERT_NE(sys_matrix.linfty_norm(), 0);
    ASSERT_NE(sys_matrix.l1_norm(), 0);
}

// INSTANTIATE_TEST_SUITE_P(HSIESurfaceTests, TestOrderFixture, ::testing::Combine( ::testing::Values(0,1,2), ::testing::Values(5,9)));
INSTANTIATE_TEST_SUITE_P(HSIESurfaceTests, TestOrderFixture, ::testing::Combine( ::testing::Values(0), ::testing::Values(5,9)));

INSTANTIATE_TEST_SUITE_P(MespPreparationTests, TestDirectionFixture,  ::testing::Combine( ::testing::Values(0), ::testing::Values(5,9), ::testing::Values(0,1,2,3,4,5)));

TEST(HSIEPolynomialTests, TestOperatorTplus) {
    std::vector<std::complex<double>> in_a;
    in_a.emplace_back(0.0, 0.0);
    in_a.emplace_back(0.0, 0.0);
    in_a.emplace_back(1.0, 0.0);
    in_a.emplace_back(0.0, 1.0);
    in_a.emplace_back(0.0, 0.0);
    std::complex<double> k0(0.0, 1.0);
    HSIEPolynomial poly(in_a, k0);
    poly.applyTplus(std::complex<double>(1,-1));
    ASSERT_EQ(poly.a[0], std::complex<double>(0.5, -0.5));
    ASSERT_EQ(poly.a[1], std::complex<double>(0, 0));
    ASSERT_EQ(poly.a[2], std::complex<double>(0.5, 0));
    ASSERT_EQ(poly.a[3], std::complex<double>(0.5, 0.5));
    ASSERT_EQ(poly.a[4], std::complex<double>(0, 0.5));
}

TEST(HSIEPolynomialTests, ProductOfDandIShouldBeIdentity) {
    std::vector<std::complex<double>> in_a;
    in_a.emplace_back(0.0, 0.0);
    in_a.emplace_back(0.0, 0.0);
    in_a.emplace_back(1.0, 0.0);
    in_a.emplace_back(0.0, 1.0);
    in_a.emplace_back(0.0, 0.0);
    std::complex<double> k0(0.0, 1.0);
    HSIEPolynomial poly(in_a, k0);
    poly.applyD();
    dealii::FullMatrix<std::complex<double>> product(HSIEPolynomial::D.size(0), HSIEPolynomial::D.size(1));
    HSIEPolynomial::D.mmult(product, HSIEPolynomial::I, false);
    ASSERT_NEAR(product.frobenius_norm(), std::sqrt(HSIEPolynomial::I.size(0)), 0.0001);
}

TEST(HSIE_ORTHOGONALITY_TESTS, EvaluationOfA) {
    std::complex<double> k0(0,-1);
    std::vector<std::complex<double>> zeroes;
    for(unsigned k = 0; k < 10; k++) {
        zeroes.emplace_back(0,0);
    }
    for(unsigned int i = 0; i < 10; i++) {
        for (unsigned int j = 0; j<10; j++) {
                std::vector<HSIEPolynomial> u;
                u.emplace_back(i, k0);
                u.emplace_back(zeroes, k0);
                u.emplace_back(zeroes, k0);
                std::vector<HSIEPolynomial> v;
                v.emplace_back(j, k0);
                v.emplace_back(zeroes, k0);
                v.emplace_back(zeroes, k0);
      std::complex<double> res = HSIESurface::evaluate_a(u, v);
            if(j != i) {
                ASSERT_NEAR(0, res.real(), 0.001);
                ASSERT_NEAR(0, res.imag(), 0.001);
            } else {
                bool eitherOr = std::abs(res.real()) > 0.0001 || std::abs(res.imag()) > 0.0001;
                ASSERT_TRUE(eitherOr);
            }
        }
    }
}

TEST(HSIE_ORTHOGONALITY_TESTS, RandomPolynomialProductTest) {
    std::complex<double> k0(0,-1);
    std::vector<std::complex<double>> zeroes;
    for(unsigned k = 0; k < 10; k++) {
        zeroes.emplace_back(0,0);
    }

    std::vector<HSIEPolynomial> u;
    u.push_back(random_poly(10,k0));
    u.push_back(random_poly(10,k0));
    u.push_back(random_poly(10,k0));
    std::vector<HSIEPolynomial> v;
    v.push_back(random_poly(10,k0));
    v.push_back(random_poly(10,k0));
    v.push_back(random_poly(10,k0));
  std::complex<double> res = HSIESurface::evaluate_a(u, v);
    std::complex<double> expected_result(0,0);
    for(unsigned j = 0; j < 3; j++) {
        for (unsigned int i = 0; i < 10; i++) {
            expected_result += u[j].a[i] * v[j].a[i];
        }
    }

    ASSERT_NEAR(expected_result.real(), res.real(), expected_result.real()/100.0);
    ASSERT_NEAR(expected_result.imag(), res.imag(), expected_result.imag()/100.0);
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_1) {
    ASSERT_EQ(LaguerreFunction::evaluate(0,13,2), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_2) {
    ASSERT_EQ(LaguerreFunction::evaluate(0,1,4), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_3) {
    for(int i = 0; i < 10; i++) {
        ASSERT_EQ(LaguerreFunction::evaluate(1,i,2), -2 + i + 1);
    }
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_4) {
    for(int i = 0; i < 10; i++) {
        ASSERT_EQ(LaguerreFunction::evaluate(2,i,2), 2 - 2*i - 4 + (i+2)*(i+1)/2);
    }
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_5) {
    for(int i = 0; i < 10; i++) {
        for(unsigned int j = 0; j < 10; j++) {
            ASSERT_EQ(LaguerreFunction::evaluate(i,j,0), LaguerreFunction::binomial_coefficient(i+j,i));
        }
    }
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_1) {
    ASSERT_EQ(LaguerreFunction::binomial_coefficient(5,2), 10);
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_2) {
    ASSERT_EQ(LaguerreFunction::binomial_coefficient(0,0), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_3) {
    ASSERT_EQ(LaguerreFunction::binomial_coefficient(1,0), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_4) {
    ASSERT_EQ(LaguerreFunction::binomial_coefficient(1,1), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_5) {
    ASSERT_EQ(LaguerreFunction::binomial_coefficient(15,5), 3003);
}

TEST(LAGUERRE_FUNCTION_TESTS, FACTORIAL_1) {
    ASSERT_EQ(LaguerreFunction::factorial(1), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, FACTORIAL_2) {
    ASSERT_EQ(LaguerreFunction::factorial(2), 2);
}

TEST(LAGUERRE_FUNCTION_TESTS, FACTORIAL_3) {
    ASSERT_EQ(LaguerreFunction::factorial(4), 24);
}

TEST(LAGUERRE_FUNCTION_TESTS, FACTORIAL_5) {
    ASSERT_EQ(LaguerreFunction::factorial(5), 120);
}
