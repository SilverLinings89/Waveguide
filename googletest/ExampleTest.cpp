//
// Created by kraft on 25.07.19.
//

#include "../Code/Core/DOFManager.h"
#include "../Code/HSIEPreconditioner/LaguerreFunction.h"

#include "gtest/gtest.h"
#include "../Code/HSIEPreconditioner/HSIESurface.h"
#include "../Code/HSIEPreconditioner/HSIEPolynomial.h"
#include "../Code/HSIEPreconditioner/HSIESurface.cpp"
#include <deal.II/grid/grid_generator.h>

static unsigned int dofs_per_edge(unsigned int hsie_Order, unsigned int inner_order) {
    switch(hsie_Order) {
        case 5:
            if(inner_order == 0) return 7;
            if(inner_order == 1) return 21;
            if(inner_order == 2) return 35;
        case 10:
            if(inner_order == 0) return 12;
            if(inner_order == 1) return 36;
            if(inner_order == 2) return 60;
        default:
            return 0;
    }
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
    std::map<dealii::Triangulation<2,3>::cell_iterator, dealii::Triangulation<3,3>::face_iterator > association;
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
        const unsigned int dest_cells = Cells_Per_Direction*Cells_Per_Direction*Cells_Per_Direction;
        ASSERT_EQ(tria.n_active_cells(), dest_cells);

        std::set<unsigned int> b_ids;
        b_ids.insert(5);
        association = dealii::GridGenerator::extract_boundary_mesh( tria, temp_triangulation, b_ids);
        const unsigned int dest_surf_cells = Cells_Per_Direction*Cells_Per_Direction;
        ASSERT_EQ(temp_triangulation.n_active_cells(), dest_surf_cells);
        dealii::GridGenerator::flatten_triangulation(temp_triangulation, surf_tria);
        ASSERT_EQ(surf_tria.n_active_cells(), dest_surf_cells);
    }
};

TEST_P(TestOrderFixture, AssemblationTestOrder5) {

    HSIESurface< 5 > surf(surf_tria, 0, 0, InnerOrder, k0, association);

    surf.initialize();

    ASSERT_EQ(surf.compute_dofs_per_vertex(), dofs_per_vertex(5) * 2 );
    ASSERT_EQ(surf.compute_dofs_per_edge(false), dofs_per_edge(5, InnerOrder) * 2 );
    ASSERT_EQ(surf.compute_dofs_per_face(false), dofs_per_face(5, InnerOrder) * 2 );

    ASSERT_EQ((Cells_Per_Direction+1) * (Cells_Per_Direction+1)  *surf.compute_dofs_per_vertex(), surf.vertex_dof_data.size());
    ASSERT_EQ(2 * Cells_Per_Direction * (Cells_Per_Direction+1) * surf.compute_dofs_per_edge(false), surf.edge_dof_data.size());
    ASSERT_EQ(Cells_Per_Direction * Cells_Per_Direction * surf.compute_dofs_per_face(false), surf.face_dof_data.size());

}

TEST_P(TestOrderFixture, AssemblationTestOrder10) {

    HSIESurface< 10 > surf(surf_tria, 0, 0, InnerOrder, k0, association);

    surf.initialize();
    ASSERT_EQ(surf.compute_dofs_per_vertex(), dofs_per_vertex(10) * 2 );
    ASSERT_EQ(surf.compute_dofs_per_edge(false), dofs_per_edge(10, InnerOrder) * 2 );
    ASSERT_EQ(surf.compute_dofs_per_face(false), dofs_per_face(10, InnerOrder) * 2 );

    ASSERT_EQ((Cells_Per_Direction+1) * (Cells_Per_Direction+1)  *surf.compute_dofs_per_vertex(), surf.vertex_dof_data.size());
    ASSERT_EQ(2 * Cells_Per_Direction * (Cells_Per_Direction+1) * surf.compute_dofs_per_edge(false), surf.edge_dof_data.size());
    ASSERT_EQ(Cells_Per_Direction * Cells_Per_Direction * surf.compute_dofs_per_face(false), surf.face_dof_data.size());
    unsigned int total_dof_count = surf.face_dof_data.size() + surf.edge_dof_data.size() + surf.vertex_dof_data.size();
    IndexSet hsie_dof_indices(total_dof_count);
    const unsigned int max_couplings =  dofs_per_vertex(5) * 2 *9 + dofs_per_edge(10, InnerOrder) * 2 * 12 + dofs_per_face(10, InnerOrder) * 2 * 4;
    dealii::SparsityPattern sp(total_dof_count, max_couplings);
    dealii::SparseMatrix<double> sys_matrix(sp);
    surf.fill_matrix(&sys_matrix, hsie_dof_indices);
    ASSERT_NE(sys_matrix.linfty_norm(), 0);
}

INSTANTIATE_TEST_SUITE_P(HSIESurfaceTests, TestOrderFixture, ::testing::Combine( ::testing::Values(0,1,2), ::testing::Values(5,9)));


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
    ASSERT_EQ(product.frobenius_norm(), std::sqrt(HSIEPolynomial::I.size(0)));
}

TEST(DOFManagerTests, StaticFunctionTest) {
    ASSERT_EQ(DOFManager::testValue(),4);
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