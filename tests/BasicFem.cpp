#include "../third_party/googletest/googletest/include/gtest/gtest.h"
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/dofs/dof_handler.h>

using namespace dealii;

TEST(BASIC_FEM_TEST, LOWEST_ORDER) {
    FE_NedelecSZ<3> fe(0);
    Triangulation<3> tria(Triangulation<3>::MeshSmoothing(Triangulation<3>::none));
    GridGenerator::hyper_cube(tria);
    DoFHandler<3> dof_handler (tria);
    dof_handler.distribute_dofs(fe);
    std::cout << "DofDetails Order 0: Cell: " << fe.n_dofs_per_cell() <<  " Face: " << fe.n_dofs_per_face() << " Edge: " << fe.n_dofs_per_line() << std::endl;
    ASSERT_EQ(fe.n_dofs_per_cell(),12);
    ASSERT_EQ(fe.n_dofs_per_face(),4);
    ASSERT_EQ(fe.n_dofs_per_line(),1);
}

TEST(BASIC_FEM_TEST, ORDER_ONE) {
    FE_NedelecSZ<3> fe(1);
    Triangulation<3> tria(Triangulation<3>::MeshSmoothing(Triangulation<3>::none));
    GridGenerator::hyper_cube(tria);
    DoFHandler<3> dof_handler (tria);
    dof_handler.distribute_dofs(fe);
    std::cout << "DofDetails Order 1: Cell: " << fe.n_dofs_per_cell() <<  " Face: " << fe.n_dofs_per_face() << " Edge: " << fe.n_dofs_per_line() << std::endl;
    ASSERT_EQ(fe.n_dofs_per_cell(),54);
    ASSERT_EQ(fe.n_dofs_per_face(),12);
    ASSERT_EQ(fe.n_dofs_per_line(),2);
}

TEST(BASIC_FEM_TEST, ORDER_TWO) {
    FE_NedelecSZ<3> fe(2);
    Triangulation<3> tria(Triangulation<3>::MeshSmoothing(Triangulation<3>::none));
    GridGenerator::hyper_cube(tria);
    DoFHandler<3> dof_handler (tria);
    dof_handler.distribute_dofs(fe);
    std::cout << "DofDetails Order 2: Cell: " << fe.n_dofs_per_cell() <<  " Face: " << fe.n_dofs_per_face() << " Edge: " << fe.n_dofs_per_line() << std::endl;
    ASSERT_EQ(fe.n_dofs_per_cell(),144);
    ASSERT_EQ(fe.n_dofs_per_face(),24);
    ASSERT_EQ(fe.n_dofs_per_line(),3);
}
