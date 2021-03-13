#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include "../Code/Helpers/staticfunctions.h"
#include "../third_party/googletest/googletest/include/gtest/gtest.h"

TEST(MESH_TESTS, SORTING_TESTS) {
    dealii::Triangulation<3> tria;
    dealii::GridGenerator::subdivided_hyper_cube(tria, 10, 0, 1, true);
    dealii::Triangulation<3> sorted_tria = reforge_triangulation(&tria);
    for(auto it: sorted_tria) {
        for(unsigned int i = 0; i < 8; i++) {
            for(unsigned int j = i+1; j < 8; j++) {
                ASSERT_TRUE(comparePositions(it.vertex(i), it.vertex(j)));
            }    
        }
    }
}