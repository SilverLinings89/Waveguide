#pragma once

// Shared setup for the tests of the boundary conditions, the inner domain and the local problem. The code under test
// reads its configuration from global objects. ProblemEnvironment sets all of them for a single process the way
// initialize_global_variables() does (see Code/GlobalObjects/GlobalObjects.cpp) and releases them again.
#include <gtest/gtest.h>
#include <unistd.h>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <deal.II/base/function.h>
#include "../Code/BoundaryCondition/BoundaryCondition.h"
#include "../Code/Core/InnerDomain.h"
#include "../Code/Core/Types.h"
#include "../Code/GlobalObjects/GlobalObjects.h"
#include "../Code/Helpers/staticfunctions.h"
#include "../Code/Solutions/ExactSolution.h"
#include "../Code/SpaceTransformations/WaveguideTransformation.h"

namespace helpers_c {

  struct SetupOptions {
    unsigned int cells_x = 4;
    unsigned int cells_y = 4;
    unsigned int cells_z = 4;
    double size_x = 4.0;
    double size_y = 3.6;
    double size_z = 2.0;
    unsigned int fe_order = 0;
    BoundaryConditionType boundary_method = BoundaryConditionType::PML;
    SignalCouplingMethod coupling = SignalCouplingMethod::Dirichlet;
    bool pec_side_walls = false;
    // 0: mode of the dielectric waveguide, 4: TE10 mode of a waveguide with metallic walls.
    unsigned int source_type = 0;
    double epsilon_in = 2.3409;
    double epsilon_out = 1.8496;
    unsigned int pml_layers = 3;
    double pml_thickness = 0.75;
    double pml_sigma_max = 10.0;
    unsigned int pml_order = 3;
    unsigned int hsie_degree = 4;
    ComplexNumber kappa_0 = ComplexNumber(4.0537, 0.0);
    double vertical_displacement = 0.0;
    unsigned int n_sectors = 1;
  };

  // Number of unknowns of a Nedelec element of order k (k = 0 is the lowest order) on a mesh of nx x ny x nz
  // hexahedra: k + 1 per edge, 2 k (k + 1) per face and 3 k^2 (k + 1) per cell.
  inline unsigned int n_nedelec_dofs(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int k) {
    const unsigned int n_lines = nx * (ny + 1) * (nz + 1) + ny * (nx + 1) * (nz + 1) + nz * (nx + 1) * (ny + 1);
    const unsigned int n_quads = nx * ny * (nz + 1) + nx * nz * (ny + 1) + ny * nz * (nx + 1);
    const unsigned int n_hexes = nx * ny * nz;
    return (k + 1) * n_lines + 2 * k * (k + 1) * n_quads + 3 * k * k * (k + 1) * n_hexes;
  }

  // The same for the tangential trace on a face that consists of na x nb quadrilaterals.
  inline unsigned int n_nedelec_dofs_on_face(unsigned int na, unsigned int nb, unsigned int k) {
    const unsigned int n_lines = na * (nb + 1) + nb * (na + 1);
    return (k + 1) * n_lines + 2 * k * (k + 1) * na * nb;
  }

  // Removes the surfaces and the surface meshes of earlier initializations. GeometryManager::initialize() only builds
  // the surface meshes once, so without this a second geometry would be combined with the meshes of the first one.
  inline void drop_geometry_objects() {
    for (unsigned int level = 0; level < 4; level++) {
      for (unsigned int surface = 0; surface < 6; surface++) {
        Geometry.levels[level].surfaces[surface].reset();
      }
    }
    for (unsigned int surface = 0; surface < 6; surface++) {
      Geometry.surface_meshes[surface].clear();
    }
    Geometry.are_surface_meshes_initialized = false;
  }

  class ProblemEnvironment {
   public:
    explicit ProblemEnvironment(const SetupOptions & options) {
      previous_cout_buffer = std::cout.rdbuf(captured_output.rdbuf());
      previous_output_folder = GlobalOutputManager.output_folder_path;

      GlobalParams = Parameters();
      GlobalParams.NumberProcesses = 1;
      GlobalParams.MPI_Rank = 0;
      GlobalParams.Blocks_in_x_direction = 1;
      GlobalParams.Blocks_in_y_direction = 1;
      GlobalParams.Blocks_in_z_direction = 1;
      GlobalParams.Sweeping_Level = 1;
      GlobalParams.Cells_in_x = options.cells_x;
      GlobalParams.Cells_in_y = options.cells_y;
      GlobalParams.Cells_in_z = options.cells_z;
      GlobalParams.Geometry_Size_X = options.size_x;
      GlobalParams.Geometry_Size_Y = options.size_y;
      GlobalParams.Geometry_Size_Z = options.size_z;
      GlobalParams.Nedelec_element_order = options.fe_order;
      GlobalParams.BoundaryCondition = options.boundary_method;
      GlobalParams.Signal_coupling_method = options.coupling;
      GlobalParams.use_tapered_input_signal = (options.coupling == SignalCouplingMethod::Tapering);
      GlobalParams.pec_side_walls = options.pec_side_walls;
      GlobalParams.Point_Source_Type = options.source_type;
      GlobalParams.Epsilon_R_in_waveguide = options.epsilon_in;
      GlobalParams.Epsilon_R_outside_waveguide = options.epsilon_out;
      GlobalParams.PML_N_Layers = options.pml_layers;
      GlobalParams.PML_thickness = options.pml_thickness;
      GlobalParams.PML_Sigma_Max = options.pml_sigma_max;
      GlobalParams.PML_skaling_order = options.pml_order;
      GlobalParams.HSIE_polynomial_degree = options.hsie_degree;
      GlobalParams.kappa_0 = options.kappa_0;
      GlobalParams.Vertical_displacement_of_waveguide = options.vertical_displacement;
      GlobalParams.Number_of_sectors = options.n_sectors;
      GlobalParams.Logging_Level = LoggingLevel::DEBUG_ALL;
      ExactSolution::load_data(repository_path("Modes/output_mode.dat"));
      // Sets omega, the indices of the process and the source field.
      GlobalParams.complete_data();

      GlobalMPI = MPICommunicator();
      GlobalMPI.initialize();

      drop_geometry_objects();
      Geometry.initialize();

      std::string directory_template = (std::filesystem::temp_directory_path() / "waveguide_unit_tests_XXXXXX").string();
      if (mkdtemp(directory_template.data()) == nullptr) {
        ADD_FAILURE() << "The temporary output directory could not be created.";
      }
      output_directory = directory_template;
      GlobalOutputManager.output_folder_path = output_directory;

      GlobalTimerManager.finalize();
      GlobalTimerManager.initialize();

      transformation.emplace();
      GlobalSpaceTransformation = &(*transformation);
      GlobalSpaceTransformation->estimate_and_initialize();
    }

    ~ProblemEnvironment() {
      // The timers hold MPI communicators and write into the output directory.
      GlobalTimerManager.finalize();
      drop_geometry_objects();
      // The inner domain was created by Geometry.initialize() and is of exactly this type.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
      delete Geometry.levels[0].inner_domain;
#pragma GCC diagnostic pop
      Geometry.levels[0].inner_domain = nullptr;
      GlobalSpaceTransformation = nullptr;
      delete GlobalParams.source_field;
      // Tests that do not set the parameters they read get the defaults.
      GlobalParams = Parameters();
      GlobalParams.source_field = nullptr;
      GlobalOutputManager.output_folder_path = previous_output_folder;
      std::error_code error;
      std::filesystem::remove_all(output_directory, error);
      std::cout.rdbuf(previous_cout_buffer);
    }

    ProblemEnvironment(const ProblemEnvironment &) = delete;
    ProblemEnvironment & operator=(const ProblemEnvironment &) = delete;

    InnerDomain & inner_domain() const {
      return *Geometry.levels[0].inner_domain;
    }

    template <class SurfaceClass>
    SurfaceClass * surface(BoundaryId b_id) const {
      return dynamic_cast<SurfaceClass *>(Geometry.levels[0].surfaces[b_id].get());
    }

    // Replaces the source field (the "exact solution"). The environment takes ownership.
    void set_source_field(dealii::Function<3, ComplexNumber> * field) {
      delete GlobalParams.source_field;
      GlobalParams.source_field = field;
    }

    // Everything the code under test has written to std::cout so far.
    std::string output() const {
      return captured_output.str();
    }

    std::string output_directory;

   private:
    std::optional<WaveguideTransformation> transformation;
    std::ostringstream captured_output;
    std::streambuf * previous_cout_buffer;
    std::string previous_output_folder;
  };

  // The field E = constant + (0, slope_x * x + slope_z * z, 0). On a mesh of axis parallel cells it is contained in the
  // Nedelec space of every order, so projections and interpolations reproduce it up to rounding.
  class LinearField : public dealii::Function<3, ComplexNumber> {
   public:
    LinearField(ComplexNumber constant_x, ComplexNumber constant_y, ComplexNumber constant_z, ComplexNumber in_slope_x = 0, ComplexNumber in_slope_z = 0)
        : dealii::Function<3, ComplexNumber>(3), constant{constant_x, constant_y, constant_z}, slope_x(in_slope_x), slope_z(in_slope_z) {}

    ComplexNumber value(const Position & p, const unsigned int component = 0) const override {
      if (component == 1) {
        return constant[1] + slope_x * p[0] + slope_z * p[2];
      }
      return constant[component];
    }

    void vector_value(const Position & p, dealii::Vector<ComplexNumber> & values) const override {
      for (unsigned int i = 0; i < 3; i++) {
        values[i] = value(p, i);
      }
    }

   private:
    const ComplexNumber constant[3];
    const ComplexNumber slope_x;
    const ComplexNumber slope_z;
  };

  // Messages that the code prints when its bookkeeping of the degrees of freedom is inconsistent.
  inline std::vector<std::string> dof_bookkeeping_complaints(const std::string & output) {
    std::vector<std::string> found;
    for (const std::string & message : {"mismatch", "CASE A", "invalid dofs", "There was an error", "Identity failure", "Error in dof assignment", "wrong values"}) {
      if (output.find(message) != std::string::npos) {
        found.push_back(message);
      }
    }
    return found;
  }

  // The local numbering of a domain mapped to the numbering of the level.
  inline std::vector<DofNumber> to_level_numbering(FEDomain & domain, const std::vector<InterfaceDofData> & dofs) {
    std::vector<DofNumber> ret;
    for (const InterfaceDofData & dof : dofs) {
      ret.push_back(domain.global_index_mapping[dof.index]);
    }
    return ret;
  }

}
