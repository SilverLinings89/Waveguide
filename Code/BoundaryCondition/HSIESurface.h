#pragma once
/**
 * @file HSIESurface.h
 * @author Pascal Kraft (kraft.pascal@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include "../Core/Types.h"
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/tria.h>
#include "DofData.h"
#include "HSIEPolynomial.h"
#include "../Helpers/Parameters.h"
#include "./BoundaryCondition.h"

/**
 * \class HSIESurface
 * 
 * \brief This class implements Hardy space infinite elements on a provided surface
 * 
 * This object implements the BoundaryCondition interface. It should be considered however, that this boundary condition type is extremely complex, represented in the number of functions and lines of code it consists of. 
 * It is recommended to read the paper "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems" for an introduction.
 * 
 * In many places, you will see a distinction between q and nedelec in this implementation: Infinite cells have two types of edges: finite ones and infinite ones. The finite ones are the ones on the surface. The infinite ones point in the infinite direction.
 * The cell is basically a normal nedelec cell, but if the edge a dof is associated with, is infinite, it requires special treatment. We treat these dofs as if they were nodal elements with the center of their hat function being the base point of their inifite edge.
 * We therefore need most computations for nodal and for edge elements.
 * 
 * In the assembly loop, we have to compute terms like \f$\langle \nabla \times u, \nabla \times v\rangle\f$ and \f$\langle u,v\rangle\f$.
 * 
 * There are NO 3D triangulations here! We only work with a 2D surface triangulation. Therefore, often when we talk about a cell, that has different properties then in objects like PMLSurface or InnerDomain, where the mesh is 3D.
 * 
 * For more details on this type of intinite element, see \cref{subsec:HSIE,sub:hsieElements,sec:HSIESweeping}.
 */
class HSIESurface : public BoundaryCondition {
  const HSIEElementOrder order;
  DofHandler2D dof_h_nedelec;
  DofHandler2D dof_h_q;
  const unsigned int Inner_Element_Order;
  dealii::FE_NedelecSZ<2> fe_nedelec;
  dealii::FE_Q<2> fe_q;
  ComplexNumber k0;
  const double kappa;
  std::array<std::array<bool,6>,4> edge_ownership_by_level_and_id;
  dealii::Tensor<2,3,double> C;
  dealii::Tensor<2,3,double> G;
  Position V0;

public:
  DofDataVector face_dof_data;
  DofDataVector edge_dof_data;
  DofDataVector vertex_dof_data;
  DofCount n_edge_dofs;
  DofCount n_face_dofs;
  DofCount n_vertex_dofs;
  
  /**
     *  @brief Constructor
     *
     *  @details Prepares the data structures and sets two values.
     * 
     *  @param surface BoundaryId of the surface of the InnerDomain this condition is going to couple to.
     *  @param level the level of sweeping this object is used on.
     */
  HSIESurface(unsigned int surface, unsigned int level);
  ~HSIESurface();

  /**
     *  @brief Builds a curl-type term required during the assembly of the system matrix for a q-type dof.
     *
     *  @details This computes the curl as a std::vetor for a monomial of given order for a shape dof, whoose projected shape function on the surface is nodal (q), and requires a local gradient value as input.
     * 
     *  @param order Order of the dof we work with.
     *  @param gradient Local surface gradient.
     *  @return A three component vector containing the curl term required during assembly.
     */
  std::vector<HSIEPolynomial> build_curl_term_q(unsigned int order, const dealii::Tensor<1, 2> gradient);
  
    /**
     *  @brief Builds a curl-type term required during the assembly of the system matrix for a nedelec-type dof.
     *
     *  @details Same as above but for a nedelec dof. The computation requires two components of the gradient of the shape function and two values of the shape function. The former are provided as Tensors, the latter as individual doubles.
     * 
     *  @param order Order of the dof we work with.
     *  @param gradient_component_0 Shape function gradient component 0.
     *  @param gradient_component_1 Shape function gradient component 1.
     *  @param value_component_0 Value of shape function component 0.
     *  @param value_component_1 Value of shape function component 1.
     *  @return A three component vector containing the curl term required during assembly.
     */
  std::vector<HSIEPolynomial> build_curl_term_nedelec(unsigned int order, const dealii::Tensor<1, 2> gradient_component_0, const dealii::Tensor<1, 2> gradient_component_1, const double value_component_0, const double value_component_1);
  
  /**
   *  @brief Builds a non-curl-type term required during the assembly of the system matrix for a q-type dof.
   *
   *  @details The computation requires the value of a shape function.
   * 
   *  @param order Order of the dof we work with.
   *  @param value_component Value of shape function component.
   *  @return A three component vector containing the curl term required during assembly.
   */
  std::vector<HSIEPolynomial> build_non_curl_term_q(unsigned int order, const double value_component);
  
  /*
    *  @brief Builds a non-curl-type term required during the assembly of the system matrix for a nedelec-type dof.
    *
    *  @details Same as above but for a nedelec dof. The computation requires two components of the gradient of the shape function and two values of the shape function. The former are provided as Tensors, the latter as individual doubles. See the description of the class.
    * 
    *  @param order Order of the dof we work with.
    *  @param gradient_component_0 Shape function gradient component 0.
    *  @param gradient_component_1 Shape function gradient component 1.
    *  @param value_component_0 Value of shape function component 0.
    *  @param value_component_1 Value of shape function component 1.
    *  @return A three component vector containing the curl term required during assembly. 
    */  
  std::vector<HSIEPolynomial> build_non_curl_term_nedelec(unsigned int, const double, const double);

  /*
    *  @brief Sets the internal base_point of the infinite edges.
    *
    *  @details For the construction of the infinite directions, one method is to choose a point within the interior domain. That point connected with points on the surface provides a directions, that will not cross in the exterior.
    * 
    *  @param pos The Position to be used to construct the infinite directions.
    */  
  void set_V0(Position pos);

  /*
    *  @brief Collects all dofs active on a cell and returns them.
    *
    *  @details To perform the assembly on a cell we need to know which dofs are active there. This function generates that information.
    *  Internally, we use a surface triangulation with q and nedelec elements. We need a pointer from each datastructure.
    *  @param pointer_q Pointer to the surface cell in the q-triangulation 
    *  @param pointer_n Pointer to the surface cell in the nedelec-triangulation
    *  @return A std::vector containing a DofData object for each dof
    */  
  auto get_dof_data_for_cell(CellIterator2D pointer_q, CellIterator2D pointer_n) -> DofDataVector;

  /**
   * @brief Writes all entries to the system matrix that originate from dof couplings on this surface.
   * 
   * It also sets the values in the rhs and it uses the constraints object to condense the matrix entries automatically (see deal.IIs description on distribute_dofs_local_to_global with a constraint object).
   * 
   * @param matrix The matrix to write into.
   * @param rhs The right hand side vector (b) in Ax = b.
   * @param constraints These represent inhomogenous and hanging node constraints that are used to condense the matrix.
   */
  void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix* matrix, NumericVectorDistributed* rhs, Constraints *constraints) override;
  
  /**
   * @brief Not yet implemented.
   * 
   * When using axis parallel infinite directions, the corner and edge domains requrie additional computation of coupling terms. The function computes the coupling terms for infinite edge cells.
   * @param other_bid BoundaryId of the surface that shares the edge with this surface.
   * @param matrix The matrix to write into.
   * @param rhs The right hand side vector to write into.
   * @param constraints These represent inhomogenous and hanging node constraints that are used to condense the matrix.
   */
  void fill_matrix_for_edge(BoundaryId other_bid, dealii::PETScWrappers::MPI::SparseMatrix* matrix, NumericVectorDistributed* rhs, Constraints *constraints);

  /**
   * @brief Fills a sparsity pattern for all the dofs active in this boundary condition.
   * 
   * @param in_dsp The sparsit pattern to fill
   * @param in_constriants The constraint object to be used to condense
   */
  void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, Constraints * in_constriants) override;

  /**
   * @brief Checks if a point is at an outward surface of the boundary triangulation
   * 
   * @param in_p The position to check
   * @param in_bid The boundary id of the other surface
   * @return true if the point is located at the edge between this surface and the surface in_bid.
   * @return false if not
   */
  bool is_point_at_boundary(Position2D in_p, BoundaryId in_bid) override;

  /**
   * @brief Get the vertices located at the provided boundary
   * 
   * @return std::vector<unsigned int> Indices of the vertices at the boundary
   */
  auto get_vertices_for_boundary_id(BoundaryId in_bid) -> std::vector<unsigned int>;

  /**
   * @brief Get the number of vertices on th eboundary with id
   * 
   * @param in_bid The boundary id of the other boundary
   * @return Number of dofs on the boundary
   */
  auto get_n_vertices_for_boundary_id(BoundaryId in_bid) -> unsigned int;

  /**
   * @brief Get the lines shared with the boundary in_bid.
   * 
   * @param in_bid BoundaryID of the other boundary.
   * @return std::vector of the line ids on the boundary
   */
  auto get_lines_for_boundary_id(BoundaryId in_bid) -> std::vector<unsigned int>;

  /**
   * @brief Get the number of lines for boundary id object
   * 
   * @param in_bid The other boundary.
   * @return unsigned int Count of lines on the edge shared with the other boundary
   */
  auto get_n_lines_for_boundary_id(BoundaryId in_bid) -> unsigned int;

  /**
   * @brief Computes the number of edge dofs for this surface.
   * The return type contains the number of pure HSIE dofs, inner dofs active on the surface and the sum of both.
   * @return DofCountsStruct containing the dof counts.
   */
  auto compute_n_edge_dofs() -> DofCountsStruct;

  /**
   * @brief Computes the number of vertex dofs and returns them as a DofCounts object (see above).
   * 
   * @return DofCountsStruct The dof counts.
   */
  auto compute_n_vertex_dofs() -> DofCountsStruct;

  /**
   * @brief Computes the number of face dofs and returns them as a Dofcounts object (see above).
   * 
   * @return DofCountsStruct The dof counts.
   */
  auto compute_n_face_dofs() -> DofCountsStruct;

  /**
   * @brief Computes the number of dofs per edge
   * 
   * @param only_hsie_dofs if set to true, it only computes the number of non-inner dofs, ie only the additional dofs introduced by the boundary condition.
   * @return DofCount Number of dofs.
   */
  auto compute_dofs_per_edge(bool only_hsie_dofs) -> DofCount;

  /**
   * @brief Computes the number of dofs on every surface face
   * 
   * @param only_hsie_dofs if set to true, it only computes the number of non-inner dofs, ie only the additional dofs introduced by the boundary condition.
   * @return DofCount 
   */
  auto compute_dofs_per_face(bool only_hsie_dofs) -> DofCount;

  /**
   * @brief Computes the number of dofs on every vertex
   * 
   * All vertex dofs are automatically hardy space dofs, therefore the parameter does not exist on this fucntion.
   * 
   * @return DofCount 
   */
  auto compute_dofs_per_vertex() -> DofCount;

  /**
   * @brief Initializes the data structures.
   * 
   */
  void initialize() override;

  /**
   * @brief Part of the initialization function. Prepares the dof handlers of q and nedelec type.
   * 
   */
  void initialize_dof_handlers_and_fe();

  /**
   * @brief Updates the numbers of dofs for an edge.
   * 
   * @param cell Cell we are operating on
   * @param edge index of the edge in the cell
   * @param in_dof_counts Dof counts to be updated
   */
  void update_dof_counts_for_edge(CellIterator2D cell, unsigned int edge, DofCountsStruct& in_dof_counts);

  /**
   * @brief Updates the numbers of dofs for a face.
   * 
   * @param cell Cell we are operating on
   * @param in_dof_counts Dof counts to be updated
   */
  void update_dof_counts_for_face(CellIterator2D cell, DofCountsStruct& in_dof_counts);

  /**
   * @brief Updates the dof counts for a vertex
   * 
   * @param cell Cell we are operating on.
   * @param edge Index of the edge in the cell.
   * @param vertex Index of the vertex in the edge.
   * @param in_dof_coutns Dof counts to be updated
   */
  void update_dof_counts_for_vertex(CellIterator2D cell, unsigned int edge, unsigned int vertex, DofCountsStruct& in_dof_coutns);

  /**
   * @brief When building the datastructures, this function adds a new dof to the list of all vertex dofs.
   * 
   * This is always a HSIE dof that relates to an infinite edge and therefore only needs the q type dof_handler in the surface fem.
   * 
   * @param cell The cell the dof was found in.
   * @param edge The index of the edge it belongs to.
   * @param vertex The index of the vertex in the edge that the dof belongs to.
   */
  void register_new_vertex_dofs(CellIterator2D cell, unsigned int edge, unsigned int vertex);

  /**
   * @brief When building the datastructures, this function adds a new dof to the list of all edge dofs.
   * 
   * 
   * @param cell The cell the dof was found in, in the nedelec dof handler
   * @param cell_2 The cell the dof was found in, in the q dof handler
   * @param edge The index of the edge it belongs to.
   */
  void register_new_edge_dofs(CellIterator2D cell, CellIterator2D cell_2, unsigned int edge);

  /**
   * @brief When building the datastructures, this function adds a new dof to the list of all face dofs.
   * 
   * Cells here are faces because the surface triangulation is 2D.
   * 
   * @param cell The cell the dof was found in, in the nedelec dof handler
   * @param cell_2 The cell the dof was found in, in the q dof handler
   * @param edge The index of the edge it belongs to.
   */
  void register_new_surface_dofs(CellIterator2D cell, CellIterator2D cell2);

  /**
   * @brief Increments the dof counter
   * 
   * @return DofNumber returns the dof counter after the increment.
   */
  auto register_dof() -> DofNumber;

  /**
   * @brief Registers a new dof with a face base structure (first argument is string)
   * 
   * There are several lists of the dofs that this object handles. This functions adds a single dof to those lists so it can be iterated over where necessary.
   * 
   * @param in_id The id of the base structures. For cells these have the type string.
   * @param in_hsie_order Order of the hardy space polynomial.
   * @param in_inner_order Order of the nedelec element of the dof.
   * @param in_dof_type There are several different types of dofs. See page 13 in the publication.
   * @param base_dof_index Index if the base dof. For example, an infinite surface dof is a combination of a hardy polynomial in the infinite direction and a surface nedelec edge dof. This number is the dof index of the nedelec edge dof.
   */
  void register_single_dof(std::string in_id, int in_hsie_order, int in_inner_order, DofType in_dof_type, DofDataVector &, unsigned int base_dof_index);

  /**
   * @brief Registers a new dof with a edge or vertex base structure (first argument is int)
   * 
   * There are several lists of the dofs that this object handles. This functions adds a single dof to those lists so it can be iterated over where necessary.
   * 
   * @param in_id The id of the base structures.
   * @param in_hsie_order Order of the hardy space polynomial.
   * @param in_inner_order Order of the nedelec element of the dof.
   * @param in_dof_type There are several different types of dofs. See page 13 in the publication.
   * @param base_dof_index Index if the base dof. For example, an infinite surface dof is a combination of a hardy polynomial in the infinite direction and a surface nedelec edge dof. This number is the dof index of the nedelec edge dof.
   */
  void register_single_dof(unsigned int in_id, int in_hsie_order, int in_inner_order, DofType in_dof_type, DofDataVector &, unsigned int, bool orientation = true);

  /**
   * @brief Evaluates the function a from the publication
   * 
   * See equation 7 in "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems".
   * 
   * @param u Term u in the equation
   * @param v Term v in the equation
   * @param G Term G in the equation
   * @return ComplexNumber Value of a.
   */
  ComplexNumber evaluate_a(std::vector<HSIEPolynomial> &u, std::vector<HSIEPolynomial> &v, dealii::Tensor<2,3,double> G);

  /**
   * @brief All functions for this type assume that x is the infinte direction. This transforms x to the actual infinite direction.
   * 
   * @param in_vector vector of length 3 that defines a field. This will be transformed to the actual coordinate system.
   */
  void transform_coordinates_in_place(std::vector<HSIEPolynomial> * in_vector);

  /**
   * @brief Checks some internal integrity conditions.
   * 
   * @return true Everything is fine.
   * @return false Everythin is not fine.
   */
  bool check_dof_assignment_integrity();

  /**
   * @brief Part of the function above
   * 
   * @return true fine
   * @return false not fine-
   */
  bool check_number_of_dofs_for_cell_integrity();

  /**
   * @brief Get the dof data for a nedelec base dof.
   * 
   * All dofs on this surface are either built based on a nedelec surface dof or a q dof on the surface. For a given index from the nedelec fe this provides all dofs that are based on it.
   * 
   * @param base_dof_index Index of the nedelec dof for whom we search all the dofs that depend on it.
   * @return All the dofs that depend on nedelec dof number base_dof_index.
   */
  auto get_dof_data_for_base_dof_nedelec(DofNumber base_dof_index) -> DofDataVector;

  /**
   * @brief Get the dof data for base dof q 
   * 
   * Same as above but for q dofs.
   * @param base_dof_index See above.
   * @return see above.
   */
  auto get_dof_data_for_base_dof_q(DofNumber base_dof_index) -> DofDataVector;

  /**
   * @brief Get the dof association vector
   * This is a part of the boundary condition interface and returns a list of all the dofs that couple to the inner domain. This is used to prepare the exchange of dof indices and to check integrity (the length of this vector has to be the same as Innerdomain->get_dof_association(boundary id of this boundary)).
   * 
   * @return std::vector<InterfaceDofData> All the dofs that couple to the interior sorted by z, then y then x.
   */
  auto get_dof_association() -> std::vector<InterfaceDofData> override;

  /**
   * @brief Returns the 3D form of a point for a provided 2D position in the surface triangulation.
   * 
   * @return Position in 3D
   */
  auto undo_transform(dealii::Point<2>) -> Position;

  /**
   * @brief Transforms the 2D value of a surface dof shape function into a 3D field in the actual 3D coordinates.
   * The input of this function has 2 components for the two dimensions of the surface triangulation. This gets transformed into the global 3D coordinate system
   * 
   * @return Position value of the shape function interpreted in 3D.
   */
  auto undo_transform_for_shape_function(dealii::Point<2>) -> Position;

  /**
   * @brief If a new dof is active on the surface and should be returned by get_dof_association, this function adds it to the list.
   * 
   * @param in_index_and_orientation Index of the dof and point it should be sorted by.
   */
  void add_surface_relevant_dof(InterfaceDofData in_index_and_orientation);

  /**
   * @brief Get the dof association by boundary id 
   * If two neighboring surfaces have HSIE on them, this can be used to compute on each surface which dofs are at the outside surface they share and the resulting data can be used to build the coupling terms.
   * 
   * @param in_boundary_id the other boundary.
   * @return std::vector<InterfaceDofData> 
   */
  auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<InterfaceDofData> override;

  /**
   * @brief We sometimes use deal.II user flags when iterating over the triangulation. This resets them.
   * 
   */
  void clear_user_flags();

  /**
   * @brief It is usefull to know, if a neighboring surface is also using hsie.
   * Updates the local cache with the information that the neighboring boundary index uses hsie or does not
   * 
   * @param int index
   * @param does if this is true, the neighbor uses hsie, if not, then not.
   */
  void set_b_id_uses_hsie(unsigned int index, bool does);

  /**
   * @brief computes the face angeling data. 
   * 
   * Face angeling data describes if the dofs here are exactly orthogonal to the surface or if they are somehow at an angle.
   * 
   * @param cell The cell to compute the data for
   * @return FaceAngelingData 
   */
  auto build_fad_for_cell(CellIterator2D cell) -> FaceAngelingData;

  /**
   * @brief This computes the coordinate ranges of the surface mesh vertices and caches the result.
   * 
   */
  void compute_extreme_vertex_coordinates();

  /**
   * @brief Computes all vertex positions for a set of vertex ids.
   * 
   * @param ids The list of ids.
   * @return std::vector<Position> with the positions in same order 
   */
  auto vertex_positions_for_ids(std::vector<unsigned int> ids) -> std::vector<Position>;

  /**
   * @brief Computes the positions for line ids.
   * 
   * @param ids The list of ids.
   * @return std::vector<Position> with the positions in same order 
   */
  auto line_positions_for_ids(std::vector<unsigned int> ids) -> std::vector<Position>;

  /**
   * @brief Does nothing. Fulfills the interface.
   * 
   * @return std::string filename
   */
  std::string output_results(const dealii::Vector<ComplexNumber> & , std::string) override;

  /**
   * @brief Computes the number of locally owned dofs. 
   * 
   * For the meaning of owned, check the dealii glossary for a definition.
   * 
   * @return DofCount Number of locally owned dofs.
   */
  DofCount compute_n_locally_owned_dofs() override;

  /**
   * @brief Compute the number of locally active dofs.
   * 
   * For the meaning of active, check the dealii glossary for a definition.
   * 
   * @return DofCount 
   */
  DofCount compute_n_locally_active_dofs() override;

  /**
   * @brief This is a DofDomain via BoundaryCondition. This function signifies that global dof inidices have been exchanged.
   * 
   */
  void finish_dof_index_initialization() override;

  /**
   * @brief Marks for every dof if it is locally owned or not.
   * This fulfills the DofDomain interface. 
   * 
   */
  void determine_non_owned_dofs() override;

  /**
   * @brief Returns an IndexSet with all dofs that are not locally owned.
   * All dofs that are not locally owned must retrieve their global index from somewhere else (usually the inner domain) since the owner gives the number.
   * This function helps prepare that step.
   * 
   * @return dealii::IndexSet All the dofs that are not locally owned in a deal.II::IndexSet
   */
  dealii::IndexSet compute_non_owned_dofs();

  /**
   * @brief Finishes the DofDomainInitialization. 
   * 
   * For each dof that is locally owned, this function sets the global index. They have a local order and the global order and indices are the same, shifted by the number of the first dof.
   * Lets see this domain has for dofs. Three are locally owned, Number 1,2 and 4 and 3 is not locally owned and already has the global index 55. If this function is called with the number 10, the global dof indices will be 10,11,55,12.
   * @param first_own_index 
   * @return true if all indices now have an index
   * @return false some indices (non locally owned) dont have an index yet.
   */
  bool finish_initialization(DofNumber first_own_index) override;
};



