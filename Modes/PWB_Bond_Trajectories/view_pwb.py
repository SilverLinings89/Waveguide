# This code demonstrates how to load trajectory and cross section information
# from an exported Photonic Wire Bond (PWB) hdf5 file.
#
# It will visualize the bond in 3D and show how to derive some more important
# parameters from it.
#
# This code requires Python 3.6 with the usual scientific python stack.
# If you do not have python, I recommend you start with the latest Anaconda Python
# version and manually install VTK and mayavi into it for 3D plotting.
#
# Matthias Blaicher, 22.7.2017

import h5py
import numpy as np
import mayavi.mlab as mlab

def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= np.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def calculate_numerical_derivative(path_coords):
    # Fifth order tangential approximation as published in section 6.1 of
    # Wang, Wenping, et al. "Computation of rotation minimizing frames."
    #  ACM Transactions on Graphics (TOG) 27.1 (2008): 2.
    path_coords_d1 = np.zeros_like(path_coords)

    path_coords_d1[0, :] = -25 * path_coords[0] + 48 * path_coords[1] \
                           - 36 * path_coords[2] + 16 * path_coords[3] - 3 * path_coords[4]
    path_coords_d1[1, :] = -3 * path_coords[0] - 10 * path_coords[1] + 18 * path_coords[2] \
                           - 6 * path_coords[3] + path_coords[4]
    path_coords_d1[2:-2, :] = path_coords[:-4] - 8 * path_coords[1:-3] + 8 * path_coords[3:-1] - path_coords[4:]
    path_coords_d1[-2, :] = 3 * path_coords[-1] + 10 * path_coords[-2] - 18 * path_coords[-3] \
                            + 6 * path_coords[-4] - path_coords[-5]
    path_coords_d1[-1, :] = 25 * path_coords[-1] - 48 * path_coords[-2] \
                            + 36 * path_coords[-3] - 16 * path_coords[-4] + 3 * path_coords[-5]
    return path_coords_d1


def main(filename):
    with h5py.File(filename, 'r') as f:

        # Read and plot the PWB mesh
        mlab.triangular_mesh(*[f['pwb/mesh/vertices'][:, i] for i in range(3)],
                              f['pwb/mesh/faces'],
                             opacity=0.2)


        # Plot the curve
        mlab.plot3d(*[f['pwb/trajectory/coordinates'][:, i] for i in range(3)],
                    tube_radius=None, representation='wireframe')


        # Plot the coordinate system at every 10th cross section.
        # Note, that the "u" and "v" do NOT coincide the the Frenet-Serret frame!
        mlab.quiver3d(*[f['pwb/trajectory/coordinates'][::2, i] for i in range(3)],
                      *[f['pwb/trajectory/cs/u'][::2, i] for i in range(3)])


        # Calculate the direction of curvature ("normal" vector) based on
        # the change of the tangential vector numerically.
        # Note the Gram–Schmidt process used here.
        t_vector = np.array(f['pwb/trajectory/cs/tangential'])

        # The derivatives are mostly not known analytically at this point anymore
        # So recalculate them numerically.
        d1_vector = calculate_numerical_derivative(f['pwb/trajectory/coordinates'])
        d2_vector = calculate_numerical_derivative(d1_vector)

        gs_dot = np.array([a.dot(b) for a, b in zip(d2_vector, t_vector)])[:, None]
        normal_vector = unit_vector(d2_vector - gs_dot * t_vector, axis=1)


        curvature = (np.apply_along_axis(np.linalg.norm, 1, np.cross(d1_vector, d2_vector))
                     / np.apply_along_axis(np.linalg.norm, 1, d1_vector) ** 3)

        # Plot direction and strength of curvature
        mlab.quiver3d(*[f['pwb/trajectory/coordinates'][::2, i] for i in range(3)],
                      *[(normal_vector*curvature[:, None])[::2, i] for i in range(3)])


        # We also are provided with polygonial cross sections
        # let's show some of them.
        for i, (pos, u, v, t) in enumerate(zip(f['pwb/trajectory/coordinates'],
                                               f['pwb/trajectory/cs/u'],
                                               f['pwb/trajectory/cs/v'],
                                               f['pwb/trajectory/cs/tangential'])):
            # It is not guaranteed that every position has a slice, so we need to check
            try:
                polygon = f[f'pwb/trajectory/crosssection/pos_{i}']
            except KeyError:
                continue

            # Transform polygon to location and plot
            transformed_polygon = polygon[:, 0] * u[:, None] + polygon[:, 1] * v[:, None] + pos[:, None]

            if not i % 10:
                mlab.plot3d(*transformed_polygon,
                            tube_radius=None, representation='wireframe')

    mlab.show()

if __name__ == '__main__':
    main('exported_bond_flat2.hdf5')

