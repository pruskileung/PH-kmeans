# import packages
import numpy as np
import gudhi as gd
from gph.python.ripser_interface import _ideal_thresh
from scipy.spatial import distance_matrix
import math
import ot


def clean_pds(pds: np.ndarray):
    """Function to clean persistence diagrams generated by giotta-tda VietorisRipsPersistence. Triples with same
    birth-death values that were originally computed for padding reasons are removed before further computations. """
    cleaned_pds = []
    for i, d in enumerate(pds):
        cleaned_pds.append(d[d[:, 0] != d[:, 1]])
    return pds


def normalise_pc(point_cloud):
    max = np.max(point_cloud[:])
    min = np.min(point_cloud[:])
    range = max - min
    return ((point_cloud - min) / range - 0.5) * 2


def get_pd(point_cloud):
    """
    Function returns list of persistence subdiagrams each representing 0, 1 or 2
    dimensional homological features. Combining the list of subdiagrams gives the
    entire persistence diagram.
    """
    max_thresh = _ideal_thresh(distance_matrix(point_cloud, point_cloud), np.inf)
    skeleton = gd.RipsComplex(points=point_cloud, max_edge_length=max_thresh)
    simplex_tree = skeleton.create_simplex_tree(max_dimension=3)
    barcode = simplex_tree.persistence()

    diag_0 = simplex_tree.persistence_intervals_in_dimension(0)
    diag_1 = simplex_tree.persistence_intervals_in_dimension(1)
    diag_2 = simplex_tree.persistence_intervals_in_dimension(2)

    for diag in [diag_0, diag_1, diag_2]:
        diag[np.isinf(diag)] = max_thresh

    return [diag_0, diag_1, diag_2]


# Persistence Measure Methods - Adapted from https://anonymous.4open.science/r/PD-subsample-2321/

# initialise parameters
nb_units = 30
float_error = 1e-8
mat_size = int(nb_units * (nb_units + 1) / 2) + 1


def get_grid_width(diagrams):
    """
    Function takes PDs and calculates the correct grid width for PM calculation
    """
    max_width = 0
    for diag in diagrams:
        concat_diag = np.concatenate(diag)
        diag_width = math.ceil(concat_diag.max())
        if diag_width > max_width:
            max_width = diag_width
    return max_width


def mesh_gen(grid_width):
    """
    generate mesh on the triangular area.
    """
    grid = []
    unit = grid_width/nb_units
    for i in range(nb_units):
        for j in range(i+1, nb_units+1):
            grid.append([unit*i, unit*j])
    return grid


def dist_mat(grid, power_index):
    """
    construct the distance matrix of the grid points.
    The underlying distance is L_infty.
    power_index specifies the type of Wasserstein distance.
    """
    M = np.zeros([mat_size, mat_size])
    for i in range(mat_size-1):
        for j in range(mat_size-1):
            M[i,j] = max([abs(grid[i][0]-grid[j][0]), \
                          abs(grid[i][1]-grid[j][1])])
    # append the diagnal
    for k in range(mat_size-1):
        M[k,mat_size-1] = (grid[k][1]-grid[k][0])/2
        M[mat_size-1,k] = (grid[k][1]-grid[k][0])/2
    Mp = np.power(M,power_index)
    return Mp


def diag_to_mesr(diag, unit_mass, grid_width):
    mesr = np.zeros(mat_size) + float_error
    mesr_vis = np.zeros([nb_units, nb_units])

    unit = grid_width / nb_units

    for point in diag:
        i = math.floor(point[0] / unit)
        j = math.floor(point[1] / unit)
        mesr_vis[i, j] += unit_mass
        mesr[nb_units * i + j - int(i * (i + 1) / 2)] += unit_mass
    return mesr, mesr_vis


def wass_dist(a, b, Mp):
    """
    compute the p-Wasserstein distance between two measures.
    note: take pth root to get a real distance
    edit: error in sinkhorn distance
    """
    a_mesr = a.tolist()
    b_mesr = b.tolist()
    a_ms_all = sum(a_mesr)
    b_ms_all = sum(b_mesr)
    a_mesr[mat_size-1] = b_ms_all
    b_mesr[mat_size-1] = a_ms_all
    ot_dist = ot.sinkhorn2(a_mesr, b_mesr, Mp, 1e-3)
    return ot_dist


def get_mean_mesr(measures, float_error):
    """
    Function takes PMs and returns the mean.
    """
    adjusted_measures = []
    for m in measures:
        m_copy = np.copy(m)
        m_copy[m_copy == float_error] = 0
        adjusted_measures.append(m_copy)

    mean_mesr = sum(adjusted_measures) / len(adjusted_measures)

    mean_mesr[mean_mesr == 0] = float_error

    return mean_mesr
