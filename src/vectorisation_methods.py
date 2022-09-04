# import packages
import numpy as np
from data_utils.pd_pm_methods import clean_pds
from gtda.diagrams import PersistenceLandscape
from gtda.diagrams import BettiCurve
from gtda.diagrams import PersistenceImage


def get_persistence_landscapes(point_clouds: np.ndarray, persistence_diagrams: np.ndarray, n_layers: int, n_bins: int):
    """Function returning persistence landscapes generated from persistence diagrams input.
    Returns np.ndarray."""
    # clean persistence_diagrams
    persistence_diagrams = clean_pds(persistence_diagrams)
    # get number of homology dimensions computed in persistence diagrams
    q = int(persistence_diagrams[0][-1][-1] + 1)
    # Create array to store persistence landscapes
    X = np.zeros((point_clouds.shape[0], n_layers * n_bins * q))
    # Initialise persistence landscape
    pl = PersistenceLandscape(n_layers=n_layers, n_bins=n_bins)
    # Iterate through persistence diagrams to create persistence landscapes
    for i, diag in enumerate(persistence_diagrams):
        diag_expanded = np.expand_dims(diag, axis=0)
        vec_rep = pl.fit_transform(diag_expanded)
        vec_rep_transformed = np.squeeze(vec_rep).reshape(1, vec_rep.shape[1] * vec_rep.shape[2])
        X[i, :] = vec_rep_transformed
    return X


# Function returning Betti curves of persistence diagrams
def get_betti_curves(point_clouds: np.ndarray, persistence_diagrams: np.ndarray, n_bins: int):
    """Function returning Betti Curves generated from persistence diagrams input.
    Returns np.ndarray."""
    # clean persistence_diagrams
    persistence_diagrams = clean_pds(persistence_diagrams)
    # get number of homology dimensions computed in persistence diagrams
    q = int(persistence_diagrams[0][-1][-1] + 1)
    # Create array to store betti curves of persistence diagrams
    Y = np.zeros((point_clouds.shape[0], n_bins * q))
    # Initialise BettiCurve
    bc = BettiCurve()
    # Iterate through persistence diagrams to create betti curvees
    for i, diag in enumerate(persistence_diagrams):
        diag_expanded = np.expand_dims(diag, axis=0)
        curve = bc.fit_transform(diag_expanded)
        curve_transformed = np.squeeze(curve).reshape(1, curve.shape[1] * curve.shape[2])
        Y[i, :] = curve_transformed
    return Y


# Function returning persistence images of persistence diagrams
def get_persistence_images(point_clouds: np.ndarray, persistence_diagrams: np.ndarray, n_bins: int):
    """Function returning persistence images generated from persistence diagrams input.
    Returns np.ndarray."""
    # clean persistence_diagrams
    persistence_diagrams = clean_pds(persistence_diagrams)
    # get number of homology dimensions computed in persistence diagrams
    q = int(persistence_diagrams[0][-1][-1] + 1)
    # create array to store persistence image vectors
    Z = np.zeros((point_clouds.shape[0], q * n_bins * n_bins))
    # initialise persistence image
    pi = PersistenceImage(n_bins=n_bins)
    # iterate through persistence diagrams to create persistence images
    for i, diag in enumerate(persistence_diagrams):
        diag_expanded = np.expand_dims(diag, axis=0)
        img = pi.fit_transform(diag_expanded)
        img_transformed = np.squeeze(img).reshape(1, (img.shape[1] * img.shape[2] * img.shape[3]))
        Z[i, :] = img_transformed
    return Z
