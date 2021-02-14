from __future__ import division
import numpy as np

def computeShapeHistogram(mesh, y_min, y_max, number_of_bins):

    """
    Complete this function to compute a histogram capturing the 
    distribution of surface point locations along the upright 
    axis (y-axis) in the given range [y_min, y_max] for a mesh. 
    The histogram should be normalized i.e., represent a 
    discrete probability distribution.
    
    input: 
    mesh structure from 'loadMesh' function
    => mesh.V contains the 3D locations of V mesh vertices (Vx3 matrix)

    output: 
    shape histogram (a row vector)
    """    

    """""""""""""""""""""""""""""""""
    ADD CODE HERE TO COMPUTE mesh.H
    """""""""""""""""""""""""""""""""
    bin_space = np.linspace(y_min,y_max,number_of_bins+1)
    sum_all_pts = sum(np.histogram(mesh,bins=bin_space,density=False)[0].tolist())
    # Note setting density==True would assume a CDF and divide by (# points in a bin * bin_length) and we don't use that 
    histogram = [i/sum_all_pts for i in np.histogram(mesh,bins=bin_space,density=False)[0].tolist()]
    return histogram