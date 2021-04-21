import argparse
import numpy as np
import plotly
import plotly.figure_factory as ff
from skimage import measure
from knnsearch import knnsearch

parser = argparse.ArgumentParser(description='Generate Surface')
parser.add_argument('--file', type=str, default = "",
                   help='filename', required = True)

def rbfReconstruction(input_point_cloud_filename, epsilon = 1e-5):
    """
    surface reconstruction with an implicit function f(x,y,z) computed
    through RBF interpolation of the input surface points and normals
    input: filename of a point cloud, parameter epsilon
    output: reconstructed mesh
    """

    #load the point cloud
    data = np.loadtxt(input_point_cloud_filename)
    points = data[:,:3]
    normals = data[:,3:]


    # construct a 3D NxNxN grid containing the point cloud
    # each grid point stores the implicit function value
    # set N=16 for quick debugging, use *N=64* for reporting results
    N = 64
    max_dimensions = np.max(points,axis=0) # largest x, largest y, largest z coordinates among all surface points
    min_dimensions = np.min(points,axis=0) # smallest x, smallest y, smallest z coordinates among all surface points
    bounding_box_dimensions = max_dimensions - min_dimensions # compute the bounding box dimensions of the point cloud
    grid_spacing = max(bounding_box_dimensions)/(N-9) # each cell in the grid will have the same size
    X, Y, Z =np.meshgrid(list(np.arange(min_dimensions[0]-grid_spacing*4, max_dimensions[0]+grid_spacing*4, grid_spacing)),
                         list(np.arange(min_dimensions[1] - grid_spacing * 4, max_dimensions[1] + grid_spacing * 4,
                                    grid_spacing)),
                         list(np.arange(min_dimensions[2] - grid_spacing * 4, max_dimensions[2] + grid_spacing * 4,
                                    grid_spacing)))

    IF = np.zeros(shape=X.shape) #this is your implicit function - fill it with correct values!
    # toy implicit function of a sphere - replace this code with the correct
    # implicit function based on your input point cloud!!!
    IF = (X - (max_dimensions[0] + min_dimensions[0]) / 2) ** 2 + \
         (Y - (max_dimensions[1] + min_dimensions[1]) / 2) ** 2 + \
         (Z - (max_dimensions[2] + min_dimensions[2]) / 2) ** 2 - \
         (max(bounding_box_dimensions) / 4) ** 2

    ''' ============================================
    #            YOUR CODE GOES HERE
    ============================================ '''
    eps = epsilon
    # Create "Y" array 
    y_solve = np.array([0]*len(points)+[eps]*len(points)+[-1*eps]*len(points))
    # Create Pts matrix as required
    append_mat = np.concatenate(((points+eps*normals),(points-eps*normals)),axis=0)
    append_pts_all = np.concatenate((points,append_mat),axis=0)
    p = append_pts_all
    # Steps to create basis matrix
    basis_li = []
    for i in range(len(p)):
        val_col = [j*j*np.log(j+0.0000001) for j in np.linalg.norm(p-p[i],axis=1)]
        basis_li.append(val_col)
    # Create the basis matrix that needs to be solved
    basis_matrix = np.array(basis_li)
    # Compute the weights matrix using linalg.solve solver 
    weights = np.linalg.solve(basis_matrix,y_solve)

    Q = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()
    IF_new = np.array([0.]*len(Q))
    
    # Steps to compute IF 
    for i in range(len(Q)):
        lin_norm = np.linalg.norm(Q[i]-p,axis=1)
        rb = np.square(lin_norm)*np.log(lin_norm)
        IF_new[i] = np.dot(rb,weights) 

    IF = IF_new.reshape(X.shape)
    ''' ============================================
    #              END OF YOUR CODE
    ============================================ '''

    verts, simplices = measure.marching_cubes_classic(IF, 0)
    
    x, y, z = zip(*verts)
    colormap = ['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)']
    fig = ff.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            colormap=colormap,
                            simplices=simplices,
                            title="Isosurface")
    plotly.offline.plot(fig)

if __name__ == '__main__':
    args = parser.parse_args()
    rbfReconstruction(args.file)
