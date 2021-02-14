from __future__ import division
from Mesh import *
import pickle as p
import numpy as np
from computeShapeHistogram import computeShapeHistogram
from utils import save_data, load_data


def forward_prop(w,b,X,true_lab,reg_lambda):
    """
    Forward propagation with gradient computations
    """
    Z = np.matmul(X,w.T)+b
    m     = X.shape[0]
    # Apply sigmoid over weights
    A = 1/(1+np.exp(-Z))
    # Apply binary-crossentropy loss
    cost_bin_crossentropy = -1*((np.matmul(true_lab,np.log(A)))+(np.matmul(1-true_lab,np.log(1-A))))
    cost_ridge        = (reg_lambda/(2*m))*np.sum(np.square(w)) # L2-Regularization
    cost              = cost_bin_crossentropy  + cost_ridge
    # Gradient Descent related code ahead
    # Note "1/m" i.e. number of samples parameter removed from regularization based on slide and tuning value in assignment, ideally it should be there 
    # If you want to run without regularization, simply put reg_lambda equal to zero in params instead of 0.1
    dw    = (np.matmul(X.T,(A-true_lab.T)).T)+((reg_lambda)*w) # partial derivative of loss wrt w: x*(a-y) 
    # Shape of dw should be same as w
    db    = np.mean((A - true_lab.T))
    # Squeeze cost function
    cost  = np.squeeze(cost)
    return cost, dw, db

def run_optimization(w,b,X,true_lab,num_iter=100,step_rate=0.1,reg_lambda=0.1):
    """
    Update step for gradient descent optimization goes here and append all the costs at each run
    """
    costs = []
    for i in range(num_iter):
        # Calculate cost and gradients
        cost,dw,db = forward_prop(w,b,X,true_lab,reg_lambda)
        # Update the weights and biases using the gradients 
        w = w - step_rate*dw
        b = b - step_rate*db
        # Append all the cost related calculations
        costs.append(cost)
    return costs,w,b,dw,db


def logisticRegressionTrain(train_dir,num_iter=100,step_rate=0.1,reg_lambda=0.1, number_of_bins=10, loadData=False):
    """
    Complete this function to train a logistic regression classifier

    input:
    train_dir is the path to a directory containing meshes
    in OBJ format used for training. The directory must
    also contain a ground_truth_labels.txt file that
    contains the training labels (category id) for each mesh
    number_of_bins specifies the number of bins to use
    for your histogram-based mesh descriptor

    output:
    a row vector storing the learned classifier parameters (you must compute it)
    histogram range (this is computed for you)

    if you want to avoid reloading meshes each time you run the code,
    change loadData to True in argument. The code automatically saves the data it loads in the first iteration
    """

    if loadData:
        meshes, min_y, max_y, N, shape_labels = load_data('tmp_train.p')
    
    else:
        #OPEN ground_truth_labels.txt
        shape_filenames = []
        shape_labels = []
        
        try:
            with open(os.path.join(train_dir,'ground_truth_labels.txt'),'rU') as ground_truth_labels_file:
                for line in ground_truth_labels_file:
                    name, label = line.split()
                    shape_filenames.append(name)
                    shape_labels.append(int(label))
        
        except IOError as e:
            print("Couldn't open file (%s)." % e)
            return

        """
        read the training meshes, compute 'lowest' and 'highest' surface
        point across all meshes, move meshes such that their centroid is
        at (0, 0, 0), scale meshes such that average vertical distance to
        mesh centroid is one.
        """

        meshes = [] #A cell array storing all meshes
        N      = len(shape_filenames) #number of training meshes
        min_y  = np.float('inf') #smallest y-axis position in dataset
        max_y  = np.float('-inf') #largest y-axis position in dataset

        for n in range(N):
            meshes.append(Mesh(train_dir, shape_filenames[n], number_of_bins))
            number_of_mesh_vertices = meshes[n].V.shape[0] #number of mesh vertices
            mesh_centroid           = np.mean(meshes[n].V, axis=0, keepdims = True)
            meshes[n].V             = meshes[n].V -  mesh_centroid #center mesh at origin
            average_distance_to_centroid_along_y = np.mean(np.abs(meshes[n].V[:,1])) #average vertical distance to centroid
            meshes[n].V             = meshes[n].V/average_distance_to_centroid_along_y #scale meshes
            min_y                   = min(min_y, min(meshes[n].V[:,1]))
            max_y                   = max(max_y, max(meshes[n].V[:,1]))
            print(shape_filenames[n] + " Processed")

        save_data(meshes, min_y, max_y, N, shape_labels, 'tmp_train.p')

    """
    this loop calls your histogram computation code!
    all shape descriptors are organized into a NxD matrix
    N training meshes, D dimensions (number of bins in your histogram)
    """
    
    X = np.zeros((N, number_of_bins))
    for n in range(N):
        # Modified a bit from the base code to adjust for ComputeShapeHistogram function
        X[n,:] = computeShapeHistogram(meshes[n].V[:,1], min_y, max_y, number_of_bins )
    
    # Initialize weights and biases
    np.random.seed(3815)
    w = .5 * np.random.randn(1, number_of_bins)
    b = .5 * np.random.randn(1,1)
    # Get true labels
    true_lab = np.reshape(np.asarray(shape_labels),(1,len(shape_labels)))
    """""""""""""""""""""""""""""""""""""""
     ADD CODE HERE TO LEARN PARAMETERS w
    """""""""""""""""""""""""""""""""""""""
    # Note that run_optimization function calls forward prop within it
    costs,w,b,dw,db = run_optimization(w,b,X,true_lab,num_iter,step_rate,reg_lambda)
    w = np.append(w,b)
    return w, min_y, max_y


