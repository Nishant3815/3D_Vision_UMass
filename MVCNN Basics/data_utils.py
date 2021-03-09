from __future__ import division
import torch
import os
import pickle as p
import numpy as np
import gzip
import time
from PIL import Image

def save_checkpoint(model, epoch):
    """save model checkpoint"""
    model_out_path = "model/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))

def predictions(ytrue, ypred):
    """Returns top1 and top5 predictions for provided true and predicted label set"""
    pred1   = np.argmax(ypred,axis=1)
    top5    = np.argpartition(-ypred, 5, axis=1)[:,:5]
    ypred5  = [1 if label in top5[i] else 0 for i, label in enumerate(ytrue)]
    return (np.mean(ytrue==pred1), np.mean(ypred5))

def grayscale_img_load(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = np.array(img.convert('L'))
        return np.expand_dims(img, axis = 0) #specific format for pytorch. Expects channel as first dimension

def load_data():
    '''load saved gzip files'''
    files = [f for f in listdir(os.getcwd()) if ".npy" in f]
    if len(files)!=0:
        info = p.load( open( "info.p", "rb" ) )
        data = {}
        t = time.time()
        for item in files:
            print('=> Loading '+item)
            f = gzip.GzipFile(item, "r")
            data[item.replace('.npy.gz','')] = np.load(f)
        print("Time Taken %.3f sec"%(time.time()-t))
        return data, info

    else:
        print("Couldn't open file. Run save_data!")
        return [], []

def listdir(path):
    ''' ignore any hidden fiels while loading files in directory'''
    return [f for f in os.listdir(path) if not f.startswith('.')]

def save_data(dataset_path, verbose = False):
    """
    Load all PNG images and their category information in numpy matrix.

    :param dataset_path: is the name of the folder that contains PNG images of
                            rendered 3D shapes. The folder should have the following structure:
                            => category_1 [folder]
                                => shape1_id [folder]
                                    => shape1_id_001.png [grayscale image]
                                    => shape1_id_002.png [grayscale image]
                                        ...
                                => shape2_id [folder]
                                    => shape2_id_001.png [grayscale image]
                                    => shape2_id_002.png [grayscale image]
                                        ...
                                => ...
                            => category_2 [folder]
    :param verbose:
    :return:
    """
    
    category_names = listdir(dataset_path)
    # get stats on the training data
    num_shapes = 0
    num_images = 0
    info = {}
    info['category_names'] = listdir( dataset_path ) # will return folders corresponding to categories, including the default directories '.' and '..'
    info['num_views'] = 0
    num_categories = len(info['category_names'])
    shape_names    = {}

    for c in range(num_categories):
        category_full_dir = os.path.join(dataset_path, info['category_names'][c])
        shape_dirs = listdir(category_full_dir )
        shape_names[c] = shape_dirs
        num_shapes = num_shapes + len(shape_names[c])
        
        for s in range(len(shape_dirs)):
            num_views = len(listdir( os.path.join(dataset_path, info['category_names'][c], shape_names[c][s])))
            info['num_views'] = max(num_views, info['num_views'])
            num_images += num_views

    print('Found %d categories, %d rendered shapes, %d total images, %d (max) num views per shape\n'%(num_categories, num_shapes, num_images, info['num_views']))

    #prepare the training image database
    data = {'X' : [], 'Y': [], 'train' : [], 'val':[]}
    image_id = 0
    shape_id = 0
    for c in range(num_categories):
        print('=> Loading category ' + info['category_names'][c])
        for s in range(len(shape_names[c])):
            
            shape_id += 1
            if verbose : print('Loading shape data %d/%d: %s\\%s \n'%(shape_id, num_shapes, info['category_names'][c], shape_names[c][s]))
            curr_dir = os.path.join(dataset_path, info['category_names'][c], shape_names[c][s])              
            views = listdir(curr_dir)
            
            for v in views: #we assume that images of shapes have filenames in this format: shape1_id_001.png, shape1_id_002.png ...
                image_full_filename = os.path.join(curr_dir, v)
                
                if verbose : print(' => Loading image: %s \n'%(image_full_filename))
                im = grayscale_img_load(image_full_filename)/255.
                data['X'].append(im.astype('float32'))
                data['Y'].append(c)

                if s < 0.9*len(shape_names[c]): #use 90% for training, 10% for validation
                    data['train'].append(image_id)
                else:
                    data['val'].append(image_id)
                image_id +=  1
    
    data['X'] = np.array(data['X'])
    data['Y'] = np.array(data['Y'])
    
    data_mean         = np.mean(data['X'], axis = 0) #computes image mean
    info['data_mean'] = data_mean
    data['X']         = data['X'] - data_mean #subtracts image mean

    '''save as gzip file for better data compression'''
    
    t = time.time()
    for item in data:
        print("=> Saving "+ item)
        f = gzip.GzipFile(item+'.npy.gz', "w")
        np.save(f, data[item])
        f.close()
    
    print("Time Taken %.3f sec"%(time.time()-t))

    p.dump(info, open( "info.p", "wb" ) )

    return