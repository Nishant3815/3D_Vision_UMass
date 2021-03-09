from __future__ import division
import os
import numpy as np
import torch
from torch.autograd import Variable
from data_utils import grayscale_img_load, listdir

# Added a softmax layer to get predictions as scores represent logits
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def testMVImageClassifier(dataset_path, model, info, pooling = 'mean', cuda=False, verbose=False):

    # save pytorch model to eval mode
    model.eval()
    if (cuda):
        model.cuda()
    
    test_err = 0
    count = 0
    print("=>Testing...")

    # for each category
    for idx, c in enumerate(info['category_names']):
        category_full_dir = os.path.join(dataset_path,c)
        shape_dirs        = listdir(category_full_dir)
        print('=>Loading shape data: %s'%(c))

        # for each shape
        for s in shape_dirs:
            if verbose: print('=>Loading shape data: %s %s'%(s, c))
            views = listdir(os.path.join(category_full_dir, s))
            scores = np.zeros((len(views),len(info['category_names'])))
            count  += 1

            # for each view
            for i, v in enumerate(views):
                image_full_filename = os.path.join(category_full_dir, s, v)
                if 'png' not in image_full_filename : continue
                if verbose: print(' => Loading image: %s ...'%image_full_filename)
                im  = grayscale_img_load(image_full_filename)/255.
                im -= info['data_mean']
                im  = Variable(torch.from_numpy(im.astype('float32')), requires_grad=False).unsqueeze(0)
                # get predicted scores for each view
                if (cuda):
                    im = im.cuda()
                    score_unnormalized = model(im)
                    score_normalized = torch.nn.functional.softmax(score_unnormalized, dim=1)
                    scores[i,:] = score_normalized.detach().cpu().numpy().squeeze()
                else:
                    score_unnormalized = model(im)
                    score_normalized = torch.nn.functional.softmax(score_unnormalized, dim=1)
                    scores[i, :] = score_normalized.detach().numpy().squeeze()
             

            ''' 
            YOUR CODE GOES HERE
            1) Get category predictions per shape and test error averaged over all the test shapes.
            2) Implement 2 strategies: 1) mean and 2) max view-pooling by specifying input arg 'pooling', like
               >> pooling = 'mean' or pooling = 'max'
            
             '''
#             print("Shape of scores array:",scores.shape)
            if (pooling=='mean'):
                predicted_label  = np.argmax(np.mean(scores,axis=0))
            else:
                predicted_label  = np.argmax(np.max(scores,axis=0)) 
            if predicted_label != idx:
                test_err += 1

            if verbose: print('predicted label:  %s, ground-truth label: %s\n'%(info['category_names'][predicted_label] ,c))

    test_err = test_err / count
    print('Test error: %f%%\n'%(test_err * 100))