from logisticRegressionTrain import logisticRegressionTrain
from logisticRegressionTest import logisticRegressionTest

# Dataset directory
train_dir = '../dataset/train'
test_dir = '../dataset/test'

# Train
w, min_y, max_y = logisticRegressionTrain(train_dir,num_iter=100,step_rate=0.1,reg_lambda=0.1, number_of_bins=10, loadData=True)

# Test
t, test_err,pred_label,true_label  = logisticRegressionTest(test_dir, w, min_y, max_y, True)