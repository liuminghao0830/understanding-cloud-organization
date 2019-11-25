import keras
import pandas as pd
import numpy as np
import segmentation_models as sm
from tta_wrapper import tta_segmentation
from sklearn.model_selection import train_test_split

from data import DataGenerator
from utils import post_process

def numpy_dice_coef(y_true, y_pred, smooth=1):
    y_true_f = np.ndarray.flatten(y_true)
    y_pred_f = np.ndarray.flatten(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


train_df = pd.read_csv('../data/train.csv')
train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
mask_count_df = pd.DataFrame(train_df['ImageId'].unique(), columns=['ImageId'])

train_idx, val_idx = train_test_split(mask_count_df.index, 
                          random_state=2019, test_size=0.2)

# # Predict on Test Set
model = sm.Unet('efficientnetb4', classes=4, input_shape=(320, 480, 3), activation='sigmoid')
model.load_weights('../efn_unet.h5')

#model = tta_segmentation(model, h_flip=True, h_shift=(-10, 10), merge='mean')

threshold = [0.45, 0.50, 0.30, 0.35]
min_size = [20000, 20000, 22500, 10000]

BATCH_SIZE = 100

val_generator = DataGenerator(val_idx, df=mask_count_df,  target_df=train_df,
                              batch_size=BATCH_SIZE, reshape=(320, 480), gamma=0.8,
                              augment=False, n_channels=3, n_classes=4)


cnt = 0
dice = 0.
for x, y_true in val_generator:
    batch_pred_masks = model.predict(x, verbose=1) 
    # Predict out put shape is (320X480X4)
    # 4  = 4 classes, Fish, Flower, Gravel Surger.
    cnt += 1

    for j in range(batch_pred_masks.shape[0]):
        # Batch prediction result set
        pred_masks = batch_pred_masks[j, ]
        
        for k in range(pred_masks.shape[-1]):
            pred_mask = pred_masks[...,k].astype('float32')
                
            pred_mask, num_predict = post_process(pred_mask, threshold[k], 
                                                min_size[k], (320, 480))
            batch_pred_masks[j, :,:, k] = pred_mask
    batch_dice = numpy_dice_coef(y_true, batch_pred_masks)
    print("Dice coeff after processing: {}".format(batch_dice))
    dice += batch_dice
print("Total Dice coeff after processing: {}".format(dice / cnt))