import keras, glob
import pandas as pd
import numpy as np
from tta_wrapper import tta_segmentation
import segmentation_models as sm
from data import DataGenerator

sub_df = pd.read_csv('../data/sample_submission.csv')
sub_df['ImageId'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])
test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])

# # Predict on Test Set
model_name = "efficientnetb4"
model_path = "efnb4-unet/efficientnetb4unet_0fold.h5"

model = sm.Unet(model_name, classes=4, input_shape=(320, 480, 3), activation='sigmoid')
model.load_weights(model_path)

model = tta_segmentation(model, h_flip=True, h_shift=(-10, 10), merge='mean')

BATCH_SIZE = 1
test_generator = DataGenerator(test_imgs.index, df=test_imgs, shuffle=False, mode='predict',
                                   dim=(350, 525), reshape=(320, 480), n_channels=3, 
                                   gamma=0.8, base_path='../data/test_images', 
                                   target_df=sub_df, batch_size=BATCH_SIZE, n_classes=4)


for i, x_batch in enumerate(test_generator):
    if i % 100 == 0:
        print("Predicting {}th batch".format(i))
    
    batch_idx = range(i*BATCH_SIZE, min(test_imgs.shape[0], i*BATCH_SIZE + BATCH_SIZE))
    
    batch_pred_masks = model.predict(x_batch)
    for j, idx in enumerate(batch_idx):
        npy_path = model_path.split('.')[0] + '_'+ str(idx) + '.npy'
        print("Write prediction confidence to " + npy_path)
        np.save(npy_path, batch_pred_masks[j, :])