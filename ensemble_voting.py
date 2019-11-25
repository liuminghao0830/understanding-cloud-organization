import cv2, glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import post_process, mask2rle

sub_df = pd.read_csv('../data/sample_submission.csv')
sub_df['ImageId'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])
test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])

threshold = 0.45
#min_size = [15000, 15000, 15000, 15000]
min_size = [20000, 20000, 22500, 10000]

# Average ensemble
print("Reading npy files")
npy_path = '*/*_{}.npy'

encoded_pixels = []

for i in tqdm(range(len(test_imgs))):
    # Batch prediction result set
    files = glob.glob(npy_path.format(i))
    # K-fold predictions
    predictions = np.zeros((len(files), 320, 480, 4))
    for p in range(len(files)):
        pred = np.load(files[p])
        for k in range(pred.shape[-1]):
            predictions[p,:,:,k] = cv2.threshold(pred[:,:,k], 
                                       threshold, 1, cv2.THRESH_BINARY)[1]
    pred_masks = np.sum(predictions, axis=0)
    
    for k in range(pred_masks.shape[-1]):
        pred_mask = pred_masks[...,k].astype('float32') 
        
        if pred_mask.shape != (350, 525):
            pred_mask = cv2.resize(pred_mask, dsize=(525, 350), 
                                    interpolation=cv2.INTER_LINEAR)
            
        pred_mask, num_predict = post_process(pred_mask, len(files) / 2.0, 
                                              min_size[k], (350, 525))
        
        if num_predict == 0:
            encoded_pixels.append('')
        else:
            r = mask2rle(pred_mask)
            encoded_pixels.append(r)

# # Submission
sub_df['EncodedPixels'] = encoded_pixels
sub_df.to_csv('efnb4_unet_5fold_submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)