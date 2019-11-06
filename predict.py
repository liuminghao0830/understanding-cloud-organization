import keras, cv2
import pandas as pd
from tta_wrapper import tta_segmentation
import segmentation_models as sm
from data import DataGenerator
from utils import post_process, mask2rle

sub_df = pd.read_csv('../data/sample_submission.csv')
sub_df['ImageId'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])
test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])

# # Predict on Test Set
model1 = sm.Unet('efficientnetb4', classes=4, input_shape=(320, 480, 3), activation='sigmoid')
model1.load_weights('../efn_unet.h5')
model2 = sm.Unet('resnet18', classes=4, input_shape=(320, 480, 3), activation='sigmoid')
model2.load_weights('../resnet18_unet.h5')
model3 = sm.Unet('densenet121', classes=4, input_shape=(320, 480, 3), activation='sigmoid')
model3.load_weights('../dense_unet.h5')

model1 = tta_segmentation(model1, h_flip=True, h_shift=(-10, 10), merge='mean')
model2 = tta_segmentation(model2, h_flip=True, h_shift=(-10, 10), merge='mean')
model3 = tta_segmentation(model3, h_flip=True, h_shift=(-10, 10), merge='mean')

threshold = 0.45
#min_size = [15000, 15000, 15000, 15000]
min_size = [20000, 20000, 22500, 10000]

test_df = []
encoded_pixels = []
BATCH_SIZE = 1

test_generator = DataGenerator(test_imgs.index, df=test_imgs, shuffle=False, mode='predict',
                                   dim=(350, 525), reshape=(320, 480), n_channels=3, 
                                   gamma=0.8, base_path='../data/test_images', 
                                   target_df=sub_df, batch_size=BATCH_SIZE, n_classes=4)


for i, x_batch in enumerate(test_generator):
    if i % 100 == 0:
        print("Predicting {}th batch".format(i))
    
    batch_idx = list(range(i*BATCH_SIZE, min(test_imgs.shape[0], i*BATCH_SIZE + BATCH_SIZE)))
    
    batch_pred_masks1 = model1.predict(x_batch)
    batch_pred_masks2 = model2.predict(x_batch)
    batch_pred_masks3 = model3.predict(x_batch)
    batch_pred_masks = (batch_pred_masks1+batch_pred_masks2+batch_pred_masks3) / 3
    # Predict out put shape is (320X480X4)
    # 4  = 4 classes, Fish, Flower, Gravel Surger.
    
    for j, idx in enumerate(batch_idx):
        filename = test_imgs['ImageId'].iloc[idx]
        image_df = sub_df[sub_df['ImageId'] == filename].copy()
        
        # Batch prediction result set
        pred_masks = batch_pred_masks[j, ]
        
        for k in range(pred_masks.shape[-1]):
            pred_mask = pred_masks[...,k].astype('float32') 
            
            if pred_mask.shape != (350, 525):
                pred_mask = cv2.resize(pred_mask, dsize=(525, 350), 
                                        interpolation=cv2.INTER_LINEAR)
                
            pred_mask, num_predict = post_process(pred_mask, threshold, 
                                                  min_size[k], (350, 525))
            
            if num_predict == 0:
                encoded_pixels.append('')
            else:
                r = mask2rle(pred_mask)
                encoded_pixels.append(r)

# # Submission
sub_df['EncodedPixels'] = encoded_pixels
sub_df.to_csv('submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)