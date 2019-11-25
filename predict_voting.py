import keras, cv2, glob
import pandas as pd
from tta_wrapper import tta_segmentation
import segmentation_models as sm
from data import DataGenerator
from utils import post_process, mask2rle

sub_df = pd.read_csv('../data/sample_submission.csv')
sub_df['ImageId'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])
test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])

# # Predict on Test Set
model1 = []
m1_path = "efnb4_unet/efficientnetb4unet_*fold.h5"
for m in glob.glob(m1_path):
    model = sm.Unet('efficientnetb4', classes=4, input_shape=(320, 480, 3), activation='sigmoid')
    model.load_weights(m)
    mddel = tta_segmentation(model, h_flip=True, h_shift=(-10, 10), merge='mean')
    model1.append(model)

model2 = []
m2_path = "resnet18_unet/resnet18unet_*fold.h5"
for m in glob.glob(m2_path):
    model = sm.Unet('resnet18', classes=4, input_shape=(320, 480, 3), activation='sigmoid')
    model.load_weights(m)
    mddel = tta_segmentation(model, h_flip=True, h_shift=(-10, 10), merge='mean')
    model2.append(model)

model3 = []
m3_path = "efnb7_unet/efficientnetb7unet_*fold.h5"
for m in glob.glob(m3_path):
    model = sm.Unet('efficientnetb7', classes=4, input_shape=(320, 480, 3), activation='sigmoid')
    model.load_weights(m)
    mddel = tta_segmentation(model, h_flip=True, h_shift=(-10, 10), merge='mean')
    model3.append(model)


threshold = 0.45
min_size = [20000, 20000, 22500, 10000]

encoded_pixels = []
BATCH_SIZE = 1

test_generator = DataGenerator(test_imgs.index, df=test_imgs, shuffle=False, mode='predict',
                                   dim=(350, 525), reshape=(320, 480), n_channels=3, 
                                   gamma=0.8, base_path='../data/test_images', 
                                   target_df=sub_df, batch_size=BATCH_SIZE, n_classes=4)

ensemble_pred_masks = []
for i, x_batch in enumerate(test_generator):
    if i % 100 == 0:
        print("Predicting {}th batch".format(i))
    
    batch_idx = list(range(i*BATCH_SIZE, min(test_imgs.shape[0], i*BATCH_SIZE + BATCH_SIZE)))
    
    batch_pred_masks1 = np.zeros((BATCH_SIZE, 320, 480, 4), dtype=np.float32)
    for m in model1:
        batch_pred_masks1 += m.predict(x_batch)
    ensemble_pred_masks.append(batch_pred_masks1 / len(model1))

    batch_pred_masks2 = np.zeros((BATCH_SIZE, 320, 480, 4), dtype=np.float32)
    for m in model2:
        batch_pred_masks2 += m.predict(x_batch)
    ensemble_pred_masks.append(batch_pred_masks2 / len(model2))

    batch_pred_masks3 = np.zeros((BATCH_SIZE, 320, 480, 4), dtype=np.float32)
    for m in model3:
        batch_pred_masks3 += m.predict(x_batch)
    ensemble_pred_masks.append(batch_pred_masks3 / len(model3))

    for j, idx in enumerate(batch_idx):
        filename = test_imgs['ImageId'].iloc[idx]
        image_df = sub_df[sub_df['ImageId'] == filename].copy()
        
        voting_masks = np.zeros((350, 525, 4))
        for batch_pred_masks in ensemble_pred_masks:
            # Batch prediction result set
            pred_masks = batch_pred_masks[j, ]
            
            for k in range(pred_masks.shape[-1]):
                pred_mask = pred_masks[...,k].astype('float32')
                
                if pred_mask.shape != (350, 525):
                    pred_mask = cv2.resize(pred_mask, dsize=(525, 350), 
                                            interpolation=cv2.INTER_LINEAR)
                
                pred_mask = cv2.threshold(pred_mask, threshold, 1, cv2.THRESH_BINARY)[1]
                
                voting_masks[:,:, k] += pred_mask

            for k in range(pred_masks.shape[-1]):
                pred_mask, num_predict = post_process(voting_masks[:,:, k], 1.9, 
                                              			min_size[k], (350, 525))

                if num_predict == 0:
                    encoded_pixels.append('')
                else:
                    r = mask2rle(pred_mask)
                    encoded_pixels.append(r)

# # Submission
sub_df['EncodedPixels'] = encoded_pixels
sub_df.to_csv('submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)