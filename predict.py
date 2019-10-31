import keras
import pandas as pd
from tta_wrapper import tta_segmentation

sub_df = pd.read_csv('../data/sample_submission.csv')
sub_df['ImageId'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])
test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])

# # Predict on Test Set
model.load_weights('model.h5')

model = tta_segmentation(model, h_flip=True, h_shift=(-10, 10), merge='mean')

best_threshold = 0.5
best_size = 15000

threshold = best_threshold
min_size = best_size

test_df = []
encoded_pixels = []
TEST_BATCH_SIZE = 500

for i in range(0, test_imgs.shape[0], TEST_BATCH_SIZE):
    batch_idx = list(
        range(i, min(test_imgs.shape[0], i + TEST_BATCH_SIZE))
    )

    test_generator = DataGenerator(
        batch_idx,
        df=test_imgs,
        shuffle=False,
        mode='predict',
        dim=(350, 525),
        reshape=(320, 480),
        n_channels=3,
        gamma=0.8,
        base_path='../data/test_images',
        target_df=sub_df,
        batch_size=1,
        n_classes=4)

    batch_pred_masks = model.predict_generator(
        test_generator, 
        workers=1,
        verbose=1
    ) 
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
                pred_mask = cv2.resize(pred_mask, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                
            pred_mask, num_predict = post_process(pred_mask, threshold, min_size)
            
            if num_predict == 0:
                encoded_pixels.append('')
            else:
                r = mask2rle(pred_mask)
                encoded_pixels.append(r)

# # Submission
sub_df['EncodedPixels'] = encoded_pixels
sub_df.to_csv('submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)