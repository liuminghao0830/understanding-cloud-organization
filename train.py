#!/usr/bin/env python
import cv2, keras
from keras import backend as K
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import segmentation_models as sm
import tensorflow as tf
from mlcomp.contrib.split import stratified_group_k_fold

from utils import *
from loss import *


# # Preprocessing
train_df = pd.read_csv('../data/train.csv')
train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
train_df['ClassName'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])
train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()
# numerical label
class_names_dict = {'Fish':1, 'Flower':2, 'Gravel':3, 'Sugar':4}
train_df['ClassId'] = train_df['ClassName'].map(class_names_dict)
train_df['ClassId'] = [row.ClassId if row.hasMask else 0 for row in train_df.itertuples()]

train_df['fold'] = stratified_group_k_fold(label='ClassId', group_column='ImageId', 
                                                          df=train_df, n_splits=5)

# # Training
MODEL = 'resnet18'
BATCH_SIZE = 16
EPOCHS = 30


#Five fold training
for fold in range(5):
    print("Training on fold--" + str(fold))
    train = train_df[train_df['fold'] != fold]
    valid = train_df[train_df['fold'] == fold]

    train_uniq_df = train.groupby('ImageId').agg({'hasMask':'sum'}).reset_index()
    train_uniq_df.sort_values('hasMask', ascending=False, inplace=True)

    valid_uniq_df = valid.groupby('ImageId').agg({'hasMask':'sum'}).reset_index()
    valid_uniq_df.sort_values('hasMask', ascending=False, inplace=True)

    train_generator = DataGenerator(train_uniq_df.index, df=train_uniq_df, target_df=train,
                                batch_size=BATCH_SIZE, reshape=(320, 480), gamma=0.8,
                                augment=True, n_channels=3, n_classes=4)

    val_generator = DataGenerator(valid_uniq_df.index, df=valid_uniq_df,  target_df=valid,
                              batch_size=BATCH_SIZE, reshape=(320, 480), gamma=0.8,
                              augment=False, n_channels=3, n_classes=4)

    opt = AdamAccumulate(lr=0.002, accum_iters=8)

    model = sm.Unet(MODEL, classes=4, input_shape=(320, 480, 3), activation='sigmoid')

    model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[dice_coef])
    #model.summary()

    checkpoint = ModelCheckpoint(MODEL + 'unet_{}fold.h5'.format(fold), save_best_only=True)
    es = EarlyStopping(monitor='val_dice_coef', min_delta=0.0001, patience=5, 
                        verbose=1, mode='max', restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.2, patience=2, 
                        verbose=1, mode='max', min_delta=0.0001)

    history = model.fit_generator(train_generator, validation_data=val_generator,
                              callbacks=[checkpoint, rlr, es], epochs=EPOCHS, verbose=1)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(MODEL + 'history_{}fold.csv'.format(fold), index=False)