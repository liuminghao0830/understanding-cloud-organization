import pandas as pd
from mlcomp.contrib.split import stratified_group_k_fold

n_fold_split = 5

train_df = pd.read_csv('../data/train.csv')
train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
train_df['ClassName'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])
train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()
# numerical label
class_names_dict = {'Fish':1, 'Flower':2, 'Gravel':3, 'Sugar':4}
train_df['ClassId'] = train_df['ClassName'].map(class_names_dict)
train_df['ClassId'] = [row.ClassId if row.hasMask else 0 for row in train_df.itertuples()]

train_df['fold'] = stratified_group_k_fold(label='ClassId', group_column='ImageId', 
                                                df=train_df, n_splits=n_fold_split)

from collections import Counter
for fold in range(n_fold_split):
    print('-'*10, f'fold: {fold}', '-'*10)
    df_fold = train_df[train_df['fold']==fold]
    print('Images per class: ', Counter(df_fold['ClassId']))

print("Write train data to 'train_{}fold.csv'.".format(n_fold_split))
train_df.to_csv('train_{}fold.csv'.format(n_fold_split), index=False)