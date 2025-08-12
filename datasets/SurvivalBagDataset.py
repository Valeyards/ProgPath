import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import  OneHotEncoder
import warnings

def process_clinical(df, columns):
    df = df.loc[ : , ~df.columns.str.contains("^Unnamed")].reset_index(drop=True)
    df['age'] = df['age'].fillna(value=df['age'].median())
    if not 'age_year' in df.columns:
        df['age_year'] = df['age'] / 100.
    feature_df = df[['filename', 'sex', 'stage', 'age_year']]
    warnings.filterwarnings("ignore")
    enc = OneHotEncoder(drop='if_binary')
    enc.fit([
            [0,1],
            [1,2],
            [0,3],
            [1,4],])
    ans = enc.transform(feature_df.drop(columns=['filename', 'age_year']).values).toarray()
    col_names = []
    ans_new = []
    if 'sex' in columns:
        col_names.extend(['sex'])
        ans_new.append(ans[:,:1])
    if 'stage' in columns:
        col_names.extend(['stage_1', 'stage_2', 'stage_3', 'stage_4'])
        ans_new.append(ans[:,1:])
    
    ans_new = np.concatenate(ans_new, axis=-1)
    if 'age' in columns:
        col_names.extend(['age_year'])
        ans_new = np.concatenate([ans_new, np.expand_dims(feature_df['age_year'].values, -1)], axis=-1)
        
    res_df = pd.DataFrame(data = ans_new, columns=col_names, dtype=float)
    res_df['filename'] = feature_df['filename']
    res = {'processed_df':res_df, 'original_df':feature_df}
    return res

class SurvivalBagDataset(Dataset):
    def __init__(self, df, data_dir, label_field='status', use_clinical=False, csv_path=None, 
                 columns= ['filename', 'sex', 'stage', 'age_year'], **kwargs):
        super(SurvivalBagDataset, self).__init__()
        self.data_dir = data_dir
        self.label_field = label_field
        # inverse censorship 
        df.status = 1-df.status
        self.df = df
        self.use_clinical = use_clinical
        print('loading test set clinical csvs')
        csv_all = pd.read_csv(csv_path)
        csv_all = csv_all.loc[ : , ~csv_all.columns.str.contains("^Unnamed")]
        csv_all = csv_all.loc[ : , ~csv_all.columns.str.contains("^level")]
        self.clinical_feature = process_clinical(csv_all, columns)['processed_df']
            
    def __len__(self):
        return len(self.df.values)

    def get_data_df(self):
        return self.df

    def get_id_list(self):
        return self.df['filename'].values
    
    def get_balance_weight(self):
        # for data balance
        label = self.df['cancer_type'].values
        label_np = np.array(label)
        classes = list(set(label))
        N = len(self.df)
        num_of_classes = [(label_np==c).sum() for c in classes]
        c_weight = [N/num_of_classes[i] for i in range(len(classes))]

        weight = [0 for _ in range(N)]
        for i in range(N):
            c_index = classes.index(label[i])
            weight[i] = c_weight[c_index]

        return weight

    def encode_cancer_type(self, cancer_type):
        cancer_types = ['blca', 'brca', 'cesc', 'crc', 'gbm', 'hnsc', 'rcc', 'lgg', 'lihc', 'luad', 'lusc', 'paad', 'skcm', 'stad', 'ucec']
        encoding = [0] * len(cancer_types)
        if cancer_type in cancer_types:
            encoding[cancer_types.index(cancer_type)] = 1
        return encoding

    def __getitem__(self, idx):
        label = self.df[self.label_field].values[idx]
        status = self.df['status'].values[idx]
        if type(self.df['filename'])==np.float64:
            self.df['filename'] = self.df['filename'].astype(int)
        slide_id = str(self.df['filename'].values[idx])
        time = self.df['time'].values[idx]
        cancer = self.df['cancer_type'].values[idx]
        cancer_encoding = self.encode_cancer_type(cancer)
        if self.use_clinical:
            clinical_feature_df = self.clinical_feature.loc[self.clinical_feature['filename'].astype(str)==slide_id]
            clinical_feature = clinical_feature_df.drop(columns='filename').values[0]

        # load from pt files
        if 'feature_path' in self.df.columns:
            full_path = self.df['feature_path'].values[idx]
        else:
            if os.path.exists(os.path.join(self.data_dir, 'patch_feature')):
                full_path = os.path.join(self.data_dir, 'patch_feature', slide_id if slide_id.endswith('.pt') else slide_id + '.pt')
            else:
                full_path = os.path.join(self.data_dir, slide_id if slide_id.endswith('.pt') else slide_id + '.pt')
        features = torch.load(full_path, map_location=torch.device('cpu'))

        res = {
            'feature': features,
            'label': torch.tensor([label]),
            'time': torch.tensor(time),
            'status': torch.tensor(status),
            'cancer_encoding': torch.tensor(cancer_encoding)
        }
        if self.use_clinical:
            res.update({'clinical_feature': torch.from_numpy(clinical_feature).float().unsqueeze(0)})
            res.update({'clinical_feature_df': clinical_feature_df})
            
        return res
