import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
## metrics
from sksurv.metrics import concordance_index_censored
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')

def log_rank(res_df):  # 2 class version
    median = res_df['risk'].median()
    res_df.loc[res_df['risk']<=median, 'pred'] = 0
    res_df.loc[res_df['risk']>median, 'pred'] = 1
    mask_0 = f'pred == 0'
    mask_1 = f'pred == 1'
    logrank_test_result = logrank_test(
        durations_A=res_df.query(mask_0)['time'],
        durations_B=res_df.query(mask_1)['time'],
        event_observed_A=res_df.query(mask_0)['status'],
        event_observed_B=res_df.query(mask_1)['status'],
    )
    pvalue_pred = logrank_test_result.p_value
    return (pvalue_pred)

def evaluation(model, loader, cfg, gc=16):
    use_multimodal = cfg.Data.use_multimodal
    use_clinical = cfg.Data.use_clinical
    res_df = loader.dataset.df.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    survive_time_all = []
    status_all = []
    pred_each = None
    val_loss = 0
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    model.to(device)
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            x, y, time, c, cancer_encoding = batch['feature'].to(device, dtype=torch.float32), \
                            batch['label'].to(device, dtype=torch.long), \
                            batch['time'].to(device), \
                            batch['status'].to(device), \
                            batch['cancer_encoding'].to(device, dtype=torch.float32)
            if use_clinical:
                feature = batch['clinical_feature'].to(device)
                if use_multimodal:
                    result = model(h=x, clinical_feature=feature, cancer_encoding=cancer_encoding)
                else:
                    result = model(clinical_feature=feature, cancer_encoding=cancer_encoding)
            else:
                result = model(h=x, cancer_encoding=cancer_encoding)                                                                                           
            bag_logits = result['bag_logits']
            pred = bag_logits[0][1:]
            res_df.loc[idx, 'risk'] = pred.cpu().numpy()
            iter_ = idx % gc +1
            survive_time_all.append(np.squeeze(time.cpu().numpy()))
            status_all.append(np.squeeze(c.cpu().numpy()))
            all_risk_scores[idx] = np.squeeze(pred.detach().cpu().numpy())
            all_event_times[idx] = time.item()
            all_censorships[idx] = c.item()
            if idx == 0:
                pred_all = pred
            if iter_ == 1:
                pred_each = pred
            else:
                pred_all = torch.cat([pred_all, pred])
                pred_each = torch.cat([pred_each, pred])
            if iter_%gc==0 or idx == len(loader)-1:
                survive_time_all = np.asarray(survive_time_all)
                status_all = np.asarray(status_all)
                pred_each = None
                survive_time_all = []
                status_all = []
    val_loss /= len(loader)
    if 'patient_id' not in res_df.columns:
        res_df['patient_id'] = res_df.filename.values
        print('no patient id, copy with slide_id')
    max_risk_id = res_df.groupby('patient_id')['risk'].idxmax()
    new_res_df = res_df.iloc[max_risk_id]
    cindex = concordance_index_censored((1-new_res_df.status.values).astype(bool), new_res_df.time.values, new_res_df.risk.values, tied_tol=1e-08)[0]
    new_res_df.status = 1-new_res_df.status

    pvalue = log_rank(new_res_df)
    return new_res_df, cindex, pvalue

def evaluation_multi(model, loaders, cfg, prefix=None):
    dataset_names = []
    res_df_all = []
    cindex_all = []
    pvalue_all = []
    for loader in loaders:
        dataset_name = loader.dataset.df['cancer_type'].values[0]
        res_df, cindex, pvalue = evaluation(model, loader, cfg)
        dataset_names.append(prefix+'_'+dataset_name)
        res_df_all.append(res_df)
        cindex_all.append(cindex)
        pvalue_all.append(pvalue)
    return {'cancer_type':dataset_names, 'res_df':res_df_all, 'cindex':cindex_all, 'pvalue':pvalue_all}