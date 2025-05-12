import argparse
import os
import torch
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from utils.utils import read_yaml
from utils.dataloader_factory import create_dataloader
from utils.model_factory import load_model
from utils.training_method_factory import create_evaluation
import warnings
from lifelines.statistics import multivariate_logrank_test
warnings.filterwarnings("ignore")

def logrank_func(df):
    if len(df.pred.unique())==2:
        mask_0 = f'pred == 0'
        mask_1 = f'pred == 1'
        logrank_test_result = logrank_test(
            durations_A=df.query(mask_0)['time'],
            durations_B=df.query(mask_1)['time'],
            event_observed_A=df.query(mask_0)['status'],
            event_observed_B=df.query(mask_1)['status'],
        )
    else:
        logrank_test_result = multivariate_logrank_test(df['time'], df['pred'], df['status'])
    return logrank_test_result.p_value

def draw_kmfig(df, fold, study, result_dir, avg=False, header=''):
    dct = {0:'low risk', 1:'high risk'}
    color_dct = {0:'#67a9cf', 1:'#ef8a62'}
    kmf1 = KaplanMeierFitter()
    kmf2 = KaplanMeierFitter()
    kmfs = [kmf1, kmf2]
    for name, grouped_df in df.groupby('pred'):
        name = int(name)
        kmf = kmfs[name]
        kmf.fit(grouped_df["time"], grouped_df["status"], label=dct[name])
        kmf.plot(ci_show=False, show_censors=True, c=color_dct[name], xlabel='Time', ylabel='Proportion Surviving')
    mask_0 = f'pred == 0'
    mask_1 = f'pred == 1'
    logrank_test_result = logrank_test(
        durations_A=df.query(mask_0)['time'],
        durations_B=df.query(mask_1)['time'],
        event_observed_A=df.query(mask_0)['status'],
        event_observed_B=df.query(mask_1)['status'],
    )
    plt.title("Project:{}  p-value:{:.3e}".format(study, logrank_test_result.p_value))
    plt.tight_layout()
    if avg:
        print(result_dir)
        plt.savefig('{}/km_avg.png'.format(result_dir))
    else:
        plt.savefig('{}/km.png'.format(result_dir))
    plt.close()

def draw_kmfig_multi(df, study, result_dir, avg=False):
    kmf = KaplanMeierFitter()
    dct = {0:'low risk', 1:'high risk'}
    color_dct = {0:'#67a9cf', 1:'#ef8a62'}
    for name, grouped_df in df.groupby('pred'):
        kmf.fit(grouped_df["time"], grouped_df["status"], label=dct[name])
        kmf.plot(ci_show=False, show_censors=True, c=color_dct[name], xlabel='Time', ylabel='Proportion Surviving')
    logrank_test_result = multivariate_logrank_test(df['time'], df['pred'], df['status'])
    plt.title("Project:{}  p-value:{:.3e}".format(study, logrank_test_result.p_value))
    plt.tight_layout()
    if avg:
        print(result_dir)
        plt.savefig('{}/km_avg.png'.format(result_dir))
    else:
        plt.savefig('{}/km.png'.format(result_dir))
    plt.close()

def eval_single(model, dataloader, result_dir, cfg, prefix=None):
    res_dict = evaluation(model, dataloader, cfg, prefix=prefix)
    res_dfs = res_dict['res_df']
    cindexs = res_dict['cindex']
    pvalues = res_dict['pvalue']
    res_dict['cancer_type'] = [dataset['project'] for dataset in cfg.Data.external_datasets]
    for j in range(len(res_dfs)):
        cancer_type = cfg.Data.external_datasets[j]['project']
        result_dir_cancer = os.path.join(result_dir, cancer_type)
        if not os.path.exists(result_dir_cancer):
            os.makedirs(result_dir_cancer)
        res_df = res_dfs[j]
        cindex = cindexs[j]
        pvalue = pvalues[j]
        median = res_df[f'{header}risk'].median()
        res_df.loc[res_df[f'{header}risk']<=median, 'pred'] = 0
        res_df.loc[res_df[f'{header}risk']>median, 'pred'] = 1
        print(cancer_type, 'cindex_now:', cindex, 'pvalue:', pvalue)
        # make km plot
        draw_kmfig_multi(res_df, cancer_type, result_dir_cancer)
        res_df.to_csv(result_dir_cancer+'/predictions.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/task0_sample.yaml')

    args = parser.parse_args()
    import sys
    if sys.platform != 'linux':
        args.config_path = args.config_path.replace('\\', '/')
    cfg = read_yaml(args.config_path)
    header = ''
    if len(cfg.Data.external_datasets)>0 and cfg.Data.external_datasets is not None:
        dataset_name = 'external'
        prefix = 'external'
    else:
        raise NotImplementedError
    model = load_model(cfg)
    evaluation = create_evaluation(cfg)
    ckpt_dir = './weights/'
    model.load_state_dict(
    torch.load(os.path.join(ckpt_dir, f'progpath.pt')))
    result_dir = os.path.join(cfg.General.result_dir, 
                            'evaluation')
    dataloader = create_dataloader(dataset_name, cfg, result_dir)
    eval_single(model, dataloader, result_dir, cfg, prefix=prefix)