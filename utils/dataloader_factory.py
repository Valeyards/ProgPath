from torch.utils.data import DataLoader
import pandas as pd

def create_dataloader(dataset, cfg, result_dir):
    if cfg.Data.dataset_name in ['SurvivalBagDataset']:
        return create_bag_dataloader(dataset, cfg, result_dir)
    else:
        raise NotImplementedError

def create_bag_dataloader(dataset, cfg, result_dir):
    if cfg.Data.dataset_name == 'SurvivalBagDataset':
        from datasets.SurvivalBagDataset import SurvivalBagDataset
        dataloader = []
        for dataset in cfg.Data.external_datasets:
            data_dir = dataset['data_dir']
            csv_path = dataset['csv_dir']
            test_df = pd.read_csv(csv_path, comment='#')
            test_df = test_df.loc[ : , ~test_df.columns.str.contains("^Unnamed")]
            dataset = SurvivalBagDataset(test_df, data_dir=data_dir,
                                csv_path=csv_path, istrain=False, istest=True, use_clinical=cfg.Data.use_clinical, 
                                columns=cfg.Data.columns)
            dataloader.append(DataLoader(dataset, batch_size=None, shuffle=False,
                                num_workers=cfg.Train.num_worker, pin_memory=True))
        print(f'loading {len(dataloader)} external val dataloaders')
    else:
        raise NotImplementedError
    return dataloader
