import torch

def load_model(cfg):
    if cfg.Model.model_type == 'experiment':
        model = load_experimental_model(cfg)
    elif cfg.Model.model_type == 'stable':
        model = load_stable_model(cfg)
    else:
        raise NotImplementedError

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    return model

def load_experimental_model(cfg):

    raise NotImplementedError

def load_stable_model(cfg):
    if cfg.Model.model_name == 'ProgPath':
        from models.stable_models.ProgPath import ProgPath
        model = ProgPath(n_classes=cfg.Data.n_classes, **cfg.Model)
        print('loading ProgPath model')
    else:
        raise NotImplementedError

    return model
