def create_evaluation(cfg):
    if cfg.Train.val_function == 'survival_cox_multi':
        from training_methods.survival_general import evaluation_multi as evaluation
    else:
        raise NotImplementedError

    return evaluation