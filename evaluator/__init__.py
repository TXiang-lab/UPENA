from .weight_evaluate import Evaluator


def make_evaluator(cfg, dataset_name=None):
    if cfg.skip_eval:
        return None
    return Evaluator(cfg, dataset_name=dataset_name)
