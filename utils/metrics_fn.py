import torch
import torchmetrics.functional as FM
from sklearn.metrics import balanced_accuracy_score, precision_recall_curve, auc, root_mean_squared_error


def metrics_fn(preds, targets, metrics, num_classes=2, task='multiclass', average='macro'):
    metrics_dict = dict()
    if 'bac' in metrics:
        metrics_dict['bac'] = torch.tensor(balanced_accuracy_score(targets.argmax(dim=1).cpu(), preds.argmax(dim=1).cpu()), device=preds.device)
    if 'aucpr' in metrics:
        precision, recall, thresholds = precision_recall_curve(targets.argmax(dim=1).cpu(), preds.argmax(dim=1).cpu(), pos_label=1)
        metrics_dict['aucpr'] = torch.tensor(auc(recall, precision), device=preds.device)
    if 'auroc' in metrics:
        metrics_dict['auroc'] = FM.auroc(preds, targets.argmax(dim=1), num_classes=num_classes, task=task,  average=average)
    if 'acc' in metrics:
        metrics_dict['acc'] = FM.accuracy(preds, targets.argmax(dim=1), num_classes=num_classes, task=task,  average=average)
    if 'f1' in metrics:
        metrics_dict['f1'] = FM.f1_score(preds, targets.argmax(dim=1), num_classes=num_classes, task=task,  average='weighted')
    if 'kappa' in metrics:
        metrics_dict['kappa'] = FM.cohen_kappa(preds, targets.argmax(dim=1), num_classes=num_classes, task=task)
    if 'mae' in metrics:
        metrics_dict['mae'] = FM.mean_absolute_error(preds, targets)
    if 'rmse' in metrics:
        metrics_dict['rmse'] = torch.tensor(root_mean_squared_error(targets.cpu().detach().numpy(), preds.to(targets.dtype).cpu().detach().numpy()), device=preds.device)
    if 'r2' in metrics:
        metrics_dict['r2'] = FM.r2_score(preds, targets)
    if 'mape' in metrics:
        metrics_dict['mape'] = FM.mean_absolute_percentage_error(preds, targets)

    return metrics_dict
