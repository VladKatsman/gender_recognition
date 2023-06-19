import numpy as np
import torch
from pytorch_metric_learning import testers
from pytorch_metric_learning.utils import common_functions as c_f


class CustomTester(testers.BaseTester):

    def get_all_embeddings(self, dataloader, trunk_model):
        embeddings, labels = self.compute_all_embeddings(dataloader, trunk_model)
        embeddings = self.maybe_normalize(torch.from_numpy(embeddings))
        return embeddings, labels

    def compute_all_embeddings(self, dataloader, trunk_model):
        s, e = 0, 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                img, label = self.data_and_label_getter(data)
                label = c_f.process_label(label, "all", self.label_mapper)
                q = self.get_embeddings_for_eval(trunk_model, img)
                if label.dim() == 1:
                    label = label.unsqueeze(1)
                if i == 0:
                    if dataloader.drop_last:
                        labels = torch.zeros(len(dataloader) * dataloader.batch_size, label.size(1))
                        all_q = torch.zeros(len(dataloader) * dataloader.batch_size, q.size(1))
                    else:
                        labels = torch.zeros(dataloader.dataset.__len__(), label.size(1))
                        all_q = torch.zeros(dataloader.dataset.__len__(), q.size(1))
                e = s + q.size(0)
                all_q[s:e] = q
                labels[s:e] = label
                s = e
            labels = labels.cpu().numpy()
            all_q = all_q.cpu().numpy()

        return all_q, labels

    def get_embeddings_for_eval(self, trunk_model, input_imgs):
        trunk_output = trunk_model(input_imgs.cuda())
        return trunk_output


def get_all_embeddings(dataloader, model, normalize=False):
    tester = CustomTester(normalize_embeddings=normalize, dataloader_num_workers=dataloader.num_workers)
    return tester.get_all_embeddings(dataloader, model)


def compute_metrics(dataloader, model, normalize=False, clip=False):
    embeddings, labels                          = get_all_embeddings(dataloader, model, normalize)
    labels                                      = np.squeeze(labels).astype(int)
    distance                                    = torch.cdist(embeddings, embeddings)
    distance.fill_diagonal_(1.)                  # remove item itself from KNN search
    pred_ids                                    = distance.argmin(axis=0)
    pred                                        = labels[pred_ids]

    precision                                   = compute_precision(labels, pred)
    precision_per_class, mean_average_precision = compute_per_class_precision(labels, pred)

    return precision, precision_per_class, mean_average_precision


def compute_precision(labels:np.ndarray, pred:np.ndarray):
    """How many data points were predicted correctly """
    return sum(pred == labels)/len(labels)


def compute_per_class_precision(labels:np.ndarray, pred:np.ndarray):
    # compute per_class accuracy
    unique_labels                   = np.unique(labels)
    comparison_labels               = unique_labels[:, None] == labels
    comparison_pred                 = unique_labels[:, None] == pred
    true_pred_per_class             = np.sum(comparison_pred & comparison_labels, axis=1)
    precision_per_class             = true_pred_per_class/np.sum(comparison_labels, axis=1)

    # how many classes we learned (good for analytics purposes)
    mean_average_precision           = np.mean(precision_per_class)
    return precision_per_class, mean_average_precision
