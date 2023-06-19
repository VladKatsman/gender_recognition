import os
import numpy as np
import timm
import torch
import torch.nn as nn
from tqdm import tqdm
from data.dataset import dataloader
from data.embeddings_ops import build_reference_table, build_centroids_table, compute_classes
from misc.torch_utils import init_device_pytorch
from misc.visualization_utils import open_and_add_text_to_img, combine_images, center_print


class Evaluator:
    def __init__(self,
                 model: nn.Module,
                 weights=None,
                 ref_data='/home/noteme/data/gender_recognition/Training',
                 test_data='/home/noteme/data/gender_recognition/Validation',
                 results_dir='/home/noteme/data/results',
                 batch_size=128,
                 knn_centroids=1,
                 knn_images=3,
                 normalize=True,
                 ):

        self.results_dir = results_dir
        self.batch_size = batch_size
        self.normalize = normalize
        self.knn_centroids = knn_centroids
        self.knn_images = knn_images

        self.device = init_device_pytorch()
        self.model = model
        if weights:
            w = torch.load(weights)
            self.model.load_state_dict(w, strict=False)
        self.model.to(self.device)
        self.model.eval()
        # if 'CLIP' in str(type(self.model)):
        #     self.batch_size = 1
        torch.set_grad_enabled(False)

        self.ref_data_loader = self.build_dataloader(ref_data)
        self.test_data_loader = self.build_dataloader(test_data)

        self.ref_images = self.ref_data_loader.dataset.images
        self.ref_labels = self.ref_data_loader.dataset.labels

        # build emb dict
        center_print("Start building Test embeddings")
        emb_ref = self.compute_embeddings(self.test_data_loader)
        self.test_query, self.q_labels, self.q_names = build_reference_table(emb_ref)

        # create centroid reference table
        center_print("Start building Reference Embeddings")
        embeddings = self.compute_embeddings(self.ref_data_loader)
        self.centroids,  self.c_labels, self.c_names = build_centroids_table(embeddings)
        self.references, self.labels,   self.names = build_reference_table(embeddings)

    def evaluate_batch(self, batch):
        if 'CLIP' in str(type(self.model)):
            result = self.model.encode_image(batch)
        else:
            result = self.model(batch)
            # we have to normalize embeddings before we are going to compare them
        if self.normalize:
            result = nn.functional.normalize(result, p=2, dim=1)
        return result

    # DATA AND EMBEDDINGS RELATED
    def build_dataloader(self, p_to_data):
        return dataloader(p_to_data, batch_size=self.batch_size)

    def compute_embeddings(self, dataloader_object):
        emb_dict = {}
        for idx, (images, labels) in tqdm(enumerate(dataloader_object), total=len(dataloader_object)):
            image_paths = dataloader_object.dataset.path_batches[idx]
            images = images.to(self.device)
            labels = labels.cpu().numpy()
            embeddings = self.evaluate_batch(images).cpu().numpy()
            self._update_emb_dict(emb_dict, labels, embeddings, image_paths)
        return emb_dict

    @staticmethod
    def _update_emb_dict(emb_dict, keys, embs, paths):
        for key, emb, path in zip(keys, embs, paths):
            if key in emb_dict:
                emb_dict[key][0].append(emb)
                emb_dict[key][1].append(path)
            else:
                emb_dict[key] = [emb], [path]

    # CLASSIFICATION RELATED
    def _predict_test(self):
        # predict classes
        self.predicted_classes, self.scores, self.pred_path = self.predict_class_using_closest_nn(self.test_query)
        self.predicted_options, self.scores_op, self.pred_pathes_opt = self.predict_options_using_centeroids(
            self.test_query)

    def predict_class_using_closest_nn(self, query):
        predicted_classes, scores, pathes = compute_classes(self.references, self.labels, self.names,
                                                            knn=self.knn_images, query=query, use_gpu=False,
                                                            same_src=False)
        return predicted_classes, scores, pathes

    def predict_options_using_centeroids(self, query):
        predicted_options, scores_op, pathes = compute_classes(self.centroids, self.c_labels, self.c_names,
                                                               knn=self.knn_centroids, query=query, use_gpu=False,
                                                               same_src=False)
        return predicted_options, scores_op, pathes

    # RESULTS EVALUATION RELATED
    def _compute_metrics(self, verbose=True):
        # compute metrics
        binary_res_k1 = self.get_binary_result(k=1)
        binary_res_kn = self.get_binary_result(k=self.knn_images)
        binary_res_centroid = self.get_binary_result(k=0)

        class_wise_acc_at_k1 = self.get_class_wise_accuracy(binary_res_k1)
        class_wise_acc_at_k5 = self.get_class_wise_accuracy(binary_res_kn)
        class_wise_acc_at_centroid = self.get_class_wise_accuracy(binary_res_centroid)

        mean_class_wise_acc_at_k1 = self.get_mean_class_wise_accuracy(binary_res_k1)
        mean_class_wise_acc_at_k5 = self.get_mean_class_wise_accuracy(binary_res_kn)
        mean_class_wise_acc_at_centroid = self.get_mean_class_wise_accuracy(binary_res_centroid)

        # print results
        if verbose:
            print("BENCHMARK FOR CLASSIFICATION FOR CURRENT TASK FINISHED\n")
            print("Here are results:\n")
            print(f"Class wise accuracy at knn = 1: {class_wise_acc_at_k1}\n")
            print(f"Class wise accuracy at knn = {self.knn_images}: {class_wise_acc_at_k5}\n")
            print(f"Class wise accuracy at centroid: {class_wise_acc_at_centroid}\n")
            print(f"Mean class wise accuracy at knn = 1: {mean_class_wise_acc_at_k1}\n")
            print(f"Mean class wise accuracy at knn = {self.knn_images}: {mean_class_wise_acc_at_k5}\n")
            print(f"Mean class wise accuracy at centroid : {mean_class_wise_acc_at_centroid}\n")
        else:
            return class_wise_acc_at_k1, class_wise_acc_at_k5, \
                   mean_class_wise_acc_at_k1, mean_class_wise_acc_at_k5

    def get_class_wise_accuracy(self, binary_res):
        return {identifier: self.get_identifier_accuracy(identifier, binary_res) for identifier in np.unique(self.q_labels)}

    def get_mean_class_wise_accuracy(self, binary_res):
        return np.mean([self.get_identifier_accuracy(identifier, binary_res) for identifier in np.unique(self.q_labels)])

    def get_binary_result(self, k=1):

        # predict centroid
        if k == 0:
            if self.knn_centroids != 1:
                res = self.q_labels == [x[0] for x in self.predicted_options]
            else:
                res = self.q_labels == self.predicted_options

        # predict top 1
        elif k == 1:
            if self.knn_images != 1:
                res = self.q_labels == [x[0] for x in self.predicted_classes]
            else:
                res = self.q_labels == self.predicted_classes

        # predict top KNN
        else:
            res = [q in self.predicted_classes[i] for i, q in enumerate(self.q_labels)]
        return np.array(res)

    @staticmethod
    def get_mean_accuracy_at_k(binary_res):
        return binary_res.sum() / binary_res.__len__()

    def get_identifier_accuracy(self, identifier, binary_res):
        id_mask = self.q_labels == identifier
        id_binary_res = binary_res[id_mask]
        return self.get_mean_accuracy_at_k(id_binary_res)

    def _analyze_predictions(self, only_errors=True):

        for i in tqdm(range(len(self.q_labels))):

            true_label = self.q_labels[i]
            test_img = self.q_names[i]
            test_img_name = os.path.basename(test_img)

            # wrong prediction check
            if only_errors:
                if self.predicted_classes[i] == true_label:
                    continue

            def put_data_on_knn_imgs():
                images       = self.pred_path[i]
                labels       = self.predicted_classes[i]
                scores       = self.scores[i]  # similarity score
                results      = labels == true_label
                titles       = [f'top {k+1} reference' for k in range(len(images))]
                return [txt_to_img(images[i], labels[i], results[i], scores[i], titles[i])
                        for i in range(len(images))]

            def put_data_on_centroid_imgs():
                image       = self.pred_pathes_opt[i]
                label       = self.predicted_options[i]
                score       = self.scores_op[i]
                result      = label == true_label
                title       = f'Predicted centroid class'
                return [txt_to_img(image, label, result, score, title)]

            def txt_to_img(img, label, res, score, title):
                description = f'Result:{res}\n' f'ID:{label}\n' f'Score:{score:.3f}'
                return open_and_add_text_to_img(img, title, description)

            # true_img txt
            title = 'Query Image'
            description = f'ID:{true_label}'
            test_img = open_and_add_text_to_img(test_img, title, description)

            knn_images      = put_data_on_knn_imgs()
            centroid_images = put_data_on_centroid_imgs()
            grid = combine_images(test_img, knn_images, centroid_images, space=16)
            grid.save(os.path.join(self.results_dir, test_img_name))


if __name__ == '__main__':
    from modelling.models import baseline_backbone, clip_vit16b, baseline_backbone_neck
    import clip

    # weights = '/home/noteme/data/results/logger/baseline_neck/epoch=01-precision=0.9882.pth'
    weights=None

    # model = baseline_backbone()
    model = clip_vit16b()
    # model = baseline_backbone_neck()

    # init class
    cls = Evaluator(model=model, weights=weights)

    # run FAISS on all the test crops
    cls._predict_test()

    # compute and print results, if you want to use metrics (fill excel or whatever) set to False and use return of func
    cls._compute_metrics(verbose=True)

    # visualization. If you want visualize good examples (for presentation e.g.) set parameter to False
    cls._analyze_predictions(only_errors=True)
