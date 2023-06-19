import numpy as np
import faiss
from scipy.spatial.distance import cdist


def build_reference_table(emb_dict, labels_to_ignore=[]):
    """ Builds querry for each class separately, so we can proceed with tests on each class independently

    :return: 2 arrays of embeddings and labels
    """
    # init list of queries and list of quaries labels
    qs = []
    qs_labels = []
    names = []
    for label, embeddings in emb_dict.items():
        if label in labels_to_ignore:
            continue
        embs, img_names = embeddings
        labels = [label] * len(embs)
        embs = [np.array(emb).astype(np.float32) for emb in embs]
        qs.append(np.array(embs))
        qs_labels.append(np.array(labels))
        names.append(np.array(img_names))
    return np.concatenate(qs), np.concatenate(qs_labels), np.concatenate(names)


def build_centroids_table(emb_dict, labels_to_ignore=[]):
    """ Builds querry for each class separately, so we can proceed with tests on each class independently

    :return: 2 arrays of embeddings and labels
    """
    # init list of queries and list of quaries labels
    qs = []
    qs_labels = []
    names = []
    for label, embeddings in emb_dict.items():
        if label in labels_to_ignore:
            continue
        embs, img_names = embeddings
        label = label
        centroid = np.mean(np.array(embs), axis=0)
        sample_wise_dists_to_centroid = cdist(np.array(embs), centroid.reshape(1, centroid.shape[0]))
        cluster_frontal_image = img_names[np.argmin(sample_wise_dists_to_centroid)]
        qs.append(centroid)
        qs_labels.append(label)
        names.append(cluster_frontal_image)

    # introduced sort in order to simplify the process
    sort_order = np.argsort(qs_labels)
    embeddings = np.array(qs).astype(np.float32)
    embeddings = embeddings[sort_order, :]
    qs_labels = np.array(qs_labels)
    qs_labels = qs_labels[sort_order]
    # names = np.concatenate(names) has no sense with centroids
    names = np.array(names)[sort_order]
    return embeddings, qs_labels, names


# adopted code from pytorch_metric_learning library
def get_knn(reference_embeddings, test_embeddings, k, embeddings_come_from_same_source=True, use_gpu=True):
    """
    Finds the k elements in reference_embeddings that are closest to each
    element of test_embeddings.
    Args:
        reference_embeddings: numpy array of size (num_samples, dimensionality).
        test_embeddings: numpy array of size (num_samples2, dimensionality).
        k: int, number of nearest neighbors to find
        embeddings_come_from_same_source: if True, then the nearest neighbor of
                                         each element (which is actually itself)
                                         will be ignored.
        use_gpu: bool, set it to False if you have an error with CUDA
    Returns:
        numpy array: indices of nearest k neighbors
        numpy array: corresponding distances
    """
    d = reference_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    if use_gpu:
        if faiss.get_num_gpus() > 0:
            index = faiss.index_cpu_to_all_gpus(index)
    index.add(reference_embeddings)
    distances, indices = index.search(test_embeddings, k + 1)
    if embeddings_come_from_same_source:
        return indices[:, 1:], distances[:, 1:]
    return indices[:, :k], distances[:, :k]


def compute_classes(emb_library, qs_labels_library, images, query, knn=1, use_gpu=True, same_src=False):
    indices, distances = get_knn(emb_library, query, k=knn, use_gpu=use_gpu, embeddings_come_from_same_source=same_src)

    # find classes and
    if knn == 1:
        predicted_classes_knn = [qs_labels_library[x].item() for x in indices]
        distances = [round(d.item(), 5) for d in distances]
        image_names = [images[x].item() for x in indices]
    else:
        predicted_classes_knn = [[qs_labels_library[y].item() for y in x] for x in indices]
        distances = [[round(d.item(), 5) for d in x] for x in distances]
        image_names = [[images[y].item() for y in x] for x in indices]

    # always assure output to be in the format of lists
    if not isinstance(predicted_classes_knn, list):
        predicted_classes_knn = [predicted_classes_knn]
    if not isinstance(distances, list):
        distances = [distances]
    if not isinstance(image_names, list):
        image_names = [image_names]
    return predicted_classes_knn, distances, image_names
