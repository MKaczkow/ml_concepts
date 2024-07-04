import csv
import os
import shutil
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from torch.nn.functional import normalize
from torchvision import datasets, transforms


def prepare_mnist_images(selected_classes, save_dir):
    transform = transforms.Compose([transforms.ToTensor()])

    def save_images(dataset, split_dir):

        # Remove split_dir if it exists
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)

        # Create class directories if not exist
        class_dirs = {
            cls: os.path.join(split_dir, str(cls)) for cls in selected_classes
        }
        for class_dir in class_dirs.values():
            os.makedirs(class_dir, exist_ok=True)

        # Iterate over dataset and save images
        unique_ids = {cls: 0 for cls in selected_classes}
        for img, label in dataset:
            if label in selected_classes:
                img = transforms.ToPILImage()(img)  # Convert tensor to PIL image
                file_name = f"{label}_{unique_ids[label]}.jpg"
                img.save(os.path.join(class_dirs[label], file_name))
                unique_ids[label] += 1

    train_dataset = datasets.MNIST(
        root=".", train=True, transform=transform, download=True
    )
    save_images(train_dataset, os.path.join(save_dir, "train"))

    test_dataset = datasets.MNIST(
        root=".", train=False, transform=transform, download=True
    )
    save_images(test_dataset, os.path.join(save_dir, "test"))


def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, _, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames


def generate_embeddings_and_fnames_simclr(
    model, dataloader
) -> Tuple[torch.Tensor, List[str]]:
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, _, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings.to("cpu"))
    return embeddings, filenames


def get_image_as_np_array(filename: str) -> np.ndarray:
    """Returns an image as an numpy array"""
    img = Image.open(filename)
    return np.asarray(img)


def plot_knn_clusters(
    embeddings: np.ndarray = None,
    original_labels: np.ndarray = None,
    n_neighbors: int = 5,
    num_examples: int = 10,
) -> None:
    """Plots multiple rows of random images with their nearest neighbors"""
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = KNeighborsClassifier(n_neighbors=n_neighbors).fit(
        embeddings, original_labels
    )
    indices = nbrs.kneighbors(embeddings, return_distance=False)
    print(f"Indices shape: {indices.shape}")
    sampled_indices = np.random.choice(len(indices), num_examples, replace=False)
    print(f"Sampled indices: {sampled_indices}")

    for i in sampled_indices:
        print(f"Original label: {original_labels[i]}")
        print(f"Predicted label: {original_labels[indices[i]]}")

    return None


def get_distance_between_points_in_cluster(
    embeddings: np.ndarray = None, labels: np.ndarray = None
) -> dict:
    """Calculates the mean distance between points in the same cluster."""

    distances = {}
    for label in np.unique(labels):
        mask = labels == label
        cluster_points = embeddings[mask]
        distance = cdist(cluster_points, cluster_points).mean()
        distances[str(label)] = distance

    return distances


def generate_embeddings_and_fnames_and_fnames(
    model, dataloader
) -> Tuple[torch.Tensor, List[str]]:
    """Generates representations for all images in the dataloader with
    the given model and 'matches' them with respective filenames
    """

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, _, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames


def plot_knn_examples(
    embeddings, filenames, path_to_data, n_neighbors=3, num_examples=6
) -> None:
    """Plots multiple rows of random images with their nearest neighbors"""
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # get 5 random samples
    samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)

    # loop through our randomly picked samples
    for idx in samples_idx:
        fig = plt.figure()
        # loop through their nearest neighbors
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            # add the subplot
            ax = fig.add_subplot(1, len(indices[idx]), plot_x_offset + 1)
            # get the correponding filename for the current index
            fname = os.path.join(path_to_data, filenames[neighbor_idx])
            # plot the image
            plt.imshow(get_image_as_np_array(fname))
            # set the title to the distance of the neighbor
            ax.set_title(f"d={distances[idx][plot_x_offset]:.3f}")
            # let's disable the axis
            plt.axis("off")

    return None


def separate_images_by_class(csv_file, source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    with open(csv_file, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_name = row["image"] + ".jpg"
            label = row["label"]
            source_path = os.path.join(source_folder, image_name)

            # Create the subfolder for the label if it doesn't exist
            label_folder = os.path.join(destination_folder, label)
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)

            destination_path = os.path.join(label_folder, image_name)

            # Copy the image to the respective label subfolder
            if os.path.exists(source_path):
                shutil.copy2(source_path, destination_path)
            else:
                print(f"Image {source_path} not found.")
    return None


def get_distances_between_centroids(
    embeddings: np.ndarray = None,
    n_clusters: int = 10,
    num_principal_components: int = 3,
) -> np.ndarray:
    """Calculate the clusters and distances between their centroids.

    Args:
        embeddings (np.ndarray, optional): Input embeddings. Defaults to None.
        n_clusters (int, optional): Should be 10 for MNIST, but generally is data-dependent. Defaults to 10.

    Returns:
        np.ndarray: matrix of distances between centroids, with rank n_clusters x n_clusters
    """

    pca = PCA(n_components=num_principal_components)
    embeddings_reduced = pca.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(embeddings_reduced)
    centroids = kmeans.cluster_centers_

    print(f"Cluster centroids dimensions: {centroids.shape}")
    print(f"Cluster centroids:\n {centroids}")
    print(f"Cluster sizes: {np.bincount(labels)}")

    distances = cdist(centroids, centroids)
    return distances


def plot_clusters(
    embeddings: Union[np.ndarray, None] = None,
    original_labels: Union[np.ndarray, None] = None,
    n_clusters: int = 10,
    proportion_of_points_to_plot: float = 0.001,
    alpha: float = 0.1,
    plot_centroids: bool = False,
    specific_labels: Union[list, None] = None,
) -> None:
    """Plots multiple rows of random images with their cluster centroids"""

    print(f"Working on embeddings of shape {embeddings.shape}")

    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    # Filter points by specific_labels if provided
    if specific_labels is not None:
        mask = np.isin(original_labels, specific_labels)
        embeddings = embeddings[mask]
        original_labels = original_labels[mask]
        labels = labels[mask]
        print(
            f"Filtering to labels {specific_labels}: {embeddings.shape[0]} points left"
        )

    # Sample proportion_of_points_to_plot of the data to plot for readability
    sampled_indices = np.random.choice(
        embeddings.shape[0],
        int(len(embeddings) * proportion_of_points_to_plot),
        replace=False,
    )

    sampled_embeddings = embeddings[sampled_indices]
    sampled_labels = original_labels[sampled_indices]
    sampled_cluster_labels = labels[sampled_indices]

    print(f"Plotting {len(sampled_embeddings)} points out of {len(embeddings)}")
    print(f"Some labels: {sampled_labels[:10]}")

    pca = PCA(n_components=2)
    to_plot_embeddings = pca.fit_transform(sampled_embeddings)
    to_plot_centroids = pca.transform(centroids)

    plt.scatter(
        to_plot_embeddings[:, 0],
        to_plot_embeddings[:, 1],
        c=sampled_labels,
        alpha=alpha,
        cmap="viridis",
    )

    if plot_centroids and specific_labels is not None:
        # Only plot centroids of clusters that have points with the specified labels
        unique_cluster_labels = np.unique(sampled_cluster_labels)
        for cluster_label in unique_cluster_labels:
            plt.scatter(
                to_plot_centroids[cluster_label, 0],
                to_plot_centroids[cluster_label, 1],
                c="red",
                marker="x",
            )

    plt.colorbar(label="Original Class Labels")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(
        f"Cluster Plot {'for labels ' + str(specific_labels) if specific_labels is not None else ''}"
    )
    plt.show()
    return None


def plot_clusters_3d(
    embeddings: Union[np.ndarray, None] = None,
    original_labels: Union[np.ndarray, None] = None,
    n_clusters: int = 10,
    proportion_of_points_to_plot: float = 0.001,
    alpha: float = 0.1,
    plot_centroids: bool = False,
    specific_labels: Union[list, None] = None,
    num_principal_components: int = 3,
) -> None:
    """Plots multiple rows of random images with their cluster centroids"""

    print(f"Working on embeddings of shape {embeddings.shape}")

    pca = PCA(n_components=num_principal_components)
    embeddings_reduced = pca.fit_transform(embeddings)
    # to_plot_centroids = pca.transform(centroids)

    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(embeddings_reduced)
    centroids = kmeans.cluster_centers_

    # Filter points by specific_labels if provided
    if specific_labels is not None:
        mask = np.isin(original_labels, specific_labels)
        embeddings = embeddings[mask]
        original_labels = original_labels[mask]
        labels = labels[mask]
        print(
            f"Filtering to labels {specific_labels}: {embeddings.shape[0]} points left"
        )

    # Sample proportion_of_points_to_plot of the data to plot for readability
    sampled_indices = np.random.choice(
        embeddings.shape[0],
        int(len(embeddings) * proportion_of_points_to_plot),
        replace=False,
    )

    sampled_embeddings = embeddings[sampled_indices]
    sampled_labels = original_labels[sampled_indices]
    sampled_cluster_labels = labels[sampled_indices]

    print(f"Plotting {len(sampled_embeddings)} points out of {len(embeddings)}")
    print(f"Some labels: {sampled_labels[:10]}")

    # pca = PCA(n_components=3)
    # to_plot_embeddings = pca.fit_transform(sampled_embeddings)
    # to_plot_centroids = pca.transform(centroids)
    to_plot_embeddings = sampled_embeddings
    to_plot_centroids = centroids

    ax = plt.figure().add_subplot(projection="3d")

    ax.scatter(
        to_plot_embeddings[:, 0],
        to_plot_embeddings[:, 1],
        to_plot_embeddings[:, 2],
        c=sampled_labels,
        alpha=alpha,
        cmap="viridis",
    )

    if plot_centroids and specific_labels is not None:
        # Only plot centroids of clusters that have points with the specified labels
        unique_cluster_labels = np.unique(sampled_cluster_labels)
        for cluster_label in unique_cluster_labels:
            ax.scatter(
                to_plot_centroids[cluster_label, 0],
                to_plot_centroids[cluster_label, 1],
                c="red",
                marker="x",
            )

    # ax.colorbar(label="Original Class Labels")
    # ax.xlabel("Component 1")
    # ax.ylabel("Component 2")
    plt.title(
        f"Cluster Plot {'for labels ' + str(specific_labels) if specific_labels is not None else ''}"
    )
    plt.show()
    return None


def check_labels_correspondence(
    embeddings: np.ndarray = None,
    n_clusters: int = 10,
    num_examples: int = 7,
    base_path: str = None,
    filenames: List[str] = None,
) -> None:

    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(embeddings)
    unique_labels = set(labels)
    print(f"Unique labels: {unique_labels}")

    for label in unique_labels:
        # print(f"Label {label} has {np.sum(labels == label)} samples")
        label_indices = np.where(labels == label)[0]
        sampled_indices = np.random.choice(
            label_indices, size=num_examples, replace=False
        )
        # print(f"Length of sampled_indices: {len(sampled_indices)}")
        # print(f"Sampled indices: {sampled_indices}")

        fig = plt.figure()

        for subplot_idx, idx in enumerate(sampled_indices):
            ax = fig.add_subplot(len(unique_labels), num_examples, subplot_idx + 1)
            fname = os.path.join(base_path, filenames[idx])
            plt.imshow(get_image_as_np_array(fname))
            ax.set_title(f"Label {label}")
            plt.axis("off")

    return None
