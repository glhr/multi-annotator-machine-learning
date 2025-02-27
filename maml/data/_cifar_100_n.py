import torch
import numpy as np
import os
import pandas as pd
import requests

from PIL import Image
from sklearn.model_selection import train_test_split, KFold
from torchvision.datasets import CIFAR100
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    Resize,
    CenterCrop,
    RandomErasing,
)
from ._base import MultiAnnotatorDataset, ANNOTATOR_FEATURES, AGGREGATION_METHODS, TRANSFORMS, VERSIONS


class CIFAR100N(MultiAnnotatorDataset):
    """CIFAR100N

    The CIFAR100N [1] dataset features about 60,000 images of 100 classes, which have been annotated by 519 annotators
    with an accuracy of about 60%.

    Parameters
    ----------
    root : str
        Path to the root directory, where the ata is located.
    version : "train" or "valid" or "test", default="train"
        Defines the version (split) of the dataset.
    download : bool, default=False
        Flag whether the dataset will be downloaded.
    annotators : None or "index" or "one-hot" or "metadata"
        Defines the representation of the annotators as either indices, one-hot encoded vectors, or`None`.
    aggregation_method : str, default=None
        Supported methods are majority voting (`aggregation_method="majority_vote") and returning the true class
        labels (`aggregation_method="ground-truth"). In the case of `aggregation_method=None`, `None` is returned
        as aggregated annotations.
    transform : "auto" or torch.nn.Module, default="auto"
        Transforms for the samples, where "auto" used pre-defined transforms fitting the respective version.

    References
    ----------
    [1] Wei, J., Zhu, Z., Cheng, H., Liu, T., Niu, G., & Liu, Y. (2022). Learning with Noisy Labels
        Revisited: A Study Using Real-World Human Annotations. Int. Conf. Learn. Represent.
    """

    url_annotations = "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/"
    annotations_filename = "CIFAR-100_human.pt"
    url_side_information = "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/"
    side_information_filename = "side_info_cifar100N.csv"

    def __init__(
        self,
        root: str,
        version: VERSIONS = "train",
        annotators: ANNOTATOR_FEATURES = None,
        download: bool = False,
        aggregation_method: AGGREGATION_METHODS = None,
        transform: TRANSFORMS = "auto",
        realistic_split: str = "cv-5-0",
    ):
        # Determine whether we load the original training subset.
        is_train = (version == "train" or realistic_split is not None) and version != "test"

        # Download data.
        cifar100 = CIFAR100(
            root=root,
            train=is_train,
            download=download,
        )
        if download:
            url_list = [
                (CIFAR100N.url_side_information, CIFAR100N.side_information_filename),
                (CIFAR100N.url_annotations, CIFAR100N.annotations_filename),
            ]
            for url, filename in url_list:
                response = requests.get(url=f"{url}/{filename}", params={"downloadformat": "csv"})
                with open(os.path.join(root, filename), mode="wb") as file:
                    file.write(response.content)

        # Check availability of data.
        is_available = os.path.exists(os.path.join(root, CIFAR100N.side_information_filename)) and os.path.exists(
            os.path.join(root, CIFAR100N.annotations_filename)
        )
        if not is_available:
            raise RuntimeError("Dataset not found. You can use `download=True` to download it.")

        # Set transforms.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if transform == "auto" and version == "train":
            self.transform = Compose(
                [
                    Resize(232),
                    RandomResizedCrop(224),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    RandomErasing(),
                    Normalize(mean, std),
                ]
            )
        elif transform == "auto" and version in ["valid", "test"]:
            self.transform = Compose([Resize(232), CenterCrop(224), ToTensor(), Normalize(mean, std)])
        else:
            self.transform = transform

        # Set samples and targets.
        self.x = cifar100.data
        self.y = torch.tensor(cifar100.targets).long()

        # Load and prepare annotations as tensor for `version="train"`.
        self.z = None
        if is_train:
            side_information_file = os.path.join(root, CIFAR100N.side_information_filename)
            annotator_ids = pd.read_csv(side_information_file)[["Worker-id"]].values
            annotator_ids = np.repeat(annotator_ids, repeats=5, axis=0).astype(int).ravel()
            self.z = torch.full((len(self.x), self.get_n_annotators()), fill_value=-1)
            annotation_file = os.path.join(root, CIFAR100N.annotations_filename)
            annotations = torch.load(annotation_file)
            self.z[np.arange(len(self.x)), annotator_ids] = torch.from_numpy(annotations["noisy_label"])
            train_indices = torch.arange(len(self.x))
            if isinstance(realistic_split, float):
                train_indices, valid_indices = train_test_split(
                    train_indices, train_size=realistic_split, random_state=0
                )
            elif isinstance(realistic_split, str) and realistic_split.startswith("cv"):
                n_splits = int(realistic_split.split("-")[1])
                split_idx = int(realistic_split.split("-")[2])
                k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=0)
                train_indices, valid_indices = list(k_fold.split(train_indices))[split_idx]
            version_indices = train_indices if version == "train" else valid_indices
            self.z = self.z[version_indices]
        elif version in ["valid", "test"]:
            version_indices = torch.arange(len(self.x))
            if realistic_split is None:
                valid_indices, test_indices = train_test_split(
                    torch.arange(len(self.x)), train_size=1000, random_state=0, stratify=self.y
                )
                version_indices = valid_indices if version == "valid" else test_indices

        # Index filenames and labels according to version.
        self.x, self.y = self.x[version_indices], self.y[version_indices]

        # Load and prepare annotator features as tensor if `annotators` is not `None`.
        self.a = self.prepare_annotator_features(annotators=annotators, n_annotators=self.get_n_annotators())

        # Aggregate annotations if `aggregation_method` is not `None`.
        self.z_agg, self.ap_confs = self.aggregate_annotations(
            z=self.z, y=self.y, aggregation_method=aggregation_method
        )

        # Print statistics.
        print(version)
        print(self)

    def __len__(self):
        """
        Returns
        -------
        length: int
            Length of the dataset.
        """
        return len(self.x)

    def get_n_classes(self):
        """
        Returns
        -------
        n_classes : int
            Number of classes.
        """
        return 100

    def get_n_annotators(self):
        """
        Returns
        -------
        n_annotators : int
            Number of annotators.
        """
        return 519

    def get_annotators(self):
        """
        Returns
        -------
        annotators : None or torch.tensor of shape (n_annotators, *)
            Representation of the annotators, e.g., one-hot encoded vectors or metadata.
        """
        return self.a

    def get_sample(self, idx: int):
        """
        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        sample : torch.tensor
            Sample with the given index.
        """
        x = Image.fromarray(self.x[idx])
        return self.transform(x) if self.transform else x

    def get_annotations(self, idx: int):
        """
        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        annotations : torch.tensor
            Annotations with the given index.
        """
        return self.z[idx] if self.z is not None else None

    def get_aggregated_annotation(self, idx: int):
        """
        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        aggregated_annotation : torch.tensor
            Aggregated annotation with the given index.
        """
        return self.z_agg[idx] if self.z_agg is not None else None

    def get_true_label(self, idx: int):
        """
        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        true_label : torch.tensor
            True class label with the given index.
        """
        return self.y[idx] if self.y is not None else None
