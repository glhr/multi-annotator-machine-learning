import torch
import numpy as np
import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, KFold
from torchvision.datasets.utils import download_and_extract_archive


from ._base import MultiAnnotatorDataset, ANNOTATOR_FEATURES, AGGREGATION_METHODS, TRANSFORMS, VERSIONS


class Reuters(MultiAnnotatorDataset):
    """Reuters

    The Reuters [1] dataset features text files of 8 classes, which have been annotated by 38 annotators
    with an accuracy of about 56%.

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
    variant :"worst-1" or "worst-2" or "worst-var" or "rand-1" or "rand-2" or "rand-3" or "rand-2" or "rand-var" or\
            "full"
        Defines subsets of annotations to reflect different learning scenarios.

    References
    ----------
    [1] F. Rodrigues, M. Lourenço, B. Ribeiro and F. C. Pereira, "Learning Supervised Topic Models for Classification
        and Regression from Crowds," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no.
        12, pp. 2409-2422, 2017.
    """

    base_folder = "Reuters"
    url = "http://fprodrigues.com/Reuters.tar.gz"
    filename = "Reuters.tar.gz"

    def __init__(
        self,
        root: str,
        version: VERSIONS = "train",
        annotators: ANNOTATOR_FEATURES = None,
        download: bool = False,
        aggregation_method: AGGREGATION_METHODS = None,
        transform: TRANSFORMS = "auto",
        variant: str = "full",
        realistic_split: str = "cv-5-0",
    ):
        # Download data.
        if download:
            download_and_extract_archive(Reuters.url, root, filename=Reuters.filename)

        # Check availability of data.
        is_available = os.path.exists(os.path.join(root, Reuters.base_folder))
        if not is_available:
            raise RuntimeError("Dataset not found. You can use `download=True` to download it.")

        # Load samples and class labels.
        file_version = "train" if realistic_split is not None and version == "valid" else version
        folder = os.path.join(root, Reuters.base_folder)
        z = pd.read_csv(os.path.join(folder, f"answers.txt"), header=None, sep=" ").values.astype(int)
        at_least_one_label = (z != -1).any(axis=-1)
        x_dict = {v: _load_features(os.path.join(folder, f"data_{v}.txt")) for v in ["train", "valid", "test"]}
        x_dict["train"] = [x_dict["train"][i] for i, v in enumerate(at_least_one_label) if v]
        y_dict = {v: _load_labels(os.path.join(folder, f"labels_{v}.txt")) for v in ["train", "valid", "test"]}
        y_dict["train"] = y_dict["train"][torch.from_numpy(at_least_one_label)]
        indices = {v: np.arange(len(y_dict[v]), dtype=int) for v in ["train", "valid", "test"]}
        self.x = x_dict[file_version]
        self.y = y_dict[file_version]

        # Load and prepare annotations as tensor.
        z = z[at_least_one_label]
        z = Reuters.mask_annotations(
            z=z, y_true=y_dict["train"].numpy(), variant=variant, n_variants=2, is_not_annotated=z == -1
        )
        provided_labels = np.sum(z != -1, axis=0) > 0
        z = z[:, provided_labels]
        z = torch.from_numpy(z)
        self.n_annotators = z.shape[-1]
        if (version == "train" or realistic_split is not None) and version != "test":
            self.z = z
        else:
            self.z = None

        if isinstance(realistic_split, float):
            indices["train"], indices["valid"] = train_test_split(
                indices["train"], train_size=realistic_split, random_state=0
            )
        elif isinstance(realistic_split, str) and realistic_split.startswith("cv"):
            n_splits = int(realistic_split.split("-")[1])
            split_idx = int(realistic_split.split("-")[2])
            k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=0)
            indices["train"], indices["valid"] = list(k_fold.split(indices["train"]))[split_idx]

        # Set transforms.
        if transform == "auto":
            pipeline = Pipeline(
                [
                    ("vectorizer", DictVectorizer(sparse=True)),
                    ("tfidf", TfidfTransformer()),
                ]
            )
            pipeline.fit([x_dict["train"][i] for i in indices["train"]])
            self.x = pipeline.transform(self.x).toarray()
            x_const = np.zeros((len(self.x), 8884))
            x_const[:, : self.x.shape[1]] = self.x
            self.x = x_const
            self.transform = None
        else:
            self.transform = transform

        # Index final sets.
        self.z = self.z[indices[version]] if self.z is not None else None
        self.y = self.y[indices[version]]
        self.x = torch.from_numpy(self.x[indices[version]]).float()

        # Load and prepare annotator features as tensor if `annotators` is not `None`.
        self.a = self.prepare_annotator_features(annotators=annotators, n_annotators=self.get_n_annotators())

        # Aggregate annotations if `aggregation_method` is not `None`.
        self.z_agg, self.ap_confs = self.aggregate_annotations(
            z=self.z, y=self.y, aggregation_method=aggregation_method
        )

        # Print statistics.
        print(f"variant: {version}")
        print(self)

    def __len__(self):
        """
        Returns
        -------
        length: int
            Length of the dataset.
        """
        return len(self.y)

    def get_n_classes(self):
        """
        Returns
        -------
        n_classes : int
            Number of classes.
        """
        return 8

    def get_n_annotators(self):
        """
        Returns
        -------
        n_annotators : int
            Number of annotators.
        """
        return self.n_annotators

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
        return self.transform(self.x[idx]) if self.transform else self.x[idx]

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


def _load_features(file_path):
    """Load document features from a text file.

    Each line is expected to be of the form:
    [M] [term_1]:[count] [term_2]:[count] ... [term_N]:[count]
    where [M] is the number of unique terms (which we ignore here).
    """
    documents = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # The first element is the count of unique terms; skip it.
            term_counts = {}
            for pair in parts[1:]:
                term, count = pair.split(":")
                term_counts[term] = int(count)
            documents.append(term_counts)
    return documents


def _load_labels(file_path):
    """Load labels from a text file (one label per line)."""
    labels = []
    with open(file_path, "r") as f:
        for line in f:
            label = line.strip()
            if label:
                labels.append(label)
    return torch.from_numpy(np.array(labels, dtype=int))
