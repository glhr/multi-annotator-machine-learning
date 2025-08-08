import numpy as np
import os
import pandas as pd
import torch
import json

from PIL import Image
from skactiveml.utils import ExtLabelEncoder, rand_argmax
from sklearn.preprocessing import StandardScaler
from torchvision.datasets.utils import extract_archive
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
from numpy.typing import ArrayLike
from typing import Literal, Optional
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import PredefinedSplit
from huggingface_hub import snapshot_download

from ._base import MultiAnnotatorDataset, AGGREGATION_METHODS, TRANSFORMS, VERSIONS


class JAMBO(MultiAnnotatorDataset):
    """JAMBO

    The JAMBO [1] dataset features 3750 images of the seabed captured over the course of several months in the
    Greater North Sea, Denmark. Each image was annotated by six annotators (three computer vision experts and three
    marine biologists) as belonging to one of 3 classes: "sand", "stone", and "bad". For around 30\% of the images,
    the annotators do not fully agree on the correct class label. Since there is no "golden" ground truth, we consider
    two label sets as the reference: the majority vote of the annotators and the true class labels provided by Bio2, 
    which is the most experienced.
    
    Parameters
    ----------
    root : str
        Path to the root directory, where the ata is located.
    version : "train" or "valid" or "test", default="train"
        Defines the version (split) of the dataset.
    download : bool, default=False
        Flag whether the dataset will be downloaded.
    annotators : None or "index" or "one-hot" or "metadata"
        Defines the representation of the annotators as either indices, one-hot encoded vectors, real metadata, or
        `None`.
    aggregation_method : str, default=None
        Supported methods are majority voting (`aggregation_method="majority_vote") and returning the true class
        labels (`aggregation_method="ground-truth"). In the case of `aggregation_method=None`, `None` is returned
        as aggregated annotations.
    transform : "auto" or torch.nn.Module, default="auto"
        Transforms for the samples, where "auto" used pre-defined transforms fitting the respective version.
    variant :"worst-1" or "worst-2" or "worst-3" or "worst-4" or "worst-v" or "rand-1" or "rand-2" or "rand-3" or
    "rand-4" or "rand-v" or "full"
        Defines subsets of annotations to reflect different learning scenarios.
    annotation_type : "class-labels" or "probabilities", default="class-labels",
        Defines which type of annotations is used.

    References
    ----------
    [1] Humblot-Renaux, G., Johansen, A. S., Schmidt, J. E., Irlind, A. F., Madsen, N., Moeslund, T. B., 
    & Pedersen, M. (2025). Underwater Uncertainty: A Multi-annotator Image Dataset for Benthic Habitat Classification, 
    Computer Vision -- ECCV 2024 Workshops (pp. 87-104). Springer. https://doi.org/10.1007/978-3-031-92387-6_6

    """

    base_folder = "jambo"
    hf_repo = "vapaau/jambo"
    image_dir = "images"
    classes = np.array(
        [
            "sand",
            "stone",
            "bad"
        ],
        dtype=object,
    )
    annotators = np.array(
        [
            "CV1",
            "CV2",
            "CV3",
            "Bio1",
            "Bio2",
            "Bio3"
        ],
        dtype=object,
    )
    variants = np.array(
        [
            "full",
        ],
        dtype=object,
    )

    def __init__(
        self,
        root: str,
        version: VERSIONS = "train",
        download: bool = False,
        annotators: Optional[Literal["one-hot", "index", "metadata"]] = None,
        aggregation_method: AGGREGATION_METHODS = None,
        transform: TRANSFORMS = "auto",
        variant: str = "full",
        annotation_type: Literal["class-labels", "probabilities"] = "class-labels",
        normalization_params: Literal["imagenet", "0.5"] = "imagenet",
        ground_truth_variant: Literal["majority", "expert"] = "expert",
    ):
        # Download the data.
        self.folder = os.path.join(root, JAMBO.base_folder)
        is_available = os.path.exists(self.folder)
        if download and not is_available:
            snapshot_download(repo_id=JAMBO.hf_repo, local_dir=self.folder, repo_type="dataset", revision="zip")
            # Extract the images zip file.
            extract_archive(os.path.join(self.folder, JAMBO.image_dir + ".zip"), self.folder, remove_finished=False)

        # Set dataset parameters.
        self.variant = variant
        self.annotation_type = annotation_type
        self.normalization_params = normalization_params
        self.ground_truth_variant = ground_truth_variant

        # Check availability of data.
        is_available = os.path.exists(self.folder)
        if not is_available:
            raise RuntimeError("Dataset not found. You can use `download=True` to download it.")

        # Load annotation file.
        if version not in ["train", "valid", "test"]:
            raise ValueError("`version` must be in `['train', 'valid', 'test']`.")
        self.img_folder = os.path.join(self.folder, "images")

        # Load and prepare true labels as tensor.
        self.y_orig, self.observation_ids = self.load_true_class_labels(version=version)
        self.le = ExtLabelEncoder(missing_label=None, classes=JAMBO.classes).fit(self.y_orig)
        self.y = self.le.transform(self.y_orig)

        # Load and prepare annotations as tensor.
        self.z = self.load_annotations() if version == "train" else None

        # Set transforms.
        if self.normalization_params == "imagenet":
            mean = (0.485, 0.456, 0.406) 
            std = (0.229, 0.224, 0.225) 
        elif self.normalization_params == "0.5":
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        if transform == "auto" and version == "train":
            self.transform = Compose(
                [
                    Resize(32),
                    #RandomResizedCrop(224),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    RandomErasing(),
                    Normalize(mean, std),
                ]
            )
        elif transform == "auto" and version in ["valid", "test"]:
            self.transform = Compose([Resize(32), CenterCrop(32), ToTensor(), Normalize(mean, std)])
        else:
            self.transform = transform

        # Transform to tensors.
        self.y = torch.from_numpy(self.y)
        self.z = torch.from_numpy(self.z) if self.z is not None else None

        # Load and prepare annotator features as tensor if `annotators` is not `None`.
        if annotators == "metadata":
            z = self.load_annotations() if self.z is None else self.z
            self.a, _ = self.load_annotator_metadata(
                classes=self.le.classes_,
                annotators=JAMBO.annotators,
                is_not_annotated=z == -1,
            )
            self.a = torch.from_numpy(StandardScaler().fit_transform(self.a)).float()
        else:
            self.a = self.prepare_annotator_features(annotators=annotators, n_annotators=self.get_n_annotators())

        # Aggregate annotations if `aggregation_method` is not `None`.
        self.z_agg = self.aggregate_annotations(z=self.z, y=self.y, aggregation_method=aggregation_method)
        if self.z_agg is not None:
            is_labeled = self.z_agg != -1
            if self.z_agg.ndim == 2:
                is_labeled = is_labeled.all(dim=-1)
            self.z_agg = self.z_agg[is_labeled]
            self.y = self.y[is_labeled]
            self.y_orig = self.y_orig[is_labeled]
            self.z = self.z[is_labeled]
            self.observation_ids = self.observation_ids[is_labeled]

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
        return len(JAMBO.classes)

    def get_n_annotators(self):
        """
        Returns
        -------
        n_annotators : int
            Number of annotators.
        """
        return len(JAMBO.annotators)

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
        x = Image.open(os.path.join(self.img_folder, f"{self.observation_ids[idx]}"))
        x = x.convert("RGB")
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

    def load_annotations(self):
        """
        Loads the annotations of the given variant and annotation type.

        Returns
        -------
        z : np.ndarray of shape (n_samples, n_annotators) or (n_samples, n_annotators, n_classes)
            Observed annotations.
        """
        y_true_train, observation_ids_train = self.load_true_class_labels(version="train")
        y_true_train = self.le.transform(y_true_train)
        likelihoods = self.load_likelihoods(
            observation_ids=observation_ids_train,
            classes=self.le.classes_,
            annotators=JAMBO.annotators,
            normalize=True,
        )
        is_not_annotated = np.any(likelihoods == -1, axis=-1)
        class_labels = rand_argmax(likelihoods, axis=-1, random_state=0)
        class_labels[is_not_annotated] = -1
        if self.annotation_type == "probabilities":
            z = likelihoods
        elif self.annotation_type == "class-labels":
            z = class_labels
        else:
            raise ValueError(
                f"`annotation_type` must be in ['class-labels', 'probabilities'], got '{self.annotation_type}' instead."
            )
        if self.variant in ["worst-1", "worst-2", "worst-3", "worst-4"]:
            n_annotators_per_sample = int(self.variant.split("-")[-1])
            is_false = np.full_like(class_labels, fill_value=0.0, dtype=float)
            is_false += (y_true_train[:, None] != class_labels).astype(float)
            is_false -= 2 * is_not_annotated.astype(float)
            is_not_worst = np.full_like(is_false, fill_value=True, dtype=bool)
            random_floats = np.random.RandomState(n_annotators_per_sample).rand(*is_false.shape)
            worst_indices = np.argsort(-(is_false + random_floats), axis=-1)[:, :n_annotators_per_sample]
            for c in range(n_annotators_per_sample):
                is_not_worst[np.arange(len(is_not_worst)), worst_indices[:, c]] = False
            z[is_not_worst] = -1
        elif self.variant in ["rand-1", "rand-2", "rand-3", "rand-4"]:
            n_annotators_per_sample = int(self.variant.split("-")[-1])
            is_annotated = (~is_not_annotated).astype(float)
            is_not_selected = np.full_like(is_annotated, fill_value=True, dtype=bool)
            random_floats = np.random.RandomState(n_annotators_per_sample + 4).rand(*is_annotated.shape)
            random_indices = np.argsort(-(is_annotated + random_floats), axis=-1)[:, :n_annotators_per_sample]
            for c in range(n_annotators_per_sample):
                is_not_selected[np.arange(len(is_annotated)), random_indices[:, c]] = False
            z[is_not_selected] = -1
        elif self.variant in ["rand-var", "worst-var"]:
            random_state = np.random.RandomState(0)
            for i in range(len(is_not_annotated)):
                # Get the indices of ones in the current row
                annotated_indices = np.where(is_not_annotated[i] == False)[0]

                # Determine the size of the subset to set to zero
                subset_size = random_state.randint(0, len(annotated_indices))

                # Select worst indices to set to zero
                if self.variant == "worst-var":
                    is_false = class_labels[i][annotated_indices] == y_true_train[i]
                    random_floats = random_state.rand(*is_false.shape)
                    worst_indices = np.argsort(-(is_false + random_floats), axis=-1)[:subset_size]
                    indices_to_true = annotated_indices[worst_indices]
                else:
                    # Randomly select indices to set to zero
                    indices_to_true = random_state.choice(annotated_indices, size=subset_size, replace=False)

                # Set the selected indices to zero
                is_not_annotated[i, indices_to_true] = True
            z[is_not_annotated] = -1
        elif self.variant == "full":
            pass
        else:
            raise ValueError(f"`variant` must be in {JAMBO.variants}, got '{self.variant}' instead.")
        return z

    def load_true_class_labels(self, version: VERSIONS = "train"):
        """
        Loads the true class of the given version.

        Parameters
        ----------
        version : "train" or "valid" or "test"
            Version (split) of the dataset.

        Returns
        -------
        z : np.ndarray of shape (n_samples,)
            True class labels.
        """
        meta_df = pd.read_csv(os.path.join(self.folder, "jambo_meta_public.csv"))
        splits_df = pd.read_csv(os.path.join(self.folder, "jambo_splits_public.csv"), index_col="filename")
        folds = splits_df["date_splits"] # TODO: make this a parameter
        ps = PredefinedSplit(folds)
        print(ps.get_n_splits(), "num splits")
        
        if self.ground_truth_variant == "majority":
            label_col = "majority_label"
        elif self.ground_truth_variant == "expert":
            label_col = "Bio2"
        
        train_index, test_index = list(ps.split())[0] # first fold in the split
        split_indices = {
            "train": meta_df['filename'].iloc[train_index].tolist(),
            "test": meta_df['filename'].iloc[test_index].tolist(),
            "valid": meta_df['filename'].iloc[test_index].tolist()
        }
        y_true_list = []
        observation_id_list = []
        for i, df_row in meta_df.iterrows():
            observation_id = df_row["filename"]
            if observation_id in split_indices[version]:
                y_true_list.append(df_row[label_col])
                observation_id_list.append(observation_id)
                
        print(f"Loaded {len(y_true_list)} true class labels for version '{version}'.")
        return np.array(y_true_list, dtype=object), np.array(observation_id_list, dtype=object)

    def load_likelihoods(
        self,
        observation_ids: ArrayLike,
        annotators: Optional[ArrayLike] = None,
        classes: Optional[ArrayLike] = None,
        normalize: bool = True,
    ):
        """
        Loads the likelihoods for the given observation IDs.

        Parameters
        ----------
        observation_ids : array-like of shape (n_obs_ids,)
            Observation IDs whose likelihoods are loaded.
        annotators : array-like of shape (n_annotators,), default=None
            Names of the annotators whose likelihoods are loaded.
        classes : array-like of shape (n_classes,), default=None
            Defines the order of the class labels.
        normalize : bool, default=True
            Flag whether likelihoods are normalized.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_annotators, n_classes)
            Observed likelihoods.
        """
        annotations_df = pd.read_csv(os.path.join(self.folder, "jambo_annotations_public.csv"))
        annotators = JAMBO.annotators if annotators is None else annotators
        classes = JAMBO.classes if classes is None else classes
        likelihoods = np.full(
            (len(observation_ids), len(annotators), len(classes)),
            fill_value=-1,
            dtype=float,
        )
        #print(observation_ids, annotators, classes)
        for annotation_id, annotation_row in annotations_df.iterrows():
            if annotation_row["filename"] not in observation_ids:
                continue
            obs_idx = np.where(annotation_row["filename"] == observation_ids)[0]
            annot_idx = np.where(annotation_row["annotator"] == annotators)[0]
            class_name = annotation_row["label"]
            cls_idx = np.where(class_name == classes)[0]
            #print(obs_idx, annot_idx, cls_idx)
            #print(likelihoods.shape)
            cls_idx_rest = np.where(class_name != classes)[0]
            likelihoods[obs_idx, annot_idx, cls_idx] = 1.0
            likelihoods[obs_idx, annot_idx, cls_idx_rest] = 0.0
            if normalize and likelihoods[obs_idx, annot_idx].sum() > 0:
                likelihoods[obs_idx, annot_idx] = (
                    likelihoods[obs_idx, annot_idx] / likelihoods[obs_idx, annot_idx].sum()
                )
        # make sure that there are no -1 values in the likelihoods
        assert np.all(likelihoods > -1), "Likelihoods contain missing values (-1)."
        return likelihoods

    def load_annotation_consistencies(
        self,
        observation_ids: ArrayLike,
        annotators: Optional[ArrayLike] = None,
        is_not_annotated: Optional[ArrayLike] = None,
    ):
        """
        Loads the annotation consistencies per annotator.

        Parameters
        ----------
        observation_ids : array-like of shape (n_obs_ids,)
            Observation IDs for which the annotation consistencies are loaded.
        annotators : array-like of shape (n_annotators,), default=None
            Names of the annotators whose annotation consistencies are loaded.
        is_not_annotated : array-like of shape (n_samples, n_annotators), default=None
            A boolean mask indicating which sample is annotated by which annotator. If `ìs_not_annotated=None`, the
            missing annotations are computed for the `full` variant.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_annotators,)
            Observed likelihoods.
        """
        raise NotImplementedError # TODO: Implement annotation consistencies for JAMBO dataset.

    def load_annotation_times(self, observation_ids: ArrayLike, annotators: Optional[ArrayLike] = None):
        """
        Loads the annotation times per sample and annotator.

        Parameters
        ----------
        observation_ids : array-like of shape (n_obs_ids,)
            Observation IDs for which annotation times are loaded.
        annotators : array-like of shape (n_annotators,), default=None
            Names of the annotators whose annotation times are loaded.

        Returns
        -------
        annotation_times : np.ndarray of shape (n_samples, n_annotators)
            Observed annotation times.
        """
        annotations_df = pd.read_csv(os.path.join(self.folder, "jambo_annotations_public.csv"))
        annotation_times = np.full((len(observation_ids), len(annotators)), fill_value=-1, dtype=float)
        for _, annotation_row in annotations_df.iterrows():
            obs_idx = np.where(annotation_row["filename"] == observation_ids)[0]
            annot_idx = np.where(annotation_row["annotator"] == annotators)[0]
            annotation_times[obs_idx, annot_idx] = annotation_row["time"]

        for annot_idx in range(len(annotators)):
            times_annot_idx = annotation_times[:, annot_idx]
            is_annotated = times_annot_idx > 0
            is_outlier = times_annot_idx > np.quantile(times_annot_idx[is_annotated], q=0.95)
            random_state = np.random.RandomState(0)
            times_annot_idx[is_outlier] = random_state.uniform(
                low=np.quantile(times_annot_idx[is_annotated], q=0.1),
                high=np.quantile(times_annot_idx[is_annotated], q=0.9),
                size=np.sum(is_outlier),
            )
            annotation_times[:, annot_idx] = times_annot_idx
        return annotation_times

    def load_annotator_metadata(
        self, annotators: Optional[ArrayLike] = None, is_not_annotated: Optional[ArrayLike] = None
    ):
        """
        Loads the metadata per annotator.

        Parameters
        ----------
        annotators : array-like of shape (n_annotators,), default=None
            Names of the annotators whose metadata are loaded.
        is_not_annotated : array-like of shape (n_samples, n_annotators), default=None
            A boolean mask indicating which sample is annotated by which annotator. If `ìs_not_annotated=None`, the
            missing annotations are computed for the `full` variant.

        Returns
        -------
        annotator_metadata_values : np.ndarray of shape (n_annotators, n_metadata_features)
            Observed metadata.
        annotator_metadata_names : np.ndarray of shape (n_metadata_features,)
            Feature names of the metadata.
        """
        raise NotImplementedError # TODO: Implement annotator metadata loading for JAMBO dataset.
