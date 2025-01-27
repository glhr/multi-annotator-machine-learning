import math
import torch

from torch.optim import Optimizer, RAdam
from torch.optim.lr_scheduler import LRScheduler
from torch import nn
from torch.nn import functional as F

from typing import Optional, Dict, Union

from ._base import MaMLClassifier
from ..utils import mixup, permute_same_value_indices


class AnnotMixModule(nn.Module):
    """AnnotMixModule

    This class implements an auxiliary module of the framework annot-mix [1], which trains a multi-annotator classifier
    using an extension of mixup [2].

    Parameters
    ----------
     n_classes : int
        Number of classes
    gt_embed_x : nn.Module
        Pytorch module the GT model' backbone embedding the input samples.
    gt_output : nn.Module
        Pytorch module of the GT model taking the embedding the samples as input to predict class-membership logits.
    ap_embed_a : nn.Module
        Pytorch module of the AP model embedding the annotator features for the AP model.
    ap_output : nn.Module
        Pytorch module of the AP model predicting the logits of the conditional confusion matrix
    ap_embed_x : nn.Module, optional (default=None)
        Pytorch module of the AP model embedding samples.
    ap_hidden : nn.Module, optional (default=None)
        Pytorch module of the AP model taking the concatenation of annotator, sample (optional) and outer product
        (optional) embedding as input to create a new embedding as input to the `ap_output` module.
        By default, it is an identity mapping.

    References
    ----------
    [1] Herde, M., Lührs, L., Huseljic, D., & Sick, B. (2024). Annot-Mix: Learning with Noisy Class Labels from
        Multiple Annotators via a Mixup Extension. arXiv:2405.03386.
    [2] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization.
        Int. Conf. Learn. Represent.
    """

    def __init__(
        self,
        n_classes: int,
        gt_embed_x: nn.Module,
        gt_output: nn.Module,
        ap_embed_a: nn.Module,
        ap_output: nn.Module,
        ap_embed_x: Optional[nn.Module] = None,
        ap_hidden: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.gt_embed_x = gt_embed_x
        self.gt_output = gt_output
        self.ap_embed_a = ap_embed_a
        self.ap_output = ap_output
        self.ap_hidden = ap_hidden
        self.ap_embed_x = ap_embed_x

    def forward(
        self, x: torch.tensor, a: Optional[torch.tensor] = None, combs: Union[str, None, torch.tensor] = "full"
    ):
        """
        Forward propagation of samples' and annotators' (optional) features through the GT and AP (optional) model.

        Parameters
        ----------
        x : torch.tensor of shape (batch_size, *)
            Sample features.
        a : torch.tensor of shape (n_annotators, *), default=None
            Annotator features, which are None by default. In this case, only the samples are forward propagated
            through the GT model.
        combs : torch.tensor of shape (n_combs, 2), default=None
            If provided, this tensor determines the pairs of samples (indexed by `combs[:, 0]`) and annotators
            (indexed by `combs[:, 1]`) to be propagated through the AP model.

        Returns
        -------
        logits_class : torch.tensor of shape (batch_size, n_classes)
            Class-membership logits.
        logits_perf : torch.tensor of shape (batch_size * n_annotators, n_classes, n_classes)
            Logits of conditional confusion matrices as proxies of the annotators' performances.
        """
        # Compute feature embedding for classifier.
        x_learned = self.gt_embed_x(x)

        # Compute class-membership probabilities.
        logits_class = self.gt_output(x_learned)

        if a is None:
            return logits_class

        # Compute annotator performances per annotator.
        annot_embeddings = self.ap_embed_a(a)
        sample_embeddings = self.ap_embed_x(x_learned.detach())
        if combs == "full":
            n_annotators = a.shape[0]
            n_samples = x.shape[0]
            combs = torch.cartesian_prod(torch.arange(n_samples), torch.arange(n_annotators))
            combs = {"x": combs[:, 0], "a": combs[:, 1]}
        if isinstance(combs, dict) and "x" in combs:
            sample_embeddings = sample_embeddings[combs["x"]]
        if isinstance(combs, dict) and "a" in combs:
            annot_embeddings = annot_embeddings[combs["a"]]

        # Compute sample and annotator embeddings.
        embeddings = torch.concat([sample_embeddings, annot_embeddings], dim=-1)

        # Propagate embeddings through hidden layers.
        embeddings = self.ap_hidden(embeddings)

        # Compute logits of annotator performances.
        logits_perf = self.ap_output(embeddings)
        logits_perf = logits_perf.reshape((-1, self.n_classes, self.n_classes))

        return logits_class, logits_perf


class AnnotMixClassifier(MaMLClassifier):
    """AnnotMixClassifier

    This class implements the framework annot-mix [1], which trains a multi-annotator classifier using an extension of
    mixup [2].

    Parameters
    ----------
    network : AnnotMixModule
        A network consisting of a ground truth (AP) and an annotator performance (AP) model.
    alpha : float, default=1.0
        Determines the parameters of the beta distribution used for sampling the mixup coefficients.
    mix_only_annotators : None or str, optional (default="x-a-mixup")
        Flag whether annotators are only mixed for the same sample.
    optimizer : torch.optim.Optimizer.__class__, optional (default=RAdam.__class__)
        Optimizer class responsible for optimizing the GT and AP parameters. If `None`, the `RAdam` optimizer is used
        by default.
    optimizer_gt_dict : dict, optional (default=None)
        Parameters passed to `optimizer` for the GT model.
    optimizer_ap_dict : dict, optional (default=None)
        Parameters passed to `optimizer` for the AP model.
    lr_scheduler : torch.optim.lr_scheduler.LRScheduler.__class__, optional (default=None)
        Learning rate scheduler responsible for optimizing the GT and AP parameters. If `None`, no learning rate
        scheduler is used by default.
    lr_scheduler_dict : dict, optional (default=None)
        Parameters passed to `lr_scheduler`.

    References
    ----------
    [1] Herde, M., Lührs, L., Huseljic, D., & Sick, B. (2024). Annot-Mix: Learning with Noisy Class Labels from
        Multiple Annotators via a Mixup Extension. arXiv:2405.03386.
    [2] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization.
        Int. Conf. Learn. Represent.
    """

    def __init__(
        self,
        network: AnnotMixModule,
        alpha: float = 1.0,
        eta: float = 0.9,
        mix_only_annotators: bool = False,
        optimizer: Optimizer.__class__ = RAdam,
        optimizer_gt_dict: Optional[dict] = None,
        optimizer_ap_dict: Optional[dict] = None,
        lr_scheduler: Optional[LRScheduler.__class__] = None,
        lr_scheduler_dict: Optional[dict] = None,
    ):
        super().__init__(
            optimizer=optimizer,
            optimizer_gt_dict=optimizer_gt_dict,
            optimizer_ap_dict=optimizer_ap_dict,
            lr_scheduler=lr_scheduler,
            lr_scheduler_dict=lr_scheduler_dict,
        )
        self.network = network
        self.alpha = alpha
        self.eta = eta
        self.mix_only_annotators = mix_only_annotators
        self.register_buffer("eye", torch.eye(self.network.n_classes))
        self.register_buffer("one_minus_eye", 1 - self.eye)

        # Set prior eta as bias.
        bias = (math.log(eta * (self.network.n_classes - 1) / (1 - eta)) * self.eye).flatten()
        self.network.ap_output.bias = torch.nn.Parameter(bias)

        # Save hyper parameters.
        self.save_hyperparameters(logger=False)

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int, dataloader_idx: Optional[int] = 0):
        """
        Computes the AnnotMix's loss.

        Parameters
        ----------
        batch : dict
            Data batch fitting the dictionary structure of `maml.data.MultiAnnotatorDataset`.
        batch_idx : int
            Index of the batch in the dataset.
        dataloader_idx : int, default=0
            Index of the used dataloader.

        Returns
        -------
        loss : torch.Float
            Computed cross-entropy loss.
        """
        # Get data.
        x, a, z = batch["x"], batch["a"], batch["z"]
        n_samples, n_annotators = x.shape[0], a.shape[1]

        # Compute all combinations of samples and annotators.
        combs = torch.cartesian_prod(
            torch.arange(n_samples, device=self.device), torch.arange(n_annotators, device=self.device)
        )

        # Prepare samples, annotators, and labels for training.
        z, a = z.ravel(), a[0]
        is_lbld = z != -1
        combs, z = combs[is_lbld], z[is_lbld]
        z = F.one_hot(z + 1, num_classes=self.network.n_classes + 1)[:, 1:].float()

        # Mixup of samples, annotators, and labels.
        if self.alpha > 0:
            x, a = x[combs[:, 0]], a[combs[:, 1]]
            permute_indices = None
            if self.mix_only_annotators:
                permute_indices = permute_same_value_indices(combs[:, 0])
            x, a, z, _, _ = mixup(x, a, z, alpha=self.alpha, permute_indices=permute_indices)
            logits_class, logits_perf = self.network(x=x, a=a, combs=combs)
        else:
            combs = {"x": combs[:, 0], "a": combs[:, 1]}
            logits_class, logits_perf = self.network(x=x, a=a, combs=combs)
            logits_class = logits_class[combs["x"]]

        # Outputs of network.
        p_class_log = F.log_softmax(logits_class, dim=-1)
        p_perf_log = F.log_softmax(logits_perf, dim=-1)
        p_annot_log = torch.logsumexp(p_class_log[:, :, None] + p_perf_log, dim=1)

        # Compute loss.
        return AnnotMixClassifier.loss(
            z=z,
            p_annot_log=p_annot_log,
        )

    @staticmethod
    def loss(z: torch.tensor, p_annot_log: torch.tensor):
        """
        Computes the cross-entropy loss between the observed annotations and the predicted annotation probabilities
        according to the article [1].

        Parameters:
        -----------
        z : torch.tensor of shape (n_samples, n_annotators)
            Annotations, where `z[i,j]=c` indicates that annotator `j` provided class label `c` for sample
            `i`. A missing label is denoted by `c=-1`.
        p_annot_log : torch.tensor of shape (n_samples, n_annotators, n_classes)
            Estimated annotation log-probabilities, where `p_annot_log[i,j,c]` refers to the estimated log-probability
            for sample `i`, annotator `j`, and class `c`.

        Returns
        -------
        loss : torch.Float
            Computed cross-entropy loss.

        References
        ----------
        [1] Herde, M., Lührs, L., Huseljic, D., & Sick, B. (2024). Annot-Mix: Learning with Noisy Class Labels from
            Multiple Annotators via a Mixup Extension. arXiv:2405.03386.
        """
        # Compute number of non-zero annotations.
        lmbda = z.sum(dim=-1)
        is_missing = lmbda == 0
        n_labels = (~is_missing).sum().float().item()

        # Compute cross-entropy loss.
        loss = -(z * p_annot_log).sum() / n_labels

        return loss

    @torch.inference_mode()
    def predict_step(self, batch: Dict[str, torch.tensor], batch_idx: int, dataloader_idx: Optional[int] = 0):
        """
        Computes the GT and (optionally) AP models' predictions.

        Parameters
        ----------
        batch : dict
            Data batch fitting the dictionary structure of `maml.data.MultiAnnotatorDataset`.
        batch_idx : int
            Index of the batch in the dataset.
        dataloader_idx : int, default=0
            Index of the used dataloader.

        Returns
        -------
        predictions : dict
            A dictionary of predictions fitting the expected structure of `maml.classifiers.MaMLClassifier`.
        """
        self.eval()
        x = batch["x"]
        a = batch.get("a", None)
        a = a[0] if a is not None else None
        output = self.network(x=x, a=a)
        if a is None:
            return {"p_class": F.softmax(output, dim=-1)}
        else:
            p_class_log = F.log_softmax(output[0], dim=-1)
            p_confusion_log = F.log_softmax(output[1], dim=-1)
            p_confusion_log = p_confusion_log.reshape((len(x), len(a), p_class_log.shape[1], p_class_log.shape[1]))
            p_perf = (p_class_log[:, None, :, None] + p_confusion_log).exp().diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            p_annot = torch.logsumexp(p_class_log[:, None, :, None] + p_confusion_log, dim=2).exp()
            return {
                "p_class": p_class_log.exp(),
                "p_perf": p_perf,
                "p_conf": p_confusion_log.exp(),
                "p_annot": p_annot
            }

    @torch.no_grad()
    def get_gt_parameters(self):
        """
        Returns the list of parameters of the GT model.

        Returns
        -------
        gt_parameters : list
            The list of the GT models' parameters.
        """
        gt_parameters = list(self.network.gt_embed_x.parameters())
        gt_parameters += list(self.network.gt_output.parameters())
        return gt_parameters

    @torch.no_grad()
    def get_ap_parameters(self):
        """
        Returns the list of parameters of the AP model.

        Returns
        -------
        ap_parameters : list
            The list of the AP models' parameters.
        """
        ap_parameters = list(self.network.ap_embed_x.parameters())
        ap_parameters += list(self.network.ap_embed_a.parameters())
        ap_parameters += list(self.network.ap_hidden.parameters())
        ap_parameters += list(self.network.ap_output.parameters())
        return ap_parameters
