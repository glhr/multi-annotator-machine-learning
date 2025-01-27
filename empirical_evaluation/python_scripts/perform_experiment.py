import os
import hydra
import sys
import warnings
import mlflow
import atexit
import signal
import torch
import numpy as np
import pandas as pd

from hydra.utils import instantiate, get_class, to_absolute_path
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from mlflow import get_experiment_by_name, set_tracking_uri, log_metric, start_run, create_experiment
from omegaconf.errors import ConfigAttributeError
from skactiveml.utils import majority_vote, compute_vote_vectors, rand_argmax
from torch import set_float32_matmul_precision
from torch.utils.data import DataLoader

# TODO: In case of issues, set the absolute path to the directory of the mult-annotator-machine-learning project.
sys.path.append("../../")
warnings.filterwarnings("ignore")
set_float32_matmul_precision("medium")
torch.multiprocessing.set_sharing_strategy('file_system')

# TODO: In case of issues, set the absolute path to the directory of hydra configuration files.
@hydra.main(config_path="../hydra_configs", config_name="experiment", version_base=None)
def evaluate(cfg):
    from maml.architectures import maml_net_params, gt_net
    from maml.data import SSLDatasetWrapper
    from maml.utils import log_params_from_omegaconf_dict

    # Setup experiment.
    set_tracking_uri(uri=f"file://{to_absolute_path(cfg.mlruns_path)}")
    exp = get_experiment_by_name(cfg.experiment_name)
    experiment_id = create_experiment(name=cfg.experiment_name) if exp is None else exp.experiment_id

    print(cfg.data.class_definition._target_)
    print(cfg.classifier.name)

    with start_run(experiment_id=experiment_id):
        # Get path to artifacts and ensure that artifact is deleted after program termination or cancellation.
        artifacts_path = mlflow.active_run().info.artifact_uri.split("file://")[1]
        def cleanup_function():
            if os.path.exists(artifacts_path):
                try:
                    os.remove(os.path.join(artifacts_path, "best.ckpt"))
                except FileNotFoundError:
                    pass
        atexit.register(cleanup_function)
        def signal_handler(sig, frame):
            cleanup_function()
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)

        # Log configuration.
        log_params_from_omegaconf_dict(cfg)

        # Set seed for deterministic results.
        seed_everything(cfg.seed, workers=True)

        # Load data.
        ds_train = instantiate(
            cfg.data.class_definition,
            version="train",
            annotators=cfg.classifier.annotators,
            aggregation_method=cfg.classifier.aggregation_method,
        )
        ds_valid = instantiate(cfg.data.class_definition, version="valid", annotators=cfg.classifier.annotators)
        if "n_annotations_per_sample" in cfg.data.class_definition:
            cfg.data.class_definition["n_annotations_per_sample"] = -1
        ds_test = instantiate(cfg.data.class_definition, version="test", annotators=cfg.classifier.annotators)
        class_definition = cfg.data.class_definition.copy()
        if "variant" in cfg.data.class_definition:
            class_definition["variant"] = "full"
        ds_train_eval = instantiate(
            class_definition,
            annotators=cfg.classifier.annotators,
            transform=ds_test.transform if ds_test.transform else "auto",
            aggregation_method=cfg.classifier.aggregation_method,
        )

        # Set embedding dimension for AP architectures.
        try:
            embed_size = cfg.classifier.embed_size
        except ConfigAttributeError:
            embed_size = None

        # Build classifier architectures depending on the dataset.
        if cfg.data.lr_scheduler.class_definition is not None:
            lr_scheduler = get_class(cfg.data.lr_scheduler.class_definition)
        else:
            lr_scheduler = None
        gt_params_dict = cfg.architecture.params if cfg.architecture.params is not None else {}
        params_dict = maml_net_params(
            gt_name=cfg.architecture.name,
            gt_params_dict={"n_classes": ds_train.get_n_classes(), **gt_params_dict},
            classifier_name=cfg.classifier.name,
            n_annotators=ds_train.get_n_annotators(),
            annotators=ds_train.get_annotators(),
            classifier_specific=cfg.classifier.params,
            optimizer=get_class(cfg.data.optimizer.class_definition),
            optimizer_gt_dict=cfg.data.optimizer.gt_params,
            optimizer_ap_dict=cfg.data.optimizer.ap_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_dict=cfg.data.lr_scheduler.params,
            embed_size=embed_size,
        )
        clf = instantiate(cfg.classifier.class_definition, **params_dict)

        # If desired, use SSL features.
        if cfg.ssl_model.name is not None:
            ssl_params_dict = cfg.ssl_model.params if cfg.ssl_model.params is not None else {}
            ssl_model, _, _ = gt_net(cfg.ssl_model.name, {"n_classes": ds_train.get_n_classes(), **ssl_params_dict})
            ssl_model = ssl_model()
            ds_train = instantiate(
                cfg.data.class_definition,
                annotators=cfg.classifier.annotators,
                transform=ds_test.transform if ds_test.transform else "auto",
                aggregation_method=cfg.classifier.aggregation_method,
            )
            device = "cuda" if cfg.accelerator == "gpu" else "cpu"
            ds_train = SSLDatasetWrapper(dataset=ds_train, model=ssl_model, cache=True, device=device)
            ds_train_eval = SSLDatasetWrapper(dataset=ds_train_eval, model=ssl_model, cache=True, device=device)
            ds_valid = SSLDatasetWrapper(dataset=ds_valid, model=ssl_model, cache=True, device=device)
            ds_test = SSLDatasetWrapper(dataset=ds_test, model=ssl_model, cache=True, device=device)

        # Build data loaders.
        dl_train = DataLoader(
            dataset=ds_train,
            batch_size=cfg.data.train_batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=True,
            drop_last=True
        )
        dl_train_eval = DataLoader(
            dataset=ds_train_eval, batch_size=cfg.data.eval_batch_size, num_workers=cfg.data.num_workers
        )
        dl_valid = DataLoader(dataset=ds_valid, batch_size=cfg.data.eval_batch_size, num_workers=cfg.data.num_workers)
        dl_test = DataLoader(dataset=ds_test, batch_size=cfg.data.eval_batch_size, num_workers=cfg.data.num_workers)

        # Create callbacks for progressbar and checkpointing.
        bar = RichProgressBar()
        checkpoint = ModelCheckpoint(
            monitor="gt_val_acc",
            dirpath=artifacts_path,
            filename="best",
            mode="max",
            save_top_k=1,
            save_last=False,
        )

        # Train multi-annotator classifier.
        trainer = Trainer(
            max_epochs=cfg.data.max_epochs,
            accelerator=cfg.accelerator,
            logger=False,
            callbacks=[bar, checkpoint],
            deterministic="warn",
        )
        trainer.fit(model=clf, train_dataloaders=dl_train)

        # Evaluate multi-annotator classifier after the last epoch.
        n_classes = ds_train.get_n_classes()
        classes = np.arange(n_classes)
        dl_list = [("train", dl_train_eval), ("valid", dl_valid), ("test", dl_test)]
        device = "cuda" if cfg.accelerator == "gpu" else "cpu"
        for state, mdl in zip(["last"], [clf]):
            print(f"\n############ {state} ############")
            mdl.to(device)
            mdl.eval()
            for version, dl in dl_list:
                eval_scores_dict = {}
                normalizer_dict = {}
                # =================================== Collect data and predictions. ===================================
                for batch_idx, batch in enumerate(dl):

                    # Helper function for loss computation.
                    loss_name_list = ["zero_one_loss", "brier_score", "log_loss"]

                    def compute_losses(probas, targets, target_name, weights=None):
                        for loss_name in loss_name_list:
                            # Compute sample-wise losses.
                            if loss_name == "zero_one_loss":
                                preds = rand_argmax(probas, axis=-1, random_state=batch_idx)
                                preds = np.eye(n_classes)[preds]
                                loss = (1 - np.einsum("ic,ic->i", targets, preds))
                            elif loss_name == "brier_score":
                                loss = ((targets - probas) ** 2).sum(axis=-1)
                            elif loss_name == "log_loss":
                                loss = (-targets * np.log(probas + np.finfo(np.float64).eps)).sum(axis=-1)
                            else:
                                raise NotImplementedError(f"`{loss_name}` is not implemented.")

                            # Sum losses.
                            if weights is not None:
                                loss = (loss * weights).sum()
                                normalizer = weights.sum()
                            else:
                                loss = loss.sum()
                                normalizer = len(targets)

                            # Log losses.
                            loss_key = f"{target_name}_{loss_name}_{state}_{version}"
                            eval_scores_dict[loss_key] = eval_scores_dict.get(loss_key, 0) + loss
                            normalizer_dict[loss_key] = normalizer_dict.get(loss_key, 0) + normalizer

                    # Compute predictions.
                    batch = {k: v.to(device) for k, v in batch.items()}
                    pred_dict = mdl.predict_step(batch=batch, batch_idx=batch_idx)
                    batch = {k: v.cpu().numpy() for k, v in batch.items()}
                    pred_dict = {k: v.cpu().numpy() for k, v in pred_dict.items()}

                    # Compute predictions and targets.
                    p_class = pred_dict["p_class"]
                    p_annot = None
                    targets_dict = {"class_true": np.eye(n_classes)[batch["y"]]}
                    weights_dict = {}
                    z = batch.get("z", None)
                    if z is not None:
                        # Define one-hot annotations.
                        z_ravel = z.ravel()
                        is_labeled = z_ravel != -1
                        z_ravel = z_ravel[is_labeled]
                        z_ravel_one_hot = np.eye(n_classes)[z_ravel]

                        # Get annotation predictions.
                        p_annot = pred_dict["p_annot"].reshape((-1, n_classes))[is_labeled]

                        # Get annotator' performances as weights.
                        p_unif = np.ones_like(z)
                        p_perf = pred_dict.get("p_perf", None)
                        p_conf = pred_dict.get("p_conf", None)
                        if p_conf is not None:
                            p_conf_diag = p_conf.diagonal(axis1=-2, axis2=-1).mean(axis=-1)
                        for suffix, weights in zip(["unif", "perf", "conf"], [p_unif, p_perf, p_conf_diag]):
                            is_unif = suffix == "unif"
                            if weights is None:
                                continue

                            # Use hard majority votes as targets.
                            agg = f"class_mv_{suffix}"
                            targets_dict[agg] = majority_vote(
                                y=z,
                                w=weights,
                                missing_label=-1,
                                random_state=batch_idx
                            )
                            targets_dict[agg] = np.eye(n_classes)[targets_dict[agg]]
                            if not is_unif:
                                weights_dict[agg] = weights.mean(axis=-1)

                            # Use hard majority votes as targets.
                            agg = f"class_smv_{suffix}"
                            targets_dict[agg] = compute_vote_vectors(
                                y=z,
                                w=pred_dict["p_perf"],
                                missing_label=-1,
                                classes=classes
                            )
                            targets_dict[agg] = targets_dict[agg] / targets_dict[agg].sum(axis=-1, keepdims=True)
                            if not is_unif:
                                weights_dict[agg] = weights.mean(axis=-1)

                            # Use noisy annotations as targets.
                            agg = f"annot_{suffix}"
                            targets_dict[agg] = z_ravel_one_hot
                            if not is_unif:
                                weights_dict[agg] = weights.ravel()[is_labeled]

                        # Use corrected class labels as targets.
                        if p_conf is not None:
                            p_conf_log = np.log(p_conf)
                            z_one_hot = np.eye(n_classes + 1)[z + 1][:, :, 1:]
                            p_bayes_log = (p_conf_log * z_one_hot[:, :, None, :]).sum(axis=(1, 3)) #+ np.log(p_class)
                            p_bayes = np.exp(p_bayes_log) / np.exp(p_bayes_log).sum(axis=-1, keepdims=True)
                            targets_dict["class_soft_bayes"] = p_bayes
                            one_hot_bayes = np.eye(n_classes)[rand_argmax(p_bayes, random_state=batch_idx, axis=-1)]
                            targets_dict["class_hard_bayes"] = one_hot_bayes

                    # Evaluate predictions.
                    for k, v in targets_dict.items():
                        probas = p_class if k.startswith("class") else p_annot
                        compute_losses(probas=probas, targets=v, target_name=k)
                        w = weights_dict.get(k, None)
                        if w is not None:
                            compute_losses(probas=probas, targets=v, target_name=f"{k}_weights", weights=w)

                # Normalize evaluation scores.
                for k, v in eval_scores_dict.items():
                    eval_scores_dict[k] = [eval_scores_dict[k] / normalizer_dict[k]]
                    log_metric(k, eval_scores_dict[k])
                eval_df = pd.DataFrame(eval_scores_dict).T
                eval_df.columns = ["value"]
                eval_df.index.name = version
                for loss_name in loss_name_list:
                    is_loss = [l for l in eval_df.index if loss_name in l]
                    print(f"{eval_df.loc[is_loss].to_markdown(tablefmt='github', floatfmt='.4f')}")
                    print("\n")


if __name__ == "__main__":
    evaluate()
