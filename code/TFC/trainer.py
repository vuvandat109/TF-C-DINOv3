"""Train and evaluate TF-C + DINOv3 without test-set model selection.

Pre-train on source only, fine-tune on a disjoint target train partition,
select the checkpoint by validation macro-F1, then evaluate the held-out
target test exactly once. Metrics are aggregated over all samples.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
)
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.nn import functional as F

from dinov3_loss import DINOv3TFCLoss


def _objective(config):
    return DINOv3TFCLoss(
        temperature=getattr(config, "dinov3_loss_temperature", 0.2),
        time_weight=getattr(config, "dinov3_loss_time_weight", 0.2),
        frequency_weight=getattr(config, "dinov3_loss_frequency_weight", 0.2),
        cross_weight=getattr(config, "dinov3_loss_cross_weight", 1.0),
    )


def _forward(model, time_data, frequency_waveform):
    _, z_time, _, z_frequency = model(time_data, frequency_waveform)
    return torch.cat([z_time, z_frequency], dim=1), z_time, z_frequency


def _epoch_pretrain(model, optimizer, loader, device, config):
    model.train()
    objective = _objective(config)
    losses = []
    for signal, _, augmented_time, freq_waveform, augmented_freq in loader:
        if len(signal) < 2:
            continue
        signal = signal.to(device)
        augmented_time = augmented_time.to(device)
        freq_waveform = freq_waveform.to(device)
        augmented_freq = augmented_freq.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, z_t, z_f = _forward(model, signal, freq_waveform)
        _, z_t_aug, z_f_aug = _forward(model, augmented_time, augmented_freq)
        loss, _ = objective(z_t, z_t_aug, z_f, z_f_aug)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    if not losses:
        raise ValueError("Pretraining requires batches containing at least two samples")
    return float(np.mean(losses))


def _evaluate(model, classifier, loader, device, expected_classes):
    model.eval()
    classifier.eval()
    all_labels, all_predictions, all_probabilities, all_embeddings = [], [], [], []
    total_loss, total_samples = 0.0, 0
    with torch.inference_mode():
        for signal, label, _, freq_waveform, _ in loader:
            signal, freq_waveform = signal.to(device), freq_waveform.to(device)
            label = label.long().to(device)
            embeddings, _, _ = _forward(model, signal, freq_waveform)
            logits = classifier(embeddings)
            total_loss += float(F.cross_entropy(logits, label, reduction="sum"))
            total_samples += len(label)
            all_labels.append(label.cpu().numpy())
            all_predictions.append(logits.argmax(1).cpu().numpy())
            all_probabilities.append(logits.softmax(dim=1).cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())
    if total_samples == 0:
        raise ValueError("Evaluation loader is empty")
    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities)
    embeddings = np.concatenate(all_embeddings)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(expected_classes),
        average="macro", zero_division=0,
    )
    result = {
        "loss": total_loss / total_samples,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "samples": int(total_samples),
    }
    # Binary/multiclass probability metrics need all expected classes to
    # occur in the evaluation split. Report null otherwise, not fake zeros.
    if set(np.unique(labels)) == set(range(expected_classes)):
        try:
            if expected_classes == 2:
                result["auroc"] = float(roc_auc_score(labels, probabilities[:, 1]))
                result["auprc"] = float(average_precision_score(labels, probabilities[:, 1]))
            else:
                onehot = np.eye(expected_classes)[labels]
                result["auroc"] = float(roc_auc_score(
                    labels, probabilities, multi_class="ovr", average="macro"
                ))
                result["auprc"] = float(average_precision_score(
                    onehot, probabilities, average="macro"
                ))
        except ValueError:
            result["auroc"] = None
            result["auprc"] = None
    else:
        result["auroc"] = None
        result["auprc"] = None
    return result, labels, predictions, probabilities, embeddings


def _epoch_finetune(model, classifier, model_optimizer, classifier_optimizer,
                    loader, device, config):
    model.train()
    classifier.train()
    objective = _objective(config)
    total, samples = 0.0, 0
    for signal, labels, augmented_time, freq_waveform, augmented_freq in loader:
        signal = signal.to(device)
        freq_waveform = freq_waveform.to(device)
        labels = labels.long().to(device)
        augmented_time = augmented_time.to(device)
        augmented_freq = augmented_freq.to(device)
        model_optimizer.zero_grad(set_to_none=True)
        classifier_optimizer.zero_grad(set_to_none=True)

        embeddings, z_t, z_f = _forward(model, signal, freq_waveform)
        logits = classifier(embeddings)
        loss = F.cross_entropy(logits, labels)
        ssl_weight = float(getattr(config, "dinov3_finetune_ssl_weight", 0.1))
        if ssl_weight and len(labels) >= 2:
            _, z_t_aug, z_f_aug = _forward(model, augmented_time, augmented_freq)
            ssl_loss, _ = objective(z_t, z_t_aug, z_f, z_f_aug)
            loss = loss + ssl_weight * ssl_loss
        loss.backward()
        model_optimizer.step()
        classifier_optimizer.step()
        total += float(loss.detach()) * len(labels)
        samples += len(labels)
    if samples == 0:
        raise ValueError("Empty fine-tuning loader")
    return total / samples


def _store_report(output_dir, history, report, labels, predictions, n_classes):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        if history:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    matrix = confusion_matrix(labels, predictions, labels=list(range(n_classes)))
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "predicted_label"])
        writer.writerows(zip(labels.tolist(), predictions.tolist()))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if history:
            fig, ax = plt.subplots()
            epochs = [item["epoch"] for item in history]
            ax.plot(epochs, [item["train_loss"] for item in history], label="Train loss")
            ax.plot(epochs, [item["validation_loss"] for item in history], label="Validation loss")
            ax.set(xlabel="Epoch", ylabel="Loss", title="Fine-tuning loss")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "loss_curve.png", dpi=150)
            plt.close(fig)
            fig, ax = plt.subplots()
            ax.plot(epochs, [item["validation_f1_macro"] for item in history])
            ax.set(xlabel="Epoch", ylabel="Macro F1", title="Validation F1")
            fig.tight_layout()
            fig.savefig(output_dir / "validation_f1.png", dpi=150)
            plt.close(fig)
        fig, ax = plt.subplots()
        im = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set(xlabel="Predicted", ylabel="True", title="Held-out test confusion matrix")
        fig.tight_layout()
        fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("Optional plots skipped: install matplotlib to generate PNG files")


def Trainer(model, model_optimizer, classifier, classifier_optimizer,
            pretrain_dl, fine_train_dl, validation_dl, test_dl,
            device, logger, config, experiment_log_dir, training_mode):
    output = Path(experiment_log_dir)
    output.mkdir(parents=True, exist_ok=True)
    n_classes = int(config.num_classes_target)
    epochs = int(config.num_epoch)

    if training_mode == "pre_train":
        pretraining_history = []
        for epoch in range(1, epochs + 1):
            loss = _epoch_pretrain(model, model_optimizer, pretrain_dl, device, config)
            pretraining_history.append({"epoch": epoch, "loss": loss})
            logger.debug(f"Pretraining {epoch}/{epochs}: loss={loss:.5f}")
            print(f"Pretraining epoch {epoch}/{epochs}: loss={loss:.5f}")
        saved = output / "saved_models"
        saved.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, saved / "ckp_last.pt")
        (output / "pretrain_history.json").write_text(
            json.dumps(pretraining_history, indent=2), encoding="utf-8"
        )
        print(f"Saved pretrained checkpoint: {saved / 'ckp_last.pt'}")
        return
    if training_mode != "fine_tune_test":
        raise ValueError(f"Unknown training_mode: {training_mode}")

    best_f1, best_loss, best_epoch = -1.0, float("inf"), 0
    checkpoint = output / "saved_models" / "best_validation.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = _epoch_finetune(
            model, classifier, model_optimizer, classifier_optimizer,
            fine_train_dl, device, config,
        )
        val_metrics, _, _, _, _ = _evaluate(
            model, classifier, validation_dl, device, n_classes
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_metrics["loss"],
            "validation_accuracy": val_metrics["accuracy"],
            "validation_f1_macro": val_metrics["f1_macro"],
        })
        f1, val_loss = val_metrics["f1_macro"], val_metrics["loss"]
        if f1 > best_f1 + 1e-10 or (
            abs(f1 - best_f1) <= 1e-10 and val_loss < best_loss
        ):
            best_f1, best_loss, best_epoch = f1, val_loss, epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "epoch": epoch, "validation_f1_macro": f1,
            }, checkpoint)
        print(
            f"Fine-tune {epoch}/{epochs}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_F1={f1:.4f}"
        )
    # No test call occurs anywhere inside the training/validation loop.
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    classifier.load_state_dict(state["classifier_state_dict"])
    test_metrics, y_true, y_pred, _, test_emb = _evaluate(
        model, classifier, test_dl, device, n_classes
    )
    report = {
        "model": "TF-C + DINOv3",
        "selected_epoch": best_epoch,
        "checkpoint_selection": "validation_macro_f1_then_loss",
        "best_validation_f1_macro": best_f1,
        "test": test_metrics,
        "warning": "Do not interpret subset test scores as a full-dataset benchmark.",
    }

    # KNN uses representations of the fine-tune TRAIN set only.
    _, train_labels, _, _, train_emb = _evaluate(
        model, classifier, fine_train_dl, device, n_classes
    )
    if len(train_labels) >= 2:
        neighbor_count = min(5, len(train_labels))
        neighbor = KNeighborsClassifier(n_neighbors=neighbor_count)
        neighbor.fit(train_emb, train_labels)
        knn_preds = neighbor.predict(test_emb)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, knn_preds, labels=np.arange(n_classes),
            average="macro", zero_division=0,
        )
        report["knn_test"] = {
            "accuracy": float(accuracy_score(y_true, knn_preds)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "neighbors": neighbor_count,
        }
    _store_report(output, history, report, y_true, y_pred, n_classes)
    print(
        f"FINAL HELD-OUT TEST: accuracy={test_metrics['accuracy']:.4f} "
        f"macro_F1={test_metrics['f1_macro']:.4f}, "
        f"best_val_epoch={best_epoch}"
    )
    logger.debug(f"Final held-out test: {json.dumps(report)}")
    print(f"Saved evaluation files in: {output}")
