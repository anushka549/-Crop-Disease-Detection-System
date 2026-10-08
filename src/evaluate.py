"""Evaluate saved model(s) on the held-out test split."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from data_loader import load_datasets, PROJECT_ROOT
import tensorflow as tf

def evaluate_model(model_path, test_ds, class_names, results_dir, suffix=""):
    model = tf.keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(test_ds, verbose=1)
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=range(len(class_names)), target_names=class_names, zero_division=0, digits=4)
    summary = {"model": model_path.name, "loss": float(loss), "accuracy": float(accuracy), "precision": float(precision), "recall": float(recall), "f1": float(f1)}
    prefix = f"{suffix}_" if suffix else ""
    report_text = (f"Model: {model_path.name}\nTest loss: {loss:.4f}\nTest accuracy: {accuracy:.4f}\nWeighted precision: {precision:.4f}\nWeighted recall: {recall:.4f}\nWeighted F1-score: {f1:.4f}\n\nClass-wise report\n{report}")
    (results_dir / f"{prefix}classification_report.txt").write_text(report_text, encoding="utf-8")
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)), xticklabels=class_names, yticklabels=class_names, xlabel="Predicted class", ylabel="True class", title=f"Test confusion matrix: {model_path.stem}")
    plt.setp(ax.get_xticklabels(), rotation=75, ha="right", rotation_mode="anchor", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()
    fig.savefig(results_dir / f"{prefix}confusion_matrix.png", dpi=180)
    plt.close(fig)
    return summary, report

def main():
    models_dir = PROJECT_ROOT / "models"
    best = models_dir / "best_crop_disease_model.keras"
    final = models_dir / "final_crop_disease_model.keras"
    if not best.exists() and not final.exists():
        raise FileNotFoundError("No trained model found. Run python src/train.py first.")
    _, _, test_ds, class_names, _ = load_datasets(verbose=False)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    summaries = []
    if best.exists():
        summary, report = evaluate_model(best, test_ds, class_names, results_dir, "initial" if final.exists() else "")
        summaries.append(summary)
        if not final.exists(): print(report)
    if final.exists():
        summary, report = evaluate_model(final, test_ds, class_names, results_dir)
        summaries.append(summary)
        print(report)
    comparison = "\n".join(f"{item['model']}: accuracy={item['accuracy']:.4f}, precision={item['precision']:.4f}, recall={item['recall']:.4f}, F1={item['f1']:.4f}" for item in summaries)
    print("Test-set model comparison (interpret after both evaluations):\n" + comparison)
    (results_dir / "evaluation_summary.txt").write_text("Held-out test comparison\n" + comparison + "\n", encoding="utf-8")
    print(f"Reports and confusion matrices saved in {results_dir}")

if __name__ == "__main__":
    main()
