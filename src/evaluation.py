# evaluation.py - Funções de avaliação e métricas científicas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_galaxy_classifier(model, X_test_data, y_test_one_hot,
                                label_encoder, model_type='MLP',
                                save_dir=None):
    """
    Evaluates the galaxy classification model and prints scientific metrics.

    Gera:
        - Classification Report (precision, recall, f1-score)
        - Confusion Matrix (heatmap)
        - Métricas científicas: Completeness (Recall) e Reliability (Precision)
          por classe morfológica

    Args:
        model           (keras.Model): The trained Keras model.
        X_test_data     (np.ndarray or list): Test features (tabular, images, or both).
        y_test_one_hot  (np.ndarray): One-hot encoded true labels.
        label_encoder   (LabelEncoder): Fitted LabelEncoder object.
        model_type      (str): Type of model ('MLP', 'CNN', 'Hybrid').
        save_dir        (str): Directory to save the confusion matrix figure.
                               If None, figure is shown but not saved.

    Returns:
        dict: Dictionary with keys 'classification_report' (str),
              'confusion_matrix' (np.ndarray), 'scientific_metrics' (pd.DataFrame),
              'y_pred_labels' (np.ndarray), 'y_true_labels' (np.ndarray).
    """
    # Get predictions
    y_pred_one_hot = model.predict(X_test_data, verbose=0)
    y_pred_labels = label_encoder.inverse_transform(np.argmax(y_pred_one_hot, axis=1))
    y_true_labels = label_encoder.inverse_transform(np.argmax(y_test_one_hot, axis=1))

    class_names = label_encoder.classes_

    # ── Classification Report ────────────────────────────────────────────────
    print(f"\n--- Evaluation for {model_type} Model ---")
    report_str = classification_report(y_true_labels, y_pred_labels,
                                       target_names=class_names)
    print("\nClassification Report:")
    print(report_str)

    # ── Confusion Matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=class_names)

    fig_cm, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'Confusion Matrix — {model_type}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()

    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir,
                                f'fig17_{model_type.lower()}_confusion_matrix.png')
        fig_cm.savefig(fig_path, bbox_inches='tight')
        print(f"\nConfusion matrix salva em: {fig_path}")

    plt.show()

    # ── Scientific Metrics (Completeness & Reliability) ──────────────────────
    results = []
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        completeness = tp / (tp + fn) if (tp + fn) > 0 else 0
        reliability = tp / (tp + fp) if (tp + fp) > 0 else 0

        results.append({
            'Class': cls,
            'N_True': cm[i, :].sum(),
            'Completeness (Recall)': f'{completeness:.3f}',
            'Reliability (Precision)': f'{reliability:.3f}',
        })

    df_metrics = pd.DataFrame(results)
    print("\nScientific Metrics (Completeness and Reliability per Class):")
    print(df_metrics.to_string(index=False))

    return {
        'classification_report': report_str,
        'confusion_matrix': cm,
        'scientific_metrics': df_metrics,
        'y_pred_labels': y_pred_labels,
        'y_true_labels': y_true_labels,
    }
