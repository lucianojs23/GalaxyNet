# visualization.py - Funções de visualização, Grad-CAM e plotagem

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def compute_gradcam(model, img_array, layer_name, class_idx=None):
    """
    Computes Grad-CAM heatmap for a given image and target convolutional layer.

    Args:
        model      (keras.Model): Trained Keras model (Functional API).
        img_array  (np.ndarray): Single image array with shape (H, W, C).
        layer_name (str): Name of the target convolutional layer.
        class_idx  (int or None): Target class index. If None, uses the
                                  predicted class.

    Returns:
        np.ndarray: Grad-CAM heatmap with shape (H, W), values in [0, 1].
    """
    img_tensor = tf.expand_dims(tf.cast(img_array, tf.float32), axis=0)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def compute_gradcam_hybrid(model, tabular_input, img_array, layer_name,
                           class_idx=None):
    """
    Computes Grad-CAM heatmap for a hybrid (dual-input) model.

    Args:
        model          (keras.Model): Trained hybrid Keras model.
        tabular_input  (np.ndarray): Single tabular feature vector, shape (n_features,).
        img_array      (np.ndarray): Single image array, shape (H, W, C).
        layer_name     (str): Name of the target convolutional layer.
        class_idx      (int or None): Target class index. If None, uses predicted.

    Returns:
        np.ndarray: Grad-CAM heatmap with shape (H, W), values in [0, 1].
    """
    tab_tensor = tf.expand_dims(tf.cast(tabular_input, tf.float32), axis=0)
    img_tensor = tf.expand_dims(tf.cast(img_array, tf.float32), axis=0)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([tab_tensor, img_tensor])
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def plot_gradcam_grid(model, X_tabular, X_images, y_true_labels,
                      layer_name, label_encoder, n_per_class=2,
                      save_path=None):
    """
    Plots a grid of Grad-CAM overlays for the hybrid model, grouped by class.

    Args:
        model          (keras.Model): Trained hybrid model.
        X_tabular      (np.ndarray): Tabular test features, shape (N, n_features).
        X_images       (np.ndarray): Image test data, shape (N, H, W, C).
        y_true_labels  (np.ndarray): True string labels, shape (N,).
        layer_name     (str): Target conv layer name for Grad-CAM.
        label_encoder  (LabelEncoder): Fitted LabelEncoder.
        n_per_class    (int): Number of examples per class to show.
        save_path      (str or None): If provided, saves figure to this path.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    classes = label_encoder.classes_
    n_classes = len(classes)

    fig, axes = plt.subplots(n_classes, n_per_class * 2, figsize=(4 * n_per_class * 2, 4 * n_classes))
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    for i, cls in enumerate(classes):
        idxs = np.where(y_true_labels == cls)[0]
        selected = idxs[:n_per_class] if len(idxs) >= n_per_class else idxs

        for j in range(n_per_class):
            col_orig = j * 2
            col_cam = j * 2 + 1

            if j < len(selected):
                idx = selected[j]
                img = X_images[idx]

                # RGB composite (g, r, i → R=r, G=g, B=i)
                rgb = np.stack([img[:, :, 1], img[:, :, 0], img[:, :, 2]], axis=-1)
                rgb = np.clip(rgb, 0, 1)

                # Grad-CAM
                heatmap = compute_gradcam_hybrid(
                    model, X_tabular[idx], img, layer_name,
                )
                heatmap_resized = tf.image.resize(
                    heatmap[..., np.newaxis], (img.shape[0], img.shape[1])
                ).numpy().squeeze()

                # Original image
                axes[i, col_orig].imshow(rgb, origin='lower')
                axes[i, col_orig].set_title(f'{cls}', fontsize=11, fontweight='bold')
                axes[i, col_orig].axis('off')

                # Grad-CAM overlay
                axes[i, col_cam].imshow(rgb, origin='lower')
                axes[i, col_cam].imshow(heatmap_resized, cmap='jet', alpha=0.4,
                                        origin='lower')
                axes[i, col_cam].set_title('Grad-CAM', fontsize=11)
                axes[i, col_cam].axis('off')
            else:
                axes[i, col_orig].axis('off')
                axes[i, col_cam].axis('off')

    fig.suptitle('Grad-CAM — Regiões de Atenção do Modelo Híbrido',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Grad-CAM figure salva em: {save_path}")

    return fig
