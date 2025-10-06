"""
Evaluation script for waste classification model
Evaluates model on test dataset and generates detailed metrics
"""
import os
import numpy as np
import torch
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm

import config
from data_preprocessing import load_dataset_from_directory, create_data_loader
from predict import WasteClassifier


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_per_class_metrics(labels, precision, recall, f1, save_path):
    """Plot per-class performance metrics"""
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label='Precision', color='#667eea')
    ax.bar(x, recall, width, label='Recall', color='#764ba2')
    ax.bar(x + width, f1, width, label='F1-Score', color='#f093fb')
    
    ax.set_xlabel('Waste Category', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Per-class metrics plot saved to {save_path}")
    plt.close()


def evaluate_model():
    """Main evaluation function"""
    print("=" * 60)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 60)
    
    # Check if test data exists
    test_images, test_labels_str = load_dataset_from_directory(config.TEST_DIR)
    
    if len(test_images) == 0:
        print("\nNo test data found!")
        print(f"Please add test images to: {config.TEST_DIR}")
        return
    
    print(f"\nTest samples: {len(test_images)}")
    
    # Initialize classifier
    print("\nLoading trained model...")
    try:
        classifier = WasteClassifier()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first using train_model.py")
        return
    
    # Make predictions
    print("\nMaking predictions on test set...")
    y_true = []
    y_pred = []
    y_probs = []
    
    for img_path, true_label in tqdm(zip(test_images, test_labels_str), 
                                     total=len(test_images)):
        try:
            result = classifier.predict(img_path, return_probabilities=True)
            y_true.append(true_label)
            y_pred.append(result['predicted_class'])
            y_probs.append(list(result['probabilities'].values()))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=classifier.label_encoder.classes_
    )
    
    print("\nPer-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    
    for i, class_name in enumerate(classifier.label_encoder.classes_):
        print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} "
              f"{f1[i]:<12.4f} {support[i]:<10.0f}")
    
    # Average metrics
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)
    
    print("-" * 60)
    print(f"{'Average':<15} {avg_precision:<12.4f} {avg_recall:<12.4f} "
          f"{avg_f1:<12.4f} {np.sum(support):<10.0f}")
    print("-" * 60)
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, 
                               target_names=classifier.label_encoder.classes_))
    
    # Save confusion matrix
    cm_path = os.path.join(config.RESULTS_DIR, 'test_confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, classifier.label_encoder.classes_, cm_path)
    
    # Save per-class metrics plot
    metrics_path = os.path.join(config.RESULTS_DIR, 'test_per_class_metrics.png')
    plot_per_class_metrics(classifier.label_encoder.classes_, 
                          precision, recall, f1, metrics_path)
    
    # Save evaluation results
    results = {
        'overall_accuracy': float(accuracy),
        'average_precision': float(avg_precision),
        'average_recall': float(avg_recall),
        'average_f1': float(avg_f1),
        'per_class_metrics': {}
    }
    
    for i, class_name in enumerate(classifier.label_encoder.classes_):
        results['per_class_metrics'][class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    results_path = os.path.join(config.RESULTS_DIR, 'test_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nEvaluation results saved to {results_path}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()
