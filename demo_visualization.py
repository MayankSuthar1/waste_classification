"""
Demo script to visualize predictions
Shows sample predictions with images
"""
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from predict import WasteClassifier
import config


def visualize_prediction(image_path, classifier):
    """
    Visualize a single prediction with image and results
    
    Args:
        image_path: Path to image
        classifier: WasteClassifier instance
    """
    # Make prediction
    result = classifier.predict(image_path, return_probabilities=True)
    
    # Load image
    img = Image.open(image_path)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Display image
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f'Input Image\n{os.path.basename(image_path)}', fontsize=12)
    
    # Display probabilities
    classes = list(result['probabilities'].keys())
    probs = list(result['probabilities'].values())
    colors = ['#667eea' if c == result['predicted_class'] else '#ccc' 
              for c in classes]
    
    ax2.barh(classes, probs, color=colors)
    ax2.set_xlabel('Probability', fontsize=11)
    ax2.set_title(f'Prediction: {result["predicted_class"].upper()}\n'
                  f'Confidence: {result["confidence"]:.2%}', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def demo_batch_predictions(image_paths, classifier, save_path=None):
    """
    Visualize predictions for multiple images in a grid
    
    Args:
        image_paths: List of image paths
        classifier: WasteClassifier instance
        save_path: Optional path to save the figure
    """
    n_images = len(image_paths)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, img_path in enumerate(image_paths):
        try:
            # Make prediction
            result = classifier.predict(img_path, return_probabilities=True)
            
            # Load and display image
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].axis('off')
            
            # Add title with prediction
            title = f'{os.path.basename(img_path)}\n'
            title += f'Predicted: {result["predicted_class"].upper()}\n'
            title += f'Confidence: {result["confidence"]:.2%}'
            axes[idx].set_title(title, fontsize=10)
            
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error: {str(e)}', 
                          ha='center', va='center')
            axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def demo_confusion_examples(classifier, test_dir=None):
    """
    Show examples of correct and incorrect predictions
    
    Args:
        classifier: WasteClassifier instance
        test_dir: Directory with test images
    """
    if test_dir is None:
        test_dir = config.TEST_DIR
    
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return
    
    from data_preprocessing import load_dataset_from_directory
    
    # Load test images
    test_images, test_labels = load_dataset_from_directory(test_dir)
    
    if len(test_images) == 0:
        print("No test images found!")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Make predictions
    correct = []
    incorrect = []
    
    for img_path, true_label in zip(test_images, test_labels):
        result = classifier.predict(img_path)
        if result['predicted_class'] == true_label:
            correct.append((img_path, true_label, result['predicted_class']))
        else:
            incorrect.append((img_path, true_label, result['predicted_class']))
    
    print(f"Correct predictions: {len(correct)}/{len(test_images)}")
    print(f"Accuracy: {len(correct)/len(test_images):.2%}")
    
    # Visualize examples
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Show correct predictions
    for i in range(min(4, len(correct))):
        img_path, true_label, pred_label = correct[i]
        img = Image.open(img_path)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'✓ Correct\nTrue: {true_label}\nPred: {pred_label}',
                            color='green', fontsize=10)
    
    # Show incorrect predictions
    for i in range(min(4, len(incorrect))):
        img_path, true_label, pred_label = incorrect[i]
        img = Image.open(img_path)
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'✗ Incorrect\nTrue: {true_label}\nPred: {pred_label}',
                            color='red', fontsize=10)
    
    # Hide empty subplots
    for i in range(len(correct), 4):
        axes[0, i].axis('off')
    for i in range(len(incorrect), 4):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(config.RESULTS_DIR, 'prediction_examples.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Examples saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("WASTE CLASSIFICATION - DEMO VISUALIZATION")
    print("=" * 60)
    
    # Initialize classifier
    print("\nLoading classifier...")
    try:
        classifier = WasteClassifier()
    except Exception as e:
        print(f"Error loading classifier: {e}")
        print("\nPlease train the model first:")
        print("  python train_model.py")
        exit(1)
    
    # Option 1: Predict single image
    print("\n" + "-" * 60)
    print("Option 1: Visualize single prediction")
    print("-" * 60)
    
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            fig = visualize_prediction(image_path, classifier)
            plt.show()
        else:
            print(f"Image not found: {image_path}")
    else:
        print("Usage: python demo_visualization.py <image_path>")
        print("\nOr run: python demo_visualization.py")
        print("  to see confusion examples from test set")
    
    # Option 2: Show confusion examples
    print("\n" + "-" * 60)
    print("Option 2: Analyzing test set predictions")
    print("-" * 60)
    
    if os.path.exists(config.TEST_DIR):
        demo_confusion_examples(classifier)
        plt.show()
    else:
        print(f"Test directory not found: {config.TEST_DIR}")
        print("Add test images to see confusion examples")
