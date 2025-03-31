import matplotlib.pyplot as plt
from PIL import Image

def plot_specific(dataset, images, y_true, y_pred, number_of_images, args):
    '''
    Plot images in general task with their true labels and predicted labels.

    Args:
        dataset: Dataset object containing the images and labels. (Used to get labels' name)
        images: List of image file names.
        y_true: List of true labels for the images.
        y_pred: List of predicted labels for the images.
        number_of_images: Number of images to display.
        args: User input arguments.
    Returns:
        None
    '''
    plt.figure(figsize=(18, 8))
    for i in range(number_of_images):
        image = Image.open(args.data_path + '/' + images[i]).convert("RGB")
        plt.subplot(1, number_of_images, i + 1)
        true_label = dataset.get_label(y_true[i])
        pred_label = dataset.get_label(y_pred[i])
        plt.title(f"True label: {true_label}\nPrediction label: {pred_label}", fontsize=8)
        plt.axis('off')
        plt.imshow(image)
    
    plt.tight_layout()
    plt.show()
        
def plot_general(dataset, images, y_true, y_pred, number_of_images, args):
    '''
    Plot images in general task with their true labels and predicted labels.

    Args:
        dataset: Dataset object containing the images and labels. (Used to get labels' name)
        images: List of image file names.
        y_true: List of true labels for the images.
        y_pred: List of predicted labels for the images.
        number_of_images: Number of images to display.
        args: User input arguments.
    Returns:
        None
    '''
    plt.figure(figsize=(18, 8))
    for i in range(number_of_images):
        image = Image.open(args.data_path + '/' + images[i]).convert("RGB")
        plt.subplot(1, number_of_images, i + 1)
        true_label_artist = dataset.get_label_artist(y_true[i][0])
        true_label_genre = dataset.get_label_genre(y_true[i][1])
        true_label_style = dataset.get_label_style(y_true[i][2])
        pred_label_artist = dataset.get_label_artist(y_pred[i][0])
        pred_label_genre = dataset.get_label_genre(y_pred[i][1])
        pred_label_style = dataset.get_label_style(y_pred[i][2])
        title += f"\nTrue artist: {true_label_artist}\nPrediction artist: {pred_label_artist}"
        title += f"\nTrue genre: {true_label_genre}\nPrediction genre: {pred_label_genre}"
        title += f"\nTrue style: {true_label_style}\nPrediction style: {pred_label_style}"
        plt.title(title, fontsize=8)
        plt.axis('off')
        plt.imshow(image)
    
    plt.tight_layout()
    plt.show()

def plot_outlier(dataset, outlier_images, outlier_labels, outlier_preds, outlier_probs, args):
    '''
    Plot outlier images with their true labels, predicted labels, and probabilities.

    Args:
        dataset: Dataset object containing the images and labels. (Used to get labels' name)
        outlier_images: List of outlier images.
        outlier_labels: List of true labels for the outlier images.
        outlier_preds: List of predicted labels for the outlier images.
        outlier_probs: List of prediction probabilities for the outlier images.
        args: User input arguments.
    '''
    if len(outlier_images) > 0:
        num_outliers_to_display = min(args.outliers, len(outlier_images))
        print(f"Found {len(outlier_images)} outliers. Displaying {num_outliers_to_display} outliers...")
        plt.figure(figsize=(18, 6))
        
        # Display each outlier
        for i in range(num_outliers_to_display):
            image = Image.open(args.data_path + '/' + outlier_images[i]).convert("RGB")
            plt.subplot(1, num_outliers_to_display, i+1)
            plt.imshow(image)
            plt.axis('off')
            
            if args.task == 'general':
                title = f"True/pred: {outlier_labels[i][0]}/{outlier_preds[i][0]} prob: {outlier_probs[i][0]:.2f}\n"
                title += f"True/pred: {outlier_labels[i][1]}/{outlier_preds[i][1]} prob: {outlier_probs[i][1]:.2f}\n"
                title += f"True/pred: {outlier_labels[i][2]}/{outlier_preds[i][2]} prob: {outlier_probs[i][2]:.2f}"
            else:
                title = f"True: {outlier_labels[i]}\nPred: {outlier_preds[i]}\nProb: {outlier_probs[i]:.2f}"
            
            plt.title(title, fontsize=10)
        
        plt.tight_layout()
        
        # Save the figure
        task_name = args.task
        plt.savefig(f"experiment/images/{task_name}_outliers.png")
        plt.show()
    else:
        print("No outliers found with probability < 0.3")