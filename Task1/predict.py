import torch
import numpy as np
from tqdm import tqdm

def get_prediction(dataset, model, args, config):
    '''
    Get predictions for a random sample of images from the validation dataset.

    Args:
        dataset: The validation dataset.
        model: The trained model.
        args: Command line arguments.
        config: User input arguments.
    Returns:
        all_images: List of images.
        y_true: List of true labels for the images.
        y_pred: List of predicted labels for the images.
    '''
    all_images, y_true, y_pred = [], [], []
    with torch.no_grad():
        with tqdm(total=args.size, desc=f'Predicting', unit='img') as pbar:
            for i in range(args.size):
                idx = np.random.randint(0, len(dataset))
                image, label = dataset[idx]['images'], dataset[idx]['labels']
                image = image.unsqueeze(0).to(config['DEVICE'], dtype=torch.float32)
                logits_pred = model(image)
                if args.task == 'general':
                    probs_pred = torch.softmax(logits_pred, dim=1)
                    pred_artist = torch.argmax(probs_pred[0], dim=1).item()
                    pred_genre = torch.argmax(probs_pred[1], dim=1).item()
                    pred_style = torch.argmax(probs_pred[2], dim=1).item()
                    pred = [pred_artist, pred_genre, pred_style]
                else:
                    probs_pred = torch.softmax(logits_pred, dim=1)
                    pred = torch.argmax(probs_pred, dim=1).item()

                raw_image = dataset.get_raw(idx)['images']
                all_images.append(raw_image)
                y_true.append(label)
                y_pred.append(pred)
                pbar.update(1)
    
    return all_images, y_true, y_pred

def get_outliers(dataset, model, args, config):
    '''
    Get outliers from the validation dataset based on model predictions.
    
    Args:
        dataset: The validation dataset.
        model: The trained model.
        args: Command line arguments.
        config: User input arguments.
    Returns:
        outlier_images: List of outlier images.
        outlier_labels: List of true labels for the outlier images.
        outlier_preds: List of predicted labels for the outlier images.
        outlier_probs: List of prediction probabilities for the outlier images.
    '''
    outlier_images = []
    outlier_labels = []
    outlier_preds = []
    outlier_probs = []

    with torch.no_grad():
        for i in tqdm(total=range(len(dataset)), desc=f'Finding outliers', unit='img'):
            idx = np.random.randint(0, len(dataset))
            image, label = dataset[idx]['images'], dataset[i]['labels']
            image_tensor = image.unsqueeze(0).to(config['DEVICE'], dtype=torch.float32)
            logits_pred = model(image_tensor)
            
            if args.task == 'general':
                probs_pred = torch.softmax(logits_pred, dim=1)
                pred_artist = torch.argmax(probs_pred[0], dim=1).item()
                pred_genre = torch.argmax(probs_pred[1], dim=1).item()
                pred_style = torch.argmax(probs_pred[2], dim=1).item()
                pred = [pred_artist, pred_genre, pred_style]
                
                # Check if this is an outlier (any of the three predictions has low confidence)
                max_prob_artist = probs_pred[0][0][pred_artist].item()
                max_prob_genre = probs_pred[1][0][pred_genre].item()
                max_prob_style = probs_pred[2][0][pred_style].item()
                
                if max_prob_artist < config['OUTLIER_THRESHOLD'] or max_prob_genre < config['OUTLIER_THRESHOLD'] or max_prob_style < config['OUTLIER_THRESHOLD']:
                    raw_image = dataset.get_raw(i)['images']
                    outlier_images.append(raw_image)
                    outlier_labels.append(label)
                    outlier_preds.append(pred)
                    outlier_probs.append([max_prob_artist, max_prob_genre, max_prob_style])
            else:
                probs_pred = torch.softmax(logits_pred, dim=1)
                pred = torch.argmax(probs_pred, dim=1).item()
                max_prob = probs_pred[0][pred].item()
                
                if max_prob < config['OUTLIER_THRESHOLD']:
                    raw_image = dataset.get_raw(i)['images']
                    outlier_images.append(raw_image)
                    outlier_labels.append(label)
                    outlier_preds.append(pred)
                    outlier_probs.append(max_prob)

    return outlier_images, outlier_labels, outlier_preds, outlier_probs
