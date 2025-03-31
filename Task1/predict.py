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
                image = dataset[idx]['images']
                image = image.unsqueeze(0).to(config['DEVICE'], dtype=torch.float32)
                logits_pred = model(image)
                if args.task == 'general':
                    # logits_pred: [artist_logits, genre_logits, style_logits]
                    print(logits_pred[0].shape)
                    probs_pred = []
                    probs_pred.append(torch.softmax(logits_pred[0], dim=1)) 
                    probs_pred.append(torch.softmax(logits_pred[1], dim=1))
                    probs_pred.append(torch.softmax(logits_pred[2], dim=1))
                    pred_artist = torch.argmax(probs_pred[0], dim=1).item()
                    pred_genre = torch.argmax(probs_pred[1], dim=1).item()
                    pred_style = torch.argmax(probs_pred[2], dim=1).item()
                    pred = [pred_artist, pred_genre, pred_style]
                else:
                    probs_pred = torch.softmax(logits_pred, dim=1)
                    pred = torch.argmax(probs_pred, dim=1).item()

                raw_image, raw_label = dataset.get_raw(idx)['images'], dataset.get_raw(idx)['labels']
                all_images.append(raw_image)
                y_true.append(raw_label)
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
        with tqdm(total=len(dataset), desc=f'Finding outliers', unit='img') as pbar:
            for i in range(len(dataset)):
                idx = np.random.randint(0, len(dataset))
                image, label = dataset[idx]['images'], dataset[idx]['labels']
                image_tensor = image.unsqueeze(0).to(config['DEVICE'], dtype=torch.float32)
                logits_pred = model(image_tensor)
                
                if args.task == 'general':
                    probs_pred = []
                    probs_pred.append(torch.softmax(logits_pred[0], dim=1)) 
                    probs_pred.append(torch.softmax(logits_pred[1], dim=1))
                    probs_pred.append(torch.softmax(logits_pred[2], dim=1))
                    pred_artist = torch.argmax(probs_pred[0], dim=1).item()
                    pred_genre = torch.argmax(probs_pred[1], dim=1).item()
                    pred_style = torch.argmax(probs_pred[2], dim=1).item()
                    pred = [pred_artist, pred_genre, pred_style]
                    
                    # Get confidence scores for the predicted classes
                    max_prob_artist = probs_pred[0][0][pred_artist].item()
                    max_prob_genre = probs_pred[1][0][pred_genre].item()
                    max_prob_style = probs_pred[2][0][pred_style].item()
                    
                    # Check if the correct predictions has low confidence
                    if pred_artist == label[0] and pred_genre == label[1] and pred_style == label[2]:
                        if max_prob_artist < args.threshold or max_prob_genre < args.threshold or max_prob_style < args.threshold:
                            # Get raw image for display
                            raw_image = dataset.get_raw(i)['images']
                            outlier_images.append(raw_image)
                            outlier_labels.append(label)
                            outlier_preds.append(pred)
                            outlier_probs.append([max_prob_artist, max_prob_genre, max_prob_style])
                else:
                    probs_pred = torch.softmax(logits_pred, dim=1)
                    pred = torch.argmax(probs_pred, dim=1).item()
                    max_prob = probs_pred[0][pred].item()

                    # Check if the correct predictions has low confidence
                    if pred == label:
                        if max_prob < args.threshold:
                            # Get raw image for display
                            raw_image = dataset.get_raw(i)['images']
                            outlier_images.append(raw_image)
                            outlier_labels.append(label)
                            outlier_preds.append(pred)
                            outlier_probs.append(max_prob)
                
                pbar.update(1)

    return outlier_images, outlier_labels, outlier_preds, outlier_probs
