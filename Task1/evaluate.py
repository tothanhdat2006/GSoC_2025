import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import torch
import torch.nn as nn

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score

def calculate_metrics(dataset, all_labels, all_pred_probs):
    '''
    Args:
        dataset: The dataset used for evaluation.
        all_labels: true labels (len(dataset), n_labels)
        all_pred_probs: probability of each class for each labels Tuple<(len(dataset), n_classes), 
                                                                        (len(dataset), n_classes) 
                                                                        (len(dataset), n_classes)>

    Returns:
        metrics: Dictionary containing the evaluation metrics (Per class and Overall precision, recall, f1 score).
    '''
    all_labels = all_labels.cpu().numpy()
    all_pred_labels = [all_pred_probs[i].argmax(axis=1) for i in range(3)]
    
    # TP[i][j] = true positive of class j in category (0: artist, 1: genre, 2: style)
    TP = [[0] * 28 for _ in range(3)]
    FP = [[0] * 28 for _ in range(3)]
    TN = [[0] * 28 for _ in range(3)]
    FN = [[0] * 28 for _ in range(3)]
    for i in range(3): # artist, genre, style
        for cls in range(dataset.get_num_classes()[i]): # class [0..n_classes]
            for idx in range(len(all_labels)):
                if all_labels[idx][i] == cls: 
                    if all_pred_labels[i][idx] == cls:
                        TP[i][cls] += 1
                    else:
                        FN[i][cls] += 1 
                else: 
                    if all_pred_labels[i][idx] == cls:
                        FP[i][cls] += 1
                    else:
                        TN[i][cls] += 1 
        
    # P: Per class, O: Overall 
    metrics = {}
    tp = 0.0
    tp_fn = 0.0
    tp_fp = 0.0
    # Artist
    num_artist_classes, num_genre_classes, num_style_classes = dataset.get_num_classes()
    for cls in range(num_artist_classes):
        cls_name = dataset.get_label_artist(cls)
        metrics[f'PP_{cls_name}'] = TP[0][cls] / max(1, TP[0][cls] + FN[0][cls])
        metrics[f'PR_{cls_name}'] = TP[0][cls] / max(1, TP[0][cls] + FP[0][cls])
        metrics[f'PF1_{cls_name}'] = 2 * (metrics[f'PP_{cls_name}'] * metrics[f'PR_{cls_name}']) / max(1, (metrics[f'PP_{cls_name}'] + metrics[f'PR_{cls_name}']))
        tp = tp + TP[0][cls]
        tp_fn = tp_fn + max(1e-5, TP[0][cls] + FN[0][cls])
        tp_fp = tp_fp + max(1e-5, TP[0][cls] + FP[0][cls])
    
    # Genre
    for cls in range(num_genre_classes):
        cls_name = dataset.get_label_genre(cls)
        metrics[f'PP_{cls_name}'] = TP[1][cls] / max(1, TP[1][cls] + FN[1][cls])
        metrics[f'PR_{cls_name}'] = TP[1][cls] / max(1, TP[1][cls] + FP[1][cls])
        metrics[f'PF1_{cls_name}'] = 2 * (metrics[f'PP_{cls_name}'] * metrics[f'PR_{cls_name}']) / max(1, (metrics[f'PP_{cls_name}'] + metrics[f'PR_{cls_name}']))
        tp = tp + TP[1][cls]
        tp_fn = tp_fn + max(1e-5, TP[1][cls] + FN[1][cls])
        tp_fp = tp_fp + max(1e-5, TP[1][cls] + FP[1][cls])
    
    # Style
    for cls in range(num_style_classes):
        cls_name = dataset.get_label_style(cls)
        metrics[f'PP_{cls_name}'] = TP[2][cls] / max(1, TP[2][cls] + FN[2][cls])
        metrics[f'PR_{cls_name}'] = TP[2][cls] / max(1, TP[2][cls] + FP[2][cls])
        metrics[f'PF1_{cls_name}'] = 2 * (metrics[f'PP_{cls_name}'] * metrics[f'PR_{cls_name}']) / max(1, (metrics[f'PP_{cls_name}'] + metrics[f'PR_{cls_name}']))
        tp = tp + TP[2][cls]
        tp_fn = tp_fn + max(1e-5, TP[2][cls] + FN[2][cls])
        tp_fp = tp_fp + max(1e-5, TP[2][cls] + FP[2][cls])
    
    metrics[f'OP'] = tp / tp_fn
    metrics[f'OR'] = tp / tp_fp
    metrics[f'OF1'] = 2 * (metrics[f'OP'] * metrics[f'OR']) / (metrics[f'OP'] + metrics[f'OR'])
    return metrics

def evaluate_specific(model, dataset, dataloader, device):
    '''
    Evaluate the model on a specific task (artist, genre, or style).

    Args:
        model: The trained model.
        dataset: The dataset used for evaluation.
        dataloader: DataLoader for the dataset.
        device: The device to use (CPU or GPU).
    Returns:
        all_label: List of true labels.
        all_pred: List of predicted labels.
    '''
    model.eval() # Set the model to evaluation mode
    all_label, all_pred = [], []
    with torch.no_grad():
        with tqdm(total=len(dataset), desc=f'Testing', unit='img') as pbar:
            for batch in dataloader:
                images, labels = batch['images'], batch['labels']
                images = images.to(device, dtype=torch.float32, memory_format=torch.channels_last)
                labels = labels.to(device, dtype=torch.long)
                
                labels_pred = model(images)
                labels_pred = nn.Softmax(dim=1)(labels_pred)

                predicted = torch.argmax(labels_pred, dim=1) 
                all_label.append(labels.cpu().numpy())
                all_pred.append(predicted.cpu().numpy()) 
                pbar.update(images.shape[0])

    all_label = np.concatenate(all_label)
    all_pred = np.concatenate(all_pred)
    return all_label, all_pred

def evaluate_general(model, dataset, dataloader, device):
    '''
    Evaluate the model on the general task (artist, genre, and style).

    Args:
        model: The trained model.
        dataset: The dataset used for evaluation.
        dataloader: DataLoader for the dataset.
        device: The device to use (CPU or GPU).
    Returns:
        metrics: Dictionary containing the evaluation metrics (Per class and Overall precision, recall, f1 score).
    '''
    model.eval()
    all_labels = []
    all_pred_probs_artist, all_pred_probs_genre, all_pred_probs_style = [], [], []
    with torch.no_grad():
        with tqdm(total=len(dataset), desc=f'Validating', unit='img') as pbar:
            for batch in dataloader:
                images, labels = batch['images'], batch['labels']
                images = images.to(device, dtype=torch.float32, memory_format=torch.channels_last)
                labels = labels.to(device, dtype=torch.long) # (batch_size, 3)
                
                logits_pred = model(images, labels) # (batch_size, 3)
                
                all_pred_probs_artist.append(torch.softmax(logits_pred[0], dim=1))
                all_pred_probs_genre.append(torch.softmax(logits_pred[1], dim=1))
                all_pred_probs_style.append(torch.softmax(logits_pred[2], dim=1))
                all_labels.append(labels)
                pbar.update(images.shape[0])

    model.train()
    all_pred_probs_artist = torch.cat(all_pred_probs_artist)
    all_pred_probs_genre = torch.cat(all_pred_probs_genre)
    all_pred_probs_style = torch.cat(all_pred_probs_style)
    all_labels = torch.cat(all_labels)
    metrics = {}
    metrics = calculate_metrics(dataset, all_labels, (all_pred_probs_artist, all_pred_probs_genre, all_pred_probs_style))
    return metrics

def show_specific_result(metrics, dataset):
    '''
    Show the evaluation results for the specific task.

    Args:
        metrics: Tuple containing the evaluation metrics (from evaluate_specific function).
        dataset: The dataset used for evaluation.
    Returns:
        None
    '''
    all_label, all_pred = metrics
    # Create confusion matrix
    cm = confusion_matrix(all_label, all_pred, labels=np.arange(0, len(dataset.classes_csv)))
    cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(0, len(dataset.classes_csv)))
    fig, ax = plt.subplots(figsize=(15,15))
    cmDisplay.plot(ax=ax)
    plt.title('Confusion Matrix')
    plt.show()
    # Calculate metrics
    print(f"Accuracy: {100 * accuracy_score(all_label, all_pred, normalize=True)}")
    print(f"Micro F1: {100 * f1_score(all_label, all_pred, average='micro')}")
    print(f"Weighted F1: {100 * f1_score(all_label, all_pred, average='weighted')}")

    print(classification_report(all_label, all_pred, target_names=dataset.classes_csv['name'].tolist(), digits=4))

def show_general_result(metrics, dataset):
    '''
    Show the evaluation results for the general task.

    Args:
        metrics: Dictionary containing the evaluation metrics (from evaluate_general function).
        dataset: The dataset used for evaluation.
    Returns:   
        None
    '''
    num_artist_classes, num_genre_classes, num_style_classes = dataset.get_num_classes()

    # Artist
    print(" \n================================ Artist ================================")
    for cls in range(num_artist_classes):
        cls_name = dataset.get_label_artist(cls)
        print(f"{cls_name} (Precision, Recall, F1 score): ({ metrics[f'PP_{cls_name}'] }, { metrics[f'PR_{cls_name}'] }, { metrics[f'PF1_{cls_name}'] })")
    
    # Genre
    print(" \n================================ Genre ================================")
    for cls in range(num_genre_classes):
        cls_name = dataset.get_label_genre(cls)
        print(f"{cls_name} (Precision, Recall, F1 score): ({ metrics[f'PP_{cls_name}'] }, { metrics[f'PR_{cls_name}'] }, { metrics[f'PF1_{cls_name}'] })")
    
    # Style
    print(" \n================================ Style ================================")
    for cls in range(num_style_classes):
        cls_name = dataset.get_label_style(cls)
        print(f"{cls_name} (Precision, Recall, F1 score): ({ metrics[f'PP_{cls_name}'] }, { metrics[f'PR_{cls_name}'] }, { metrics[f'PF1_{cls_name}'] })")
    
    print(" \n================================ Overall ================================")
    print(f"Overall Precision: {metrics['OP']}")
    print(f"Overall Recall: {metrics['OR']}")
    print(f"Overall F1: {metrics['OF1']}")


def evaluate(task, model, dataset, dataloader, device):
    if task == 'artist' or task == 'style' or task == 'genre':
        return evaluate_specific(model, dataset, dataloader, device)
    else:
        return evaluate_general(model, dataset, dataloader, device)
