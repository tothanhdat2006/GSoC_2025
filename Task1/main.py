import argparse
import numpy as np

import torch

from configs.config import get_config
from utils.dataset import create_dataset, create_dataloader
from models.model import load_model
from evaluate import evaluate, show_general_result, show_specific_result
from predict import get_prediction, get_outliers
from utils.visualization import plot_general, plot_specific, plot_outlier

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data', help='Directory of the downloaded data')
    parser.add_argument('--model_weights', type=str, default='./weights', help='Directory of the pretrained model weights')
    parser.add_argument('--model_name', type=str, default='resnet50', help='Name of the model to use (Resnet, ViT)')
    parser.add_argument('--task', type=str, default='artist', help='Task to perform (artist, style, genre, or general)')
    parser.add_argument('--evaluate', type=int, default=0, help='Turn on evaluation (if needed)')
    parser.add_argument('--size', type=int, default=5, help='Number of images to classify and display')
    parser.add_argument('--outliers', type=int, default=5, help='Number of outliers to display')
    parser.add_argument('--threshold', type=float, default=0.25, help='Threshold for outlier detection')
    parser.add_argument('--seed', type=int, default=86, help='Random seed for reproducibility')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    config = get_config()
    config['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Load model weights (Optional)
    model = load_model(args, config)
    model.to(config['DEVICE'])

    # 2. Classify images
    # 2.1. Load images
    val_dataset = create_dataset(args.data_path, config['MODE'], args.task)
    val_dataloader = create_dataloader(val_dataset, config['MODE'], args.task, config['BATCH_SIZE'], config['NUM_WORKERS'])

    # 2.3. Evaluate model performance
    if args.evaluate:
        metrics = evaluate(args.task, model, val_dataset, val_dataloader, config['DEVICE'])
        if args.task == 'general':
            show_general_result(metrics, val_dataset)
        else:
            show_specific_result(metrics, val_dataset)


    # 2.4. Predict random images and display results
    np.random.seed(config['SEED'])
    all_images, y_true, y_pred = get_prediction(val_dataset, model, args, config)
    if args.task == 'general':
        plot_general(val_dataset, all_images, y_true, y_pred, args.size, args)
    else:
        plot_specific(val_dataset, all_images, y_true, y_pred, args.size, args)


    # 3. Detect and display outliers
    outlier_images, outlier_labels, outlier_preds, outlier_probs = get_outliers(val_dataset, model, args, config)
    plot_outlier(val_dataset, outlier_images, outlier_labels, outlier_preds, outlier_probs, args)
