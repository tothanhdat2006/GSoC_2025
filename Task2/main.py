import argparse
import numpy as np

from tqdm import tqdm
import torch

from configs.config import get_config
from utils.download_data import download_data
from utils.dataset import NGADataset
from models.model import Resnet50_extractor
from utils.visualize import CosineSimilarityVisualizer, CosineDistanceVisualizer, SSIMVisualizer, RMSEVisualizer

def get_image_features(dataset, model):
    model.eval()
    all_features = []
    with tqdm(total=len(dataset), desc='Extracting features from images', unit='img') as pbar:
        for i in range(len(dataset)):
            images = dataset.get_item_val(i)['images']
            images = images.unsqueeze(0) # Add batch dimension
            images = images.to(model.device, dtype=torch.float32, memory_format=torch.channels_last)
            feature = model(images)

            all_features.extend(feature.detach().cpu().numpy())
            pbar.update(images.shape[0])

    model.train()
    return all_features

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data', help='Directory of the NGA')
    parser.add_argument('--output_path', type=str, default='./output', help='Directory of the downloaded data')
    parser.add_argument('--size', type=int, default=-1, help='Size of the dataset to download (-1 means all data)')
    parser.add_argument('--metric', type=str, default="cosine", help='Metric to use for visualization (cosine_similarity, cosine_distance, ssim or rmse)')
    parser.add_argument('--save_figure', type=bool, default=True, help='Save the visualization figure')
    parser.add_argument('--figure_name', type=str, default='./experiment/images/query_results', help='Name for the saved figure')
    parser.add_argument('--seed', type=int, default=86, help='Seed for random query image')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # 0. Get config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    config = get_config()
    config['SEED'] = args.seed
    if args.save_figure:
        config['SAVE_FIGURE'] = args.save_figure
        config['FIGURE_NAME'] = args.figure_name + f"_{args.metric}_seed{args.seed}.png" # Complete the name of the figure
    
    # 1. Download dataset
    download_data(args.data_path, args.output_path, args.size)

    # 2. Create dataset
    dataset = NGADataset(data_path=args.output_path)

    # 3. Extract features
    model = Resnet50_extractor(device)
    model = model.to(memory_format = torch.channels_last)
    features = get_image_features(dataset, model)

    # 4. Query image similarty and visualize
    np.random.seed(config['SEED'])
    query_image_idx = np.random.randint(0, len(dataset))
    if args.metric == 'cosine_similarity':
        CosineSimilarityVisualizer(dataset, features, config).visualize_results(query_image_idx, 5)
    elif args.metric == 'cosine_distance':
        CosineDistanceVisualizer(dataset, features, config).visualize_results(query_image_idx, 5)
    elif args.metric == 'ssim':
        SSIMVisualizer(dataset, features, config).visualize_results(query_image_idx, 5)
    elif args.metric == 'rmse':
        RMSEVisualizer(dataset, features, config).visualize_results(query_image_idx, 5)
    else:
        raise ValueError("Invalid metric. Choose from 'cosine_similarity', 'cosine_distance', 'ssim' or 'rmse'.")

