import numpy as np
import matplotlib.pyplot as plt

from utils.metrics import cosine_similarity, cosine_distance, ssim, rmse

'''
Summary:
- KNNCosineDistanceVisualizer: Visualizes the results of using KNN with cosine distance to query similar images.
- CosineDistanceVisualizer: Visualizes the results of using only cosine distance to query similar images.
- CosineSimilarityVisualizer: Visualizes the results of using cosine similarity to query similar images.
- SSIMVisualizer: Visualizes the results of using SSIM to query similar images.
- RMSEVisualizer: Visualizes the results of using RMSE to query similar images.
'''

class CosineDistanceVisualizer():
    '''
    A class to visualize the results of using RMSE with input metric to query most similar images.
    It uses the NearestNeighbors class from sklearn to find the most similar images based on cosine distance.
    '''
    def __init__(self, dataset, features, config):
        '''
        Initialize with a dataset and features of the dataset.

        Args:
            dataset: CustomDataset containing the images.
            features: List of features for all images in the dataset.
            config: Configuration dictionary containing visualization settings.
        '''
        self.dataset = dataset
        self.features = features
        self.config = config

    def query_similar_images(self, query_image_idx, number_of_images):
        '''
        Using KNN to find the most similar images to the query image.

        Args:
            query_image_idx: Index of the query image in the dataset.
            number_of_images: Number of similar images to retrieve.

        Returns:
            similar_images_sorted: List contains top 5 similar images to the query image.
        '''
        features_copy = self.features.copy()
        query_image_feature = features_copy.pop(query_image_idx)

        # Calculate cosine distance between the EMBEDDING of query image and the EMBEDDING of all other images
        similar_images = []
        query_image_feature = self.features[query_image_idx]
        for idx, feature in enumerate(self.features):
            if idx == query_image_idx:
                continue
            current_cd = cosine_distance(query_image_feature, feature)
            similar_images.append((self.dataset.get_filename(idx), current_cd))

        # Sort the images based on cosine distance and return the top N similar images
        similar_images_sorted = sorted(similar_images, key=lambda x: x[1], reverse=False)[:number_of_images]
        return similar_images_sorted
    
    def visualize_results(self, query_image_idx, number_of_images=5):
        '''
        Visualize the query image and its top 5 similar images (using KNN to determined).

        Args:
            query_image_idx: Index of the query image in the dataset.
            number_of_images: Number of similar images to retrieve.

        Returns:
            None
        '''
        plt.figure(figsize=(20, 10))
        plt.subplot(1, number_of_images+1, 1)
        query_image_filename = self.dataset.get_filename(query_image_idx)
        query_image = self.dataset.get_image(query_image_filename)
        plt.imshow(query_image.convert('RGB'))
        plt.title('Query image\nQuery using Cosine Distance')
        plt.axis('off')

        similar_images_sorted = self.query_similar_images(query_image_idx, number_of_images)

        for idx, (filename, cos_dist) in enumerate(similar_images_sorted):
            target_image = self.dataset.get_image(filename)
            plt.subplot(1, number_of_images+1, idx + 2)
            plt.imshow(target_image.convert('RGB'))
            plt.title(f'Top {idx+1}\nCosine Distance: {cos_dist:.4f}')
            plt.axis('off')

        plt.tight_layout()
        if self.config["SAVE_FIGURE"]:
            plt.savefig(self.config["FIGURE_NAME"])
        plt.show()


class CosineSimilarityVisualizer():
    '''
    A class to visualize the results of using RMSE with input metric to query most similar images.
    It uses the NearestNeighbors class from sklearn to find the most similar images based on cosine distance.
    '''
    def __init__(self, dataset, features, config):
        '''
        Initialize with a dataset and features of the dataset.

        Args:
            dataset: CustomDataset containing the images.
            features: List of features for all images in the dataset.
            config: Configuration dictionary containing visualization settings.
        '''
        self.dataset = dataset
        self.features = features
        self.config = config

    def query_similar_images(self, query_image_idx, number_of_images):
        '''
        Using KNN to find the most similar images to the query image.

        Args:
            query_image_idx: Index of the query image in the dataset.
            number_of_images: Number of similar images to retrieve.

        Returns:
            similar_images_sorted: List contains top 5 similar images to the query image.
        '''
        # Calculate cosine similarity between the EMBEDDING of query image and the EMBEDDING of all other images
        similar_images = []
        query_image_feature = self.features[query_image_idx]
        for idx, feature in enumerate(self.features):
            if idx == query_image_idx:
                continue
            current_cs = cosine_similarity(query_image_feature, feature)
            similar_images.append((self.dataset.get_filename(idx), current_cs))

        # Sort the images based on cosine similarity and return the top N similar images
        similar_images_sorted = sorted(similar_images, key=lambda x: x[1], reverse=True)[:number_of_images]
        return similar_images_sorted
    
    def visualize_results(self, query_image_idx, number_of_images=5):
        '''
        Visualize the query image and its top 5 similar images (using KNN to determined).

        Args:
            query_image_idx: Index of the query image in the dataset.
            number_of_images: Number of similar images to retrieve.

        Returns:
            None
        '''
        plt.figure(figsize=(20, 10))
        plt.subplot(1, number_of_images+1, 1)
        query_image_filename = self.dataset.get_filename(query_image_idx)
        query_image = self.dataset.get_image(query_image_filename)
        plt.imshow(query_image.convert('RGB'))
        plt.title('Query image\nQuery using Cosine Similarity')
        plt.axis('off')

        similar_images_sorted = self.query_similar_images(query_image_idx, number_of_images)

        for idx, (filename, cos_sim) in enumerate(similar_images_sorted):
            target_image = self.dataset.get_image(filename)
            plt.subplot(1, number_of_images+1, idx + 2)
            plt.imshow(target_image.convert('RGB'))
            plt.title(f'Top {idx+1}\nCosine Similarity: {cos_sim:.4f}')
            plt.axis('off')

        plt.tight_layout()
        if self.config["SAVE_FIGURE"]:
            plt.savefig(self.config["FIGURE_NAME"])
        plt.show()

class SSIMVisualizer():
    '''
    A class to visualize the results of using SSIM with input metric to query most similar images.
    It uses the NearestNeighbors class from sklearn to find the most similar images based on cosine distance.
    '''
    def __init__(self, dataset, features, config):
        '''
        Initialize with a dataset and features of the dataset.

        Args:
            dataset: CustomDataset containing the images.
            features: List of features for all images in the dataset.
            config: Configuration dictionary containing visualization settings.
        '''
        self.dataset = dataset
        self.features = features
        self.config = config

    def query_similar_images(self, query_image_idx, number_of_images):
        '''
        Using KNN to find the most similar images to the query image.

        Args:
            query_image_idx: Index of the query image in the dataset.
            number_of_images: Number of similar images to retrieve.

        Returns:
            similar_images_sorted: List contains top 5 similar images to the query image.
        '''
        query_image = self.dataset.get_image(self.dataset.get_filename(query_image_idx))
        np_query_image = np.array(query_image)

        # Calculate SSIM score between the query image and all other images
        similar_images = []
        for idx in range(len(self.dataset)):
            if idx == query_image_idx:
                continue
            target_image = self.dataset.get_image(self.dataset.get_filename(idx))
            current_ssim = ssim(np_query_image, np.array(target_image))
            similar_images.append((self.dataset.get_filename(idx), current_ssim))

        # Sort the images based on SSIM score and return the top N similar images
        similar_images_sorted = sorted(similar_images, key=lambda x: x[1], reverse=True)[:number_of_images]
        return similar_images_sorted
    
    def visualize_results(self, query_image_idx, number_of_images=5):
        '''
        Visualize the query image and its top 5 similar images (using KNN to determined).

        Args:
            query_image_idx: Index of the query image in the dataset.
            number_of_images: Number of similar images to retrieve.

        Returns:
            None
        '''
        plt.figure(figsize=(20, 10))
        plt.subplot(1, number_of_images+1, 1)
        query_image_filename = self.dataset.get_filename(query_image_idx)
        query_image = self.dataset.get_image(query_image_filename)
        plt.imshow(query_image.convert('RGB'))
        plt.title('Query image\nQuery using SSIM score')
        plt.axis('off')

        similar_images_sorted = self.query_similar_images(query_image_idx, number_of_images)

        for idx, (filename, ssim_score) in enumerate(similar_images_sorted):
            target_image = self.dataset.get_image(filename)
            plt.subplot(1, number_of_images+1, idx + 2)
            plt.imshow(target_image.convert('RGB'))
            plt.title(f'Top {idx+1}\nSSIM: {ssim_score:.4f}')
            plt.axis('off')

        plt.tight_layout()
        if self.config["SAVE_FIGURE"]:
            plt.savefig(self.config["FIGURE_NAME"])
        plt.show()
        

class RMSEVisualizer():
    '''
    A class to visualize the results of using RMSE with input metric to query most similar images.
    It uses the NearestNeighbors class from sklearn to find the most similar images based on cosine distance.
    '''
    def __init__(self, dataset, features, config):
        '''
        Initialize with a dataset and features of the dataset.

        Args:
            dataset: CustomDataset containing the images.
            features: List of features for all images in the dataset.
            config: Configuration dictionary containing visualization settings.
        '''
        self.dataset = dataset
        self.features = features
        self.config = config

    def query_similar_images(self, query_image_idx, number_of_images):
        '''
        Using KNN to find the most similar images to the query image.

        Args:
            query_image_idx: Index of the query image in the dataset.
            number_of_images: Number of similar images to retrieve.

        Returns:
            similar_images_sorted: List contains top 5 similar images to the query image.
        '''
        query_image = self.dataset.get_image(self.dataset.get_filename(query_image_idx))
        np_query_image = np.array(query_image)

        # Calculate RMSE between the query image and all other images
        similar_images = []
        for idx in range(len(self.dataset)):
            if idx == query_image_idx:
                continue
            target_image = self.dataset.get_image(self.dataset.get_filename(idx))
            current_rmse = rmse(np_query_image, np.array(target_image))
            similar_images.append((self.dataset.get_filename(idx), current_rmse))

        # Sort the images based on RMSE and return the top N similar images
        similar_images_sorted = sorted(similar_images, key=lambda x: x[1], reverse=False)[:number_of_images]
        return similar_images_sorted
    
    def visualize_results(self, query_image_idx, number_of_images=5):
        '''
        Visualize the query image and its top 5 similar images (using KNN to determined).

        Args:
            query_image_idx: Index of the query image in the dataset.
            number_of_images: Number of similar images to retrieve.

        Returns:
            None
        '''
        plt.figure(figsize=(20, 10))
        plt.subplot(1, number_of_images+1, 1)
        query_image_filename = self.dataset.get_filename(query_image_idx)
        query_image = self.dataset.get_image(query_image_filename)
        plt.imshow(query_image.convert('RGB'))
        plt.title('Query image\nQuery using RMSE')
        plt.axis('off')

        similar_images_sorted = self.query_similar_images(query_image_idx, number_of_images)

        for idx, (filename, rmse_score) in enumerate(similar_images_sorted):
            target_image = self.dataset.get_image(filename)

            plt.subplot(1, number_of_images+1, idx + 2)
            plt.imshow(target_image.convert('RGB'))
            plt.title(f'Top {idx+1}\nRMSE: {rmse_score:.4f}')
            plt.axis('off')

        plt.tight_layout()
        if self.config["SAVE_FIGURE"]:
            plt.savefig(self.config["FIGURE_NAME"])
        plt.show()