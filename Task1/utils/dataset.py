import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms

# Due to download issues, some images are corrupted and cannot be opened.
corrupted_images = [
    'Baroque/rembrandt_woman-standing-with-raised-hands.jpg',
    'Post_Impressionism/vincent-van-gogh_l-arlesienne-portrait-of-madame-ginoux-1890.jpg'
]

TASKS_LIST = ['artist', 'genre', 'style', 'general']

# -------------------------------------------- Specific Wikiart Dataset --------------------------------------------
class TRAIN_Specific_WikiartDataset(Dataset):
    def __init__(self, data_path, task='genre'):
        super().__init__()
        assert (task in TASKS_LIST), f'Task should be either {TASKS_LIST}\n'
        
        self.data_path = data_path
        self.task = task
        self.classes_csv = pd.read_csv(data_path / f'{task}_class.txt', sep=" ", names=['label', 'name'])

        # --------------------------- Cleaning data ---------------------------
        data_csv = pd.read_csv(data_path / f'{task}_train.csv', names=['filename', 'label'])

        # Remove corrupted images if any
        data_csv = data_csv.query("filename not in @corrupted_images")
        
        # Processing filenames to remove special characters and convert to ASCII
        import re
        import unicodedata
        def process_filename(f): # Remove non-ascii characters and '/' from filenames
            dirname, filename = f.split('/', 1)
            normalized = unicodedata.normalize('NFKD', filename)
            ascii_filename = ''
            for char in normalized:
                if ord(char) < 128 and ord(char) != 39:
                    ascii_filename += char
                else:
                    replacements = {
                        'ä': 'a', 'ö': 'o', 'ü': 'u', 'ß': 'ss',
                        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
                        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
                        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
                    }
                    ascii_filename += replacements.get(char, '_') # Replace with _ 
        
            ascii_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', ascii_filename)
            return dirname + "/" + ascii_filename
        
        data_csv.loc[:, "filename"] = data_csv["filename"].map(process_filename) # Pandas 3.0

        self.data_csv = data_csv
        self.imgs_path = data_csv["filename"].tolist()
        self.labels = data_csv["label"].tolist()

        # --------------------------- Custom transforms for train dataset ---------------------------
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop((224,224)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomVerticalFlip(0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.imgs_path)

    def get_label_name(self, label):
        return self.classes_csv['name'][label]

    def get_raw(self, idx):
        img = self.imgs_path[idx]
        label = self.labels[idx]

        return {
            'images': img,
            'labels': label
        }
    
    def transform(self, img):
        img = self.train_transforms(img)
        return img

    def __getitem__(self, idx):
        img = Image.open(self.data_path / self.imgs_path[idx])
        label = self.labels[idx]

        img = self.transform(img)
        return {
            'images': img,
            'labels': label
        }

class VAL_Specific_WikiartDataset(Dataset):
    def __init__(self, data_path, task='genre'):
        super().__init__()
        assert (task in TASKS_LIST), f'Task should be either {TASKS_LIST}\n'
        
        self.data_path = data_path
        self.task = task
        self.classes_csv = pd.read_csv(data_path / f'{task}_class.txt', sep=" ", names=['label', 'name'])

        # --------------------------- Cleaning data ---------------------------
        data_csv = pd.read_csv(data_path / f'{task}_val.csv', names=['filename', 'label'])

        # Remove corrupted images if any
        data_csv = data_csv.query("filename not in @corrupted_images")
        
        # Processing filenames to remove special characters and convert to ASCII
        import re
        import unicodedata
        def process_filename(f):
            dirname, filename = f.split('/', 1)
            normalized = unicodedata.normalize('NFKD', filename)
            ascii_filename = ''
            for char in normalized:
                if ord(char) < 128 and ord(char) != 39:
                    ascii_filename += char
                else:
                    replacements = {
                        'ä': 'a', 'ö': 'o', 'ü': 'u', 'ß': 'ss',
                        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
                        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
                        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
                    }
                    ascii_filename += replacements.get(char, '_') # Replace with _ 
        
            ascii_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', ascii_filename)
            return dirname + "/" + ascii_filename
        
        data_csv.loc[:, "filename"] = data_csv["filename"].map(process_filename) # Pandas 3.0

        self.data_csv = data_csv
        self.imgs_path = data_csv["filename"].tolist()
        self.labels = data_csv["label"].tolist()

        # --------------------------- Custom transforms for valid dataset ---------------------------
        self.val_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.imgs_path)

    def get_label_name(self, label):
        return self.classes_csv['name'][label]
    
    def get_label(self, idx):
        return self.labels[idx]
    
    def get_raw(self, idx):
        img = self.imgs_path[idx]
        label = self.labels[idx]

        return {
            'images': img,
            'labels': label
        }
    
    def transform(self, img):
        img = self.val_transforms(img)
        return img

    def __getitem__(self, idx):
        img = Image.open(self.data_path / self.imgs_path[idx])
        label = self.labels[idx]

        img = self.transform(img)
        return {
            'images': img,
            'labels': label
        }


# -------------------------------------------- General Wikiart Dataset --------------------------------------------
class TRAIN_General_WikiartDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.artist_classes_csv = pd.read_csv(data_path / f'artist_class.txt', sep=" ", names=['label', 'name'])
        self.genre_classes_csv = pd.read_csv(data_path / f'genre_class.txt', sep=" ", names=['label', 'name'])
        self.style_classes_csv = pd.read_csv(data_path / f'style_class.txt', sep=" ", names=['label', 'name'])

        # --------------------------- Cleaning data ---------------------------
        
        artist_csv = pd.read_csv(data_path / f'artist_train.csv', names=['filename', 'label'])
        genre_csv = pd.read_csv(data_path / f'genre_train.csv', names=['filename', 'label'])
        style_csv = pd.read_csv(data_path / f'style_train.csv', names=['filename', 'label'])

        # Remove corrupted images if any
        artist_csv = artist_csv.query("filename not in @corrupted_images")
        genre_csv = genre_csv.query("filename not in @corrupted_images")
        style_csv = style_csv.query("filename not in @corrupted_images")
        
        # Processing filenames to remove special characters and convert to ASCII
        import re
        import unicodedata
        def process_filename(f):
            dirname, filename = f.split('/', 1)
            normalized = unicodedata.normalize('NFKD', filename)
            ascii_filename = ''
            for char in normalized:
                if ord(char) < 128 and ord(char) != 39:
                    ascii_filename += char
                else:
                    replacements = {
                        'ä': 'a', 'ö': 'o', 'ü': 'u', 'ß': 'ss',
                        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
                        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
                        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
                    }
                    ascii_filename += replacements.get(char, '_') # Replace with _ 
        
            ascii_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', ascii_filename)
            return dirname + "/" + ascii_filename
        
        artist_csv.loc[:, "filename"] = artist_csv["filename"].map(process_filename) # Pandas 3.0
        genre_csv.loc[:, "filename"] = genre_csv["filename"].map(process_filename) # Pandas 3.0
        style_csv.loc[:, "filename"] = style_csv["filename"].map(process_filename) # Pandas 3.0

        # --------------------------- Merge data ---------------------------
        artist_genre = artist_csv.merge(genre_csv, how='outer', on='filename') # OUTER JOIN
        self.data_csv = artist_genre.merge(style_csv, how='outer', on='filename') # OUTER JOIN
        self.data_csv = self.data_csv.rename(columns={'label_x': 'artist', 'label_y': 'genre', 'label': 'style'})

        # Add dummy class for genre and style
        self.data_csv['artist'] = self.data_csv['artist'].fillna(len(self.artist_classes_csv))
        self.data_csv['genre'] = self.data_csv['genre'].fillna(len(self.genre_classes_csv))
        self.data_csv['style'] = self.data_csv['style'].fillna(len(self.style_classes_csv))
        
        self.imgs_path = self.data_csv["filename"].tolist()
        labels_artist = self.data_csv["artist"].tolist()
        labels_genre = self.data_csv["genre"].tolist()
        labels_style = self.data_csv["style"].tolist()
        self.labels = [(labels_artist[idx], labels_genre[idx], labels_style[idx]) for idx in range(len(self.imgs_path))]

        # --------------------------- Custom transforms for train dataset ---------------------------
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop((224,224)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomVerticalFlip(0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.imgs_path)

    def get_num_classes(self):
        return (len(self.artist_classes_csv)+1, 
                len(self.genre_classes_csv)+1, 
                len(self.style_classes_csv)+1)

    def get_label_artist(self, label):
        return self.artist_classes_csv['name'][label]
        
    def get_label_genre(self, label):
        if label == len(self.genre_classes_csv):
            return "unknown genre"
        return self.genre_classes_csv['name'][label]
        
    def get_label_style(self, label):
        if label == len(self.style_classes_csv):
            return "unknown style"
        return self.style_classes_csv['name'][label]
    
    def get_raw(self, idx):
        img = self.imgs_path[idx]
        label = self.labels[idx]

        return {
            'images': img,
            'labels': label
        }
    
    def transform(self, img):
        img = self.train_transforms(img)
        return img
    
    def __getitem__(self, idx):
        img = Image.open(self.data_path / self.imgs_path[idx])
        label = self.labels[idx] # (artist, genre, style)
        img = self.transform(img)
        return {
            'images': img,
            'labels': torch.tensor(label) # (3)
        }


class VAL_General_WikiartDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.artist_classes_csv = pd.read_csv(data_path / f'artist_class.txt', sep=" ", names=['label', 'name'])
        self.genre_classes_csv = pd.read_csv(data_path / f'genre_class.txt', sep=" ", names=['label', 'name'])
        self.style_classes_csv = pd.read_csv(data_path / f'style_class.txt', sep=" ", names=['label', 'name'])

        # --------------------------- Cleaning data ---------------------------
        
        artist_csv = pd.read_csv(data_path / f'artist_train.csv', names=['filename', 'label'])
        genre_csv = pd.read_csv(data_path / f'genre_train.csv', names=['filename', 'label'])
        style_csv = pd.read_csv(data_path / f'style_train.csv', names=['filename', 'label'])

        # Remove corrupted images if any
        artist_csv = artist_csv.query("filename not in @corrupted_images")
        genre_csv = genre_csv.query("filename not in @corrupted_images")
        style_csv = style_csv.query("filename not in @corrupted_images")
        
        # Processing filenames to remove special characters and convert to ASCII
        import re
        import unicodedata
        def process_filename(f):
            dirname, filename = f.split('/', 1)
            normalized = unicodedata.normalize('NFKD', filename)
            ascii_filename = ''
            for char in normalized:
                if ord(char) < 128 and ord(char) != 39:
                    ascii_filename += char
                else:
                    replacements = {
                        'ä': 'a', 'ö': 'o', 'ü': 'u', 'ß': 'ss',
                        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
                        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
                        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
                    }
                    ascii_filename += replacements.get(char, '_') # Replace with _ 
        
            ascii_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', ascii_filename)
            return dirname + "/" + ascii_filename
        
        artist_csv.loc[:, "filename"] = artist_csv["filename"].map(process_filename) # Pandas 3.0
        genre_csv.loc[:, "filename"] = genre_csv["filename"].map(process_filename) # Pandas 3.0
        style_csv.loc[:, "filename"] = style_csv["filename"].map(process_filename) # Pandas 3.0

        # --------------------------- Merge data ---------------------------
        artist_genre = artist_csv.merge(genre_csv, how='outer', on='filename') # OUTER JOIN
        self.data_csv = artist_genre.merge(style_csv, how='outer', on='filename') # OUTER JOIN
        self.data_csv = self.data_csv.rename(columns={'label_x': 'artist', 'label_y': 'genre', 'label': 'style'})

        # Add dummy class for genre and style
        self.data_csv['artist'] = self.data_csv['artist'].fillna(len(self.artist_classes_csv))
        self.data_csv['genre'] = self.data_csv['genre'].fillna(len(self.genre_classes_csv))
        self.data_csv['style'] = self.data_csv['style'].fillna(len(self.style_classes_csv))
        
        self.imgs_path = self.data_csv["filename"].tolist()
        labels_artist = self.data_csv["artist"].tolist()
        labels_genre = self.data_csv["genre"].tolist()
        labels_style = self.data_csv["style"].tolist()
        self.labels = [(labels_artist[idx], labels_genre[idx], labels_style[idx]) for idx in range(len(self.imgs_path))]


        # --------------------------- Custom transforms for train dataset ---------------------------
        self.val_transforms = transforms.Compose([
            transforms.RandomResizedCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.imgs_path)

    def get_num_classes(self):
        return (len(self.artist_classes_csv)+1, 
                len(self.genre_classes_csv)+1, 
                len(self.style_classes_csv)+1)

    def get_label_artist(self, label):
        if label == len(self.artist_classes_csv):
            return "unknown artist"
        return self.artist_classes_csv['name'][label]
        
    def get_label_genre(self, label):
        if label == len(self.genre_classes_csv):
            return "unknown genre"
        return self.genre_classes_csv['name'][label]
        
    def get_label_style(self, label):
        if label == len(self.style_classes_csv):
            return "unknown style"
        return self.style_classes_csv['name'][label]
    
    def get_raw(self, idx):
        img = self.imgs_path[idx]
        label = self.labels[idx]

        return {
            'images': img,
            'labels': label
        }
    
    def transform(self, img):
        img = self.val_transforms(img)
        return img
    
    def __getitem__(self, idx):
        img = Image.open(self.data_path / self.imgs_path[idx])
        label = self.labels[idx] # (artist, genre, style)
        img = self.transform(img)
        return {
            'images': img,
            'labels': torch.tensor(label) # (3)
        }

def create_dataset(data_path, mode='val', task='artist'):
    '''
    Create a custom dataset for specific tasks (artist, style, genre) or general task.

    Args:
        data_path (str): Path to the dataset directory.
        mode (str): Mode of the dataset ('train' or 'val').
        task (str): Task to perform ('artist', 'style', 'genre', or 'general').
    Returns:
        dataset (Dataset): Custom training or validation dataset for specific task or general task.
    '''
    assert (task in TASKS_LIST), f'Task should be either {TASKS_LIST}\n'
    data_path = Path(data_path)
    
    if task == 'general':
        if mode == 'train':
            return TRAIN_General_WikiartDataset(data_path)
        else:
            return VAL_General_WikiartDataset(data_path)
    else:
        if mode == 'train':
            return TRAIN_Specific_WikiartDataset(data_path, task=task)
        else:
            return VAL_Specific_WikiartDataset(data_path, task=task)

def oversampling_dataset(labels):
    unique, counts = np.unique(labels, return_counts=True)
    class_weights = [1.0/c for c in counts]
    weights_y = [class_weights[i] for i in labels]
    return WeightedRandomSampler(weights_y, len(weights_y))

def create_dataloader(dataset, mode='val', task='artist', batch_size=16, num_workers=0):
    '''
    Create a dataloader for the dataset.

    Args:
        dataset (Dataset): Custom dataset for specific task or general task.
        mode (str): Mode of the dataset ('train' or 'val').
        task (str): Task to perform ('artist', 'style', 'genre', or 'general').
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of workers for the dataloader.
    Returns:
        dataloader (DataLoader): Dataloader for the dataset.
    '''
    assert (task in TASKS_LIST), f'Task should be either {TASKS_LIST}\n'
    assert (mode in ['train', 'val']), f'Mode should be either train or val\n'
    
    if task == 'general':
        if mode == 'train':
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=oversampling_dataset(dataset.labels))

    return dataloader