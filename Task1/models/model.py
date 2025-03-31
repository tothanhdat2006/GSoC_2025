import torch
import torch.nn as nn
from timm import create_model
from torchvision.models import resnet50, ResNet50_Weights

MODEL_DICT = {
    'Resnet': 'resnet50',
    'ViT': 'vit_base_patch16_224'
}

class CNN_RNN(nn.Module):
    def __init__(self, num_artist_classes, num_genre_classes, num_style_classes, lstm_hidden_dim=512):
        super().__init__()

        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_artist_classes = num_artist_classes
        self.num_genre_classes = num_genre_classes
        self.num_style_classes = num_style_classes

        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.cnn = nn.Sequential(*(list(resnet.children())[:-2]))
        self.pool = nn.AdaptiveAvgPool2d(7)

        self.lstm = nn.LSTM(input_size=2048, 
                            hidden_size=lstm_hidden_dim, 
                            num_layers=1, 
                            batch_first=True)
        
        self.artist_fc = nn.Linear(lstm_hidden_dim, num_artist_classes)
        self.genre_fc = nn.Linear(lstm_hidden_dim, num_genre_classes)
        self.style_fc = nn.Linear(lstm_hidden_dim, num_style_classes)

    def forward(self, images, labels=None):
        batch_size = images.shape[0]

        # CNN
        cnn_features = self.pool(self.cnn(images)) # (B, 2048, 7, 7)
        cnn_features = cnn_features.reshape(batch_size, 49, 2048) # (B, 49, 2048)

        # RNN
        _, (h_n, _) = self.lstm(cnn_features) # h_n: (num_layers, B, lstm_hidden_dim)
        final_feature = h_n[-1] # (B, lstm_hidden_dim)

        out_artist = self.artist_fc(final_feature) # (B, num_artist_classes)
        out_genre = self.genre_fc(final_feature) # (B, num_genre_classes)
        out_style = self.style_fc(final_feature) # (B, num_style_classes)
        return out_artist, out_genre, out_style
    

def load_model(args, config):
    """
    Load the model weights from the specified path.
    """

    assert args.model_weights is not None, "Model weights path is not specified."
    assert args.model_weights.endswith('.pt'), "Model weights file must be a .pt file."
    assert args.task in ['artist', 'style', 'genre', 'general'], "Invalid task specified."

    if args.task == 'artist':
        model = create_model(MODEL_DICT[args.model_name], pretrained=True, num_classes=config['NUM_ARTIST_CLASSES'])
    elif args.task == 'style':
        model = create_model(MODEL_DICT[args.model_name], pretrained=True, num_classes=config['NUM_GENRE_CLASSES'])
    elif args.task == 'genre':
        model = create_model(MODEL_DICT[args.model_name], pretrained=True, num_classes=config['NUM_STYLE_CLASSES'])
    elif args.task == 'general':
        model = CNN_RNN(
            num_artist_classes=config['NUM_ARTIST_CLASSES']+1,
            num_genre_classes=config['NUM_GENRE_CLASSES']+1,
            num_style_classes=config['NUM_STYLE_CLASSES']+1
        )

    try:
        state_dict = torch.load(args.model_weights, map_location=config['DEVICE'])
        model.load_state_dict(state_dict['model_state_dict'])
    except RuntimeError:
        print("Loading model weights failed. Please check the model name, model weights file and try again.")
        raise
    
    model.eval()
    return model