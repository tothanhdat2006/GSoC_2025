def get_config():
    return {
        'DATA_PATH': './data',
        'MODEL_WEIGHTS': './weights',
        'MODEL_NAME': 'resnet50',  # resnet50, vit_base_patch16_224
        'MODE': 'val',  # train, test, or val
        'TASK': 'artist',  # artist, style, genre, or general
        'NUM_ARTIST_CLASSES': 23,
        'NUM_GENRE_CLASSES': 10,
        'NUM_STYLE_CLASSES': 27,
        'BATCH_SIZE': 16,
        'NUM_WORKERS': 0,
        'OUTLIER_THRESHOLD': 0.25,  # Threshold for outlier detection
        'DEVICE': 'cuda',
        'SEED': 86,
    }