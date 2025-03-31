def get_config():
    return {
        "DATA_PATH": '/kaggle/input/ngoa-opendata',
        "REGION": 'full',
        "SIZE": '224,224',
        "ROTATION_ANGLE": '0',
        "QUALITY": 'default',
        "IMAGE_FORMAT": 'jpg',
        "OUTPUT_PATH": './data',
        "SAVE_FIGURE": True,
        "FIGURE_NAME": './experiment/images/query_results.png',
        "SEED": 86
    }