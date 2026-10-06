import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, 'PigImageData')


class DatasetInfo(object):
    dataset_info = {
        'train': {
            'image_dir': os.path.join(DATA_ROOT, 'Train'),
            'anno_dir': os.path.join(DATA_ROOT, 'Train_annotations.json'),
            'split': 'train',
            'data_file': os.path.join(DATA_ROOT, 'weight_data.json')
        },
        'val': {
            'image_dir': os.path.join(DATA_ROOT, 'Val'),
            'anno_dir': os.path.join(DATA_ROOT, 'Val_annotations.json'),
            'split': 'val',
            'data_file': os.path.join(DATA_ROOT, 'weight_data.json')
        },
        'test': {
            'image_dir': os.path.join(DATA_ROOT, 'Test'),
            'anno_dir': os.path.join(DATA_ROOT, 'Test_annotations.json'),
            'split': 'test',
            'data_file': os.path.join(DATA_ROOT, 'weight_data.json')
        },

        'bf_train': {
            'image_dir': os.path.join(DATA_ROOT, 'Train'),
            'anno_dir': os.path.join(DATA_ROOT, 'Train_annotations.json'),
            'split': 'train',
            'data_file': os.path.join(DATA_ROOT, 'bf_data.json')
        },
        'bf_val': {
            'image_dir': os.path.join(DATA_ROOT, 'Val'),
            'anno_dir': os.path.join(DATA_ROOT, 'Val_annotations.json'),
            'split': 'val',
            'data_file': os.path.join(DATA_ROOT, 'bf_data.json')
        },
        'bf_test': {
            'image_dir': os.path.join(DATA_ROOT, 'Test'),
            'anno_dir': os.path.join(DATA_ROOT, 'Test_annotations.json'),
            'split': 'test',
            'data_file': os.path.join(DATA_ROOT, 'bf_data.json')
        }
    }