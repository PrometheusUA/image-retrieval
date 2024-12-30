import argparse
import sys
from os.path import normpath, abspath, join as path_join

sys.path.append('./')

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from utils.data import get_loaders, get_default_transforms
from utils.main import read_py, get_device
from utils.constants import TEST_CSV, TEST_DIR


def predict(checkpoint, train_config, data, output, use_512_leak, distractor_threshold):
    train_config = read_py(train_config)

    model = train_config.CONFIG['model'](**train_config.CONFIG['model_params'])
    forward = train_config.CONFIG['forward'](model, **train_config.CONFIG['forward_params'])
    forward.load_state_dict(torch.load(checkpoint)['state_dict'])
    forward.eval()
    
    if normpath(abspath(data)) == normpath(abspath(TEST_CSV)):
        loaders_config = train_config.CONFIG['loaders']
        if 'test_transforms' not in loaders_config:
            loaders_config['test_transforms'] = get_default_transforms('test')
        _, _, test_loader = get_loaders(**loaders_config)
    else:
        print(normpath(abspath(data)))
        print(normpath(abspath(TEST_CSV)))
        raise NotImplementedError('Only test data is supported for now')
    
    device = get_device()
    forward = forward.to(device)

    all_preds = []
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        preds = forward.test_step(batch)
        preds = preds.detach().cpu().numpy()
        all_preds.append(preds)

    all_preds_probas = np.concatenate(all_preds, axis=0)

    test_df = pd.read_csv(data)
    test_df['class'] = np.argsort(all_preds_probas, axis=1)[:, -5:][:, ::-1].tolist()
    test_df['class'] = test_df['class'].apply(lambda classes: list(map(str, classes)))

    if use_512_leak:
        test_df['512x512'] = test_df['file_id'].apply(lambda file_id: (Image.open(path_join(TEST_DIR, f'{file_id}.jpg')).size) == (512, 512))
        test_df.loc[test_df['512x512'], 'class'] = test_df.loc[test_df['512x512'], 'class'].apply(lambda classes: '-1 ' + ' '.join(classes[:-1]))
        test_df.loc[~test_df['512x512'], 'class'] = test_df.loc[~test_df['512x512'], 'class'].apply(lambda classes: ' '.join(classes))
    else:
        cutoff_location = np.sum(all_preds_probas > distractor_threshold, axis=1)
        for i in range(all_preds_probas.shape[0]):
            if cutoff_location[i] < 5:
                chosen_classes = test_df.loc[i, 'class']
                for j in range(1, 5 - cutoff_location[i]):
                    chosen_classes[5 - j] = chosen_classes[5 - j - 1]
                chosen_classes[cutoff_location[i]] = -1
                test_df.loc[i, 'class'] = ' '.join(map(str, chosen_classes))
        
        test_df['class'] = test_df['class'].apply(lambda classes: ' '.join(map(str, classes)) if not isinstance(classes, str) else classes)

    test_df[['file_id', 'class']].to_csv(output, index=False)
                                        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a prediction for the test dataset')
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--train_config', type=str, help='Path to the training configuration')
    parser.add_argument('--output', type=str, help='Path to save the predictions')
    parser.add_argument('--data', type=str, help='Path to the data to make a prediction on', default=TEST_CSV)
    parser.add_argument('--use_512_leak', action='store_true', default=False)
    parser.add_argument('--distractor_threshold', type=float, default=0.5)
    args = parser.parse_args()

    predict(args.checkpoint, args.train_config, args.data, args.output, args.use_512_leak, args.distractor_threshold)
