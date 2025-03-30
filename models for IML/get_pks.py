import os
import pickle
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='your dataset dir path')
args = parser.parse_args()

if not os.path.exists('pks'):
    os.makedirs('pks')

data_list = [{'filename': img_name, 'ann': {'seg_map': img_name[:-4]+'.png'}} for img_name in tqdm(os.listdir(os.path.join(args.dataset, 'imgs')))]

dataset_name = args.dataset[:-1].split('/')[-1] if args.dataset.endswith('/') else args.dataset.split('/')[-1]

with open('pks/'+dataset_name+'.pk', 'wb') as f:
    pickle.dump(data_list, f)
