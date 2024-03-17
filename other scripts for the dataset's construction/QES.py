import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--mask", required=True, type=str, help="The path of the input mask.")
args = parser.parse_args()

t1 = (1 / 16)
t2 = (15 / 16)

mask = (cv2.imread(args.mask, 0).astype(np.float32)/255.0)
QES = ((mask>t2).sum()/((mask>t1).sum()+1e-8))

print('The Quality Evaluation Score is:', QES)


