from mmdet.apis import DetInferencer
from PIL import Image
import os


# Choose to use a config
config_path = '../work_dirs/faster-rcnn_r50_fpn_1x_voc/faster-rcnn_r50_fpn_1x_voc.py'
# Setup a checkpoint file to load
checkpoint = '../work_dirs/faster-rcnn_r50_fpn_1x_voc/epoch_12.pth'

inferencer = DetInferencer(model=config_path, weights=checkpoint)

origin_dir = './rpn_vis/origin/'
out_dir = './rpn_vis/final'

os.makedirs(out_dir, exist_ok=True)
files = os.listdir(origin_dir)
image_files = [f for f in files if f.endswith('.jpg')]
# print(image_files)

for image_file in image_files:
    input_path = os.path.join(origin_dir, image_file)
    inferencer(input_path, out_dir=out_dir)