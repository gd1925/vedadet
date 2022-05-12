####
import argparse

import cv2
import numpy as np
import torch

import os 

from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights, ProgressBar
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine

from typing import List
import time

# img_dir = "/home/uie94465/vedadet/tools/1/"
# img_dir = "/home/uie94465/Desktop/save_frames/Head_Avail/Yongki_Hong_HETE_cut/"
img_dir = "/home/uie94465/vedadet/data/WIDERFace/WIDER_val/0--Parade/"

images_list = os.listdir(img_dir)
images_list = np.sort(images_list)
####
# Specify the path to model config and checkpoint file
config_file = '/home/uie94465/vedadet/configs/infer/tinaface/tinaface_r50_fpn_gn_dcn.py'
# checkpoint_file = '/home/uie94465/vedadet/weights/pretrain/tinaface_r50_fpn_gn_dcn.pth'
checkpoint_file = '/home/uie94465/vedadet/weights/retrained/epoch_143_weights.pth'

prog_bar = ProgressBar(len(images_list))

####
# video = mmcv.VideoReader('/home/uie94465/vedadet/1.mp4')
####
class AverageFpsCounter:
  def __init__(self, window_size: int):
    self.window_size:int = window_size
    self._times_s: List[float] = []
    
  def update(self, new_time_s: float) -> float:
    if len(self._times_s) < self.window_size:
        self._times_s.append(new_time_s)
    else:
        self._times_s.pop(0)
        self._times_s.append(new_time_s)

    if len(self._times_s) < 2:
        return 0
    else:
        n_frames = len(self._times_s) - 1
        duration = self._times_s[-1] - self._times_s[0]
        if duration == 0:
            return 0
        return n_frames / duration

def prepare(cfg):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    engine = build_engine(cfg.infer_engine)

    engine.model.to(device)
    load_weights(engine.model, cfg.weights.filepath)

    data_pipeline = Compose(cfg.data_pipeline)
    return engine, data_pipeline, device
cfg = Config.fromfile(config_file)
engine, data_pipeline, device = prepare(cfg)


# def plot_result(result, imgfp, outfp='out.jpg'):
def plot_result(result, imgfp, outfp,fps_text):
    font_scale = 0.5
    bbox_color = 'green'
    text_color = 'green'
    thickness = 1

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    img = imread(imgfp)

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness)
        cv2.putText(img, " ",(bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        cv2.putText(img, fps_text, (10, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0))
    # cv2.imshow("frames",img)
    # outfp=outfp+'.jpg'
    imwrite(img, outfp)
## For FPS calculation
# # test a video and show the results

# def test(engine, data_loader):
#     engine.eval()
#     results = []
#     dataset = data_loader.dataset
#     prog_bar = ProgressBar(len(dataset))
#     for i, data in enumerate(data_loader):

#         with torch.no_grad():
#             result = engine(data)[0]

#         results.append(result)
#         batch_size = len(data['img_metas'][0].data)
#         for _ in range(batch_size):
#             prog_bar.update()
#     return results
avg_fps_counter = AverageFpsCounter(30)
def main():
    
    
    for frame in images_list:
        avg_fps = avg_fps_counter.update(time.time())
        print(img_dir+frame)
        data = dict(img_info=dict(filename=img_dir+frame), img_prefix=None)

        data = data_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if device != 'cpu':
            # scatter to specified GPU
            data = scatter(data, [device])[0]
            batch_size = len(data['img_metas'])
            # print("batch_size",batch_size)
        else:
            # just get the actual data from DataContainer
            data['img_metas'] = data['img_metas'][0].data
            data['img'] = data['img'][0].data

        # batch_size = len(data['img_metas'][0].data)
        # print(batch_size)
        # output of engine at the first index
        result = engine.infer(data['img'], data['img_metas'])[0]
        avg_fps = avg_fps_counter.update(time.time())
        fps_text = f"FPS: {avg_fps:.02f}"
        # cv2.putText(img_show, fps_text, (10, 20),
        #             cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0))
        plot_result(result,img_dir+frame,frame,fps_text)
        # result = test(engine, data)

if __name__== '__main__':
    main()