import mmcv
import numpy as np
import os.path as osp
import cv2
import matplotlib.pyplot as plt
import random
import argparse

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
# from tensorboardX import SummaryWritter
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
import datetime
import time
import pickle
from multiprocessing import Pool
from torch import multiprocessing
_TIMESTAMP_BIAS = 600
_TIMESTAMP_START = 840  # 60*14min
_TIMESTAMP_END = 1860  # 60*31min
_FPS = 30


torch.set_num_threads(0)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()

cfg.merge_from_file(args.config)
cfg.CUDA = True
# device = torch.device('cuda')
# random.seed(1)
# Firstly Load Proposal


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class Tracking_Proposal(object):
    def __init__(self,
                 img_prefix,
                 proposal_path,
                 video_stat_file,
                 new_length=32,
                 new_step=2,
                 with_pysot=True,
                 with_shuffle=False,
                 num_gpus=1,
                 num_workers=1):
        self.img_prefix = img_prefix
        self.new_step = new_step
        self.new_length = new_length
        self.proposal_dict, self.org_proposal_dict = self.load_proposal(proposal_path)
        self.video_stats = dict([tuple(x.strip().split(' ')) for x in open(video_stat_file)])
        self.with_model = False
        self.with_pysot = with_pysot
        self.with_shuffle = with_shuffle
        self.result_dict = {}
        self.num_gpus = num_gpus
        self.num_workers = num_workers


    def load_proposal(self, path):
        proposal_dict = mmcv.load(path)
        convert_dict = {}
        for key, value in proposal_dict.items():
            video_id, frame = key.split(',')
            if convert_dict.get(video_id,None) is None:
                convert_dict[video_id] = {}
            elif convert_dict[video_id].get(frame, None) is None:
                convert_dict[video_id][frame] = {}
            convert_dict[video_id][frame] = value
        return convert_dict, proposal_dict

    def _load_image(self, directory, image_tmpl, modality, idx):
        if modality in ['RGB', 'RGBDiff']:
            return mmcv.imread(osp.join(directory, image_tmpl.format(idx)))
        elif modality == 'Flow':
            x_imgs = mmcv.imread(
                osp.join(directory, image_tmpl.format('x', idx)),
                flag='grayscale')
            y_imgs = mmcv.imread(
                osp.join(directory, image_tmpl.format('y', idx)),
                flag='grayscale')
            return [x_imgs, y_imgs]
        else:
            raise ValueError(
                'Not implemented yet; modality should be '
                '["RGB", "RGBDiff", "Flow"]')

    def spilit_proposal_dict(self):
        keys = list(self.proposal_dict.keys())
        if self.with_shuffle:
            random.shuffle(keys)
        shuffle_dict = [(key, self.proposal_dict[key]) for key in keys]
        video_per_gpu = int(len(shuffle_dict) // (self.num_gpus*self.num_workers))
        self.sub_shuffle_dict_list = [shuffle_dict[i * video_per_gpu:(i + 1) * video_per_gpu] if i != self.num_gpus - 1
                                 else shuffle_dict[i * video_per_gpu:]
                                 for i in range(self.num_gpus*self.num_workers)]
        self.shuffle_dict = shuffle_dict

    def spilit_org_proposal_dict(self):
        keys = list(self.org_proposal_dict.keys())
        if self.with_shuffle:
            random.shuffle(keys)
        shuffle_dict = [(key, self.org_proposal_dict[key]) for key in keys if '-5KQ66BBWC' in key and 900<=int(key[-4:])<=910]
        video_per_gpu = int(len(shuffle_dict) // (self.num_gpus*self.num_workers))
        self.sub_shuffle_dict_list = [shuffle_dict[i * video_per_gpu:(i + 1) * video_per_gpu] if i != (self.num_gpus*self.num_workers) - 1
                                      else shuffle_dict[i * video_per_gpu:]
                                      for i in range(self.num_gpus*self.num_workers)]
        # [print((i*video_per_gpu,(i+1)*video_per_gpu)) if i != (self.num_gpus*self.num_workers-1) else print(i*video_per_gpu,len(shuffle_dict)) for i in range(self.num_gpus*self.num_workers)]

        self.shuffle_dict = shuffle_dict

    def tracking(self, index):
        self.index = index
        # self.spilit_dict()
        self.spilit_org_proposal_dict()

        # for video_id, frame_info in self.sub_shuffle_dict_list[index]:
        #     cnt = 0
        len_proposal = sum([len(proposals) for frame_info,proposals in self.sub_shuffle_dict_list[index]])
        print("Process:{},proposal lenght:{}".format(self.index, len_proposal))
        cnt_time = 0
        cnt_proposal = 0
        begin = time.time()
        cnt = 0
        for id_frame, (frame_info, proposals) in enumerate(self.sub_shuffle_dict_list[index]):
            video_id, timestamp = frame_info.split(',')
            indice = _FPS * (int(timestamp) - _TIMESTAMP_START) + 1
            image_tmpl = 'img_{:05}.jpg'
            # forward tracking

            for ik, proposal in enumerate(proposals):
                # begin = time.time()
                width, height  = [int(ll) for ll in self.video_stats[video_id].split('x')]
                ROI = np.array([int(x) for x in  (proposal * np.array([
                    width, height, width, height, 1
                ]))[:4]])
                track_window = tuple(np.concatenate([ROI[:2],ROI[-2:]-ROI[:2]],axis=0).tolist())

                ann_frame = self._load_image(osp.join(self.img_prefix,
                                                      video_id),
                                                      image_tmpl, 'RGB', indice)
                if False:
                    plt.imshow(ann_frame[:,:,::-1])
                    color = (random.random(), random.random(), random.random())
                    rect = plt.Rectangle((track_window[0],track_window[1]),
                                         track_window[2],
                                         track_window[3], fill=False,
                                         edgecolor=color, linewidth=5)
                    plt.gca().add_patch(rect)
                    plt.show()
                # Forcasting Tracking
                p = indice - self.new_step
                for i, ind in enumerate(
                        range(-2, -(self.new_length+1), -self.new_step)):
                    unann_frame = self._load_image(osp.join(self.img_prefix,
                                                            video_id),
                                                            image_tmpl, 'RGB', p)
                    if self.with_pysot:
                        track_window = self.pysot_tracking_roi(track_window,
                                                                  key_frame=ann_frame,
                                                                  tracked_frame=unann_frame)
                    else:
                        track_window = self.cv2_tracking_roi(track_window,
                                                              org_frame=ann_frame,
                                                              tracked_frame=unann_frame)
                    self.result_dict['{},{},{},{}'.format(video_id, '{:04d}'.format(int(timestamp)), ik, ind)] = np.array(track_window) / np.array([width, height, width, height])

                    ann_frame = unann_frame.copy()
                    p -= self.new_step
                    if ik == 9 and frame_info == '-5KQ66BBWC4,0902':
                        print(np.array(ROI) / np.array([width, height, width, height]), np.array(track_window) / np.array([width, height, width, height]))
                track_window = tuple(np.concatenate([ROI[:2], ROI[-2:] - ROI[:2]], axis=0).tolist())
                ann_frame = self._load_image(osp.join(self.img_prefix,
                                                      video_id),
                                                      image_tmpl, 'RGB', indice)
                self.result_dict['{},{},{},{}'.format(video_id, '{:04d}'.format(int(timestamp)), proposal,
                                                      0)] = np.array(ROI) / np.array([width, height, width, height])

                # Backcasting Tracking
                p = indice + self.new_step
                for i, ind in enumerate(
                        range(0, self.new_length-2, self.new_step)):
                    unann_frame = self._load_image(osp.join(self.img_prefix,
                                                            video_id),
                                                            image_tmpl, 'RGB', p)
                    if self.with_pysot:
                        track_window = self.pysot_tracking_roi(track_window,
                                                               key_frame=ann_frame,
                                                               tracked_frame=unann_frame)
                    else:
                        track_window = self.cv2_tracking_roi(track_window,
                                                             org_frame=ann_frame,
                                                             tracked_frame=unann_frame)
                    self.result_dict['{},{},{},{}'.format(video_id, '{:04d}'.format(int(timestamp)), ik, ind+2)] =np.array(track_window) / np.array([width, height, width, height])

                    ann_frame = unann_frame
                    p += self.new_step

                end = time.time()
                cnt_time +=(end-begin)
                cnt_proposal += 1
                avg_time = (end-begin)/cnt_proposal
                left_time = (len_proposal-cnt_proposal)*avg_time
                # print(left_time)
                if cnt_proposal % 100== 0:
                    print('Process:{}, length_process:{},  video_id:{}, frame:{}, proposal_id:{}th, proposal_len:{}, per_cost_time:{} , left_time:{}'.format(self.index, len_proposal,
                        video_id, timestamp,  ik, len(proposals), avg_time, datetime.timedelta(seconds=int(left_time))))

            # print('cnt->>>{}!!!!'.format(cnt))
            #     cnt += 1
            #     if cnt >= 1:
            #         break
            # break



    def build_model(self):
        model = ModelBuilder()
        # load model
        model.load_state_dict(torch.load(args.snapshot,
                                         map_location=lambda storage, loc: storage.cpu()))
        # import ipdb
        # ipdb.set_trace()
        device = torch.device('cuda:{}'.format(int(self.index//self.num_workers)) if cfg.CUDA else 'cpu')
        print(device)
        model.eval().to(device)
        # build tracker
        tracker = build_tracker(model)
        return tracker

    def init_tracker(self, track_window, frame):
        self.tracking_model = self.build_model()
        self.tracking_model.init(frame, track_window)

    def pysot_tracking_roi(self, track_window, key_frame, tracked_frame, vis=False):
        if not self.with_model:
            self.init_tracker(track_window, key_frame)
            self.with_model = True

        outputs = self.tracking_model.track(tracked_frame)
        # import ipdb
        # ipdb.set_trace()
        if 'polygon' in outputs:
            cv2.polylines(tracked_frame, [polygon.reshape((-1, 1, 2))],
                          True, (0, 255, 0), 3)
            mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
            mask = mask.astype(np.uint8)
            mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
            tracked_frame = cv2.addWeighted(tracked_frame, 0.77, mask, 0.23, -1)
        else:
            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(tracked_frame, (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          (0, 255, 0), 1) 
        bbox = list(map(int, outputs['bbox']))
        bbox = [bbox[0],bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        if vis:
            plt.imshow(tracked_frame[:,:,::-1])
            plt.show()
        return bbox

    def cv2_tracking_roi(self, track_window, org_frame, tracked_frame, vis=True):
        x, y, w, h = track_window
        roi = org_frame[y:y+h, x:x+w]

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 0.,0.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        hsv = cv2.cvtColor(tracked_frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        if vis:
            # Draw it on image
            # pts = cv2.boxPoints(ret)
            # pts = np.int0(pts)
            # img2 = cv2.polylines(tracked_frame, [pts], True, 255, 2)
            x, y, w, h = track_window
            img2 = cv2.rectangle(tracked_frame, (x, y), (x + w, y + h), 255, 1)
            plt.imshow(img2[:,:,::-1])
            plt.show()
        return track_window

    def save_result(self):
        with open('./pkl_results/train_results_dict_{}.pkl'.format(self.index),'wb') as f:
            pickle.dump(self.result_dict, f)



data_root = '/home/yhxu/code/mmaction/data/ava/rawframes/'
def multi_track(index):
    tracking_inst = Tracking_Proposal(
        img_prefix=data_root,
        proposal_path='/home/yhxu/code/mmaction/data/ava/ava_dense_proposals_train.FAIR.recall_93.9.pkl',
        video_stat_file='/home/yhxu/code/mmaction/data/ava/ava_video_resolution_stats.csv',
        new_length=32,
        new_step=2,
        with_shuffle=False
    )
    tracking_inst.tracking(index)
    tracking_inst.save_result()


if __name__ == '__main__':


    # multi_track(0)

    # multiprocessing.set_start_method('spawn')
    # pool = multiprocessing.Pool(processes=8)
    # results = []
    # for rank in range(32):
    #     results.append(pool.apply_async(multi_track, (rank,)))
    #
    # rst = [result.get() for result in results]
    # import ipdb
    # ipdb.set_trace()

    #s
    # processes = []
    # for rank in range(8):
    #     p = multiprocessing.Process(target=multi_track,args=(rank,))
    #     p.start()
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()
    #
    # import ipdb
    # ipdb.set_trace()

    ctx = multiprocessing.get_context('spawn')
    workers = [ctx.Process(target=multi_track,args=(rank,)) for rank in range(1)]
    index_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for w in workers:
        w.daemon = True
        w.start()
    print('end->>!!!')
    for i in range(1000):
        index_queue.put(i)
    for i in range(1000):
        idx, res = result_queue.get()
    #import ipdb
    #ipdb.set_trace()
    # for
    # #     rst = result_queue.get()    #
    # print('saving_results')


    # tracking_inst.save_result()
