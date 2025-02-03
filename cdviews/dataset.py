import json
import os

import torch
from torch.utils.data import Dataset


class ViewLabelDataset(Dataset):
    def __init__(self,
                 cfg,
                 mode='train',
                 num_img=10):

        self.mode = mode
        self.dataset = cfg.dataset
        self.pair_file = cfg.view_labeling_file.format(self.dataset)
        self.ans_file = cfg.ScanQA_file[self.mode] if self.dataset == 'ScanQA' else cfg.SQA_file[self.mode]
        self.image_feat_path = cfg.feature_folder
        self.num_img = num_img

        if self.mode == 'train':
            self._filter_empty_pair()
        else:
            self.ans_list = json.load(open(self.ans_file))

    def _filter_empty_pair(self):
        ans_list = json.load(open(self.ans_file))
        self.view_labeling_info = json.load(open(self.pair_file))

        self.ans_list = []
        for ans in ans_list:
            question_id = ans['question_id']
            cur_pair = self.view_labeling_info.get(str(question_id), None)

            if cur_pair is None:
                continue
            elif len(cur_pair['Positive']) == 0 or len(cur_pair['Negative']) == 0:
                continue
            else:
                self.ans_list.append(ans)

    def __len__(self):
        # return 20
        return len(self.ans_list)

    def __getitem__(self, idx):
        ans = self.ans_list[idx]
        scene_id = ans['scene_id']
        question = ans['question']
        question_id = ans['question_id']
        cur_feat_file = os.path.join(self.image_feat_path, '{}.pth'.format(scene_id))
        cur_feats = torch.load(cur_feat_file)

        if self.mode == 'train':
            cur_pair = self.view_labeling_info[str(question_id)]
            pos_images, neg_images = cur_pair['Positive'], cur_pair['Negative']
            num_img = min(len(pos_images), len(neg_images), self.num_img // 2)
            image_files = [pos_images[i] for i in torch.randperm(len(pos_images))[:num_img]] + \
                          [neg_images[i] for i in torch.randperm(len(neg_images))[:num_img]]
            labels = torch.cat([torch.ones(num_img),
                                torch.zeros(num_img)]).float()
        else:
            labels = None
            image_files = [image for image in cur_feats.keys()]    # score all the image files for inference

        image_feats = []
        for image_file in image_files:
            image_feats.append(cur_feats[image_file].unsqueeze(0))
        image_feats = torch.cat(image_feats)
        return question_id, question, image_feats, labels, image_files
