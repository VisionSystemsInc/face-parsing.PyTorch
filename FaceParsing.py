import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from face_parsing.model import BiSeNet

class FaceParsing(object):
    def __init__(self, model_path=None):
        if model_path is None:
            # REVIEW this is a terrible default
            model_path = '../../../external/data/models/face_parsing/face_parsing_79999_iter.pth'

        self.net = BiSeNet(n_classes=19)
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.label_by_idx = collections.OrderedDict(
                [(-1, 'unlabeled'), (0, 'background'), (1, 'skin'),
                 (2, 'l_brow'), (3, 'r_brow'), (4, 'l_eye'), (5, 'r_eye'),
                 (6, 'eye_g (eye glasses)'), (7, 'l_ear'), (8, 'r_ear'), (9, 'ear_r (ear ring)'),
                 (10, 'nose'), (11, 'mouth'), (12, 'u_lip'), (13, 'l_lip'),
                 (14, 'neck'), (15, 'neck_l (necklace)'), (16, 'cloth'),
                 (17, 'hair'), (18, 'hat')])
        self.idx_by_label = {v: k for k,v in self.label_by_idx.iteritems()}

    def to_tensor(self, images):
        # images : N,H,W,C numpy.array
        return self.transform(images)

    def label_for_idx(idx):
        return self.label_by_idx[idx]

    def idx_for_label(label):
        return self.idx_by_label[label]

    def parse_face(self, images, device=0):
        # images : list of PIL Images
        # device : which CUDA device to run on
        #
        # returns parsings : list of PIL Images
        #
        # 0 - background, 1 - skin, 2 - l_brow, 3 - r_brow, 4 - l_eye, 5 - r_eye, 6 - eye_g (eye glasses),
        # 7 - l_ear, 8 - r_ear, 9 - ear_r (ear ring), 10 - nose, 11 - mouth, 12 - u_lip, 13 - l_lip,
        # 14 - neck, 15 - neck_l (necklace), 16 - cloth, 17 - hair, 18 - hat

        # move the network to the correct device
        self.net.to('cuda:{}'.format(device))

        assert all(im.size[0] == im.size[1] for im in images)
        in_sizes = [im.size[0] for im in images] # im is square

        pt_images = []
        for img in images:
            # seems to work best with images around 512
            img = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(img)
            pt_images.append(img)
        pt_images = torch.stack(pt_images, dim=0)

        # move the data to the device
        pt_images = pt_images.to('cuda:{}'.format(device))

        out = self.label(pt_images)
        parsings = out.cpu().numpy().argmax(axis=1).astype(np.uint8)

        parsings = [Image.fromarray(parsing).resize((in_size, in_size))
                for parsing, in_size in zip(parsings, in_sizes)]

        return parsings # list of PIL Images

    def label(self, pt_images):
        # N,H,W,C torch.tensor
        with torch.no_grad():
            return self.net(pt_images)[0]

