import os
import numpy as np
import torch
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

def get_annotations_boxes_from_xml(dir):
    tree = ET.parse(dir)
    root = tree.getroot()

    annotations, labels = [], []

    for neighbor in root.iter('bndbox'):
        xmin = int(neighbor.find('xmin').text)
        ymin = int(neighbor.find('ymin').text)
        xmax = int(neighbor.find('xmax').text)
        ymax = int(neighbor.find('ymax').text)

        annotations.append([xmin, ymin, xmax, ymax])
    
    for neighbor in root.iter('object'):
        label = neighbor.find('name').text
        if label == 'without_mask': 
            labels.append(2)
        else:
            labels.append(1)

    return annotations, labels


class FaceMaskDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'images'))))
        self.anns = list(sorted(os.listdir(os.path.join(root, 'annotations'))))
        self.img_dir = os.path.join(root, 'images')
        self.ann_dir  = os.path.join(root, 'annotations')
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        curr_img_dir = os.path.join(self.img_dir, self.imgs[idx])
        curr_ann_dir = os.path.join(self.ann_dir, self.anns[idx])

        image = Image.open(curr_img_dir, mode='r').convert('RGB')
        boxes, labels = get_annotations_boxes_from_xml(curr_ann_dir)

        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, category_ids=labels)

        tenn = transforms.ToTensor()
        image = tenn(image)

        return image, boxes, labels

    def collate_fn(self, batch):
        return tuple(zip(*batch))