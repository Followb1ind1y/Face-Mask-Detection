"""
Trains a Object Detection Model for Face Mask Detection
"""

import os
import torch
import torchvision
import splitfolders
import numpy as np
import torch.optim as optim
import segmentation_models_pytorch as smp
import utils,engine,data_setup,predictions
import albumentations as A

from torch.optim import lr_scheduler
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Setup hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 4

root_dir = '/content/FaceMask'
splitfolders.ratio(root_dir, output="train_split", seed=42, ratio=(0.8, 0.1, 0.1))
output_dir = '/content/train_split'

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create data augmentation
data_transform = A.Compose([A.HorizontalFlip(p=0.5),
                            A.ShiftScaleRotate(p=0.5),
                            A.RandomBrightnessContrast(p=0.3),
                            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
                            ],
                            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
                      
# Create DataLoaders from data_setup.py
image_datasets = {x: data_setup.FaceMaskDataset(root=os.path.join(output_dir, x), transforms=data_transform) 
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                              shuffle=True, collate_fn=image_datasets[x].collate_fn) 
                                              for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

#  Create Object Detection Model
weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)

## Model inItialization
model = model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer_RCNN = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
exp_lr_scheduler_RCNN = lr_scheduler.StepLR(optimizer_RCNN, step_size=7, gamma=0.1)

# Trainer
trainer = engine.Trainer(model=model,
                         dataloaders=dataloaders,
                         epochs=30,
                         metric=None,
                         criterion=None, 
                         optimizer=optimizer_RCNN,
                         scheduler=exp_lr_scheduler_RCNN,
                         save_dir="RCNN_Model_Output",
                         device=device)

## Training process
model_results = trainer.train_model()

## Evaluate the model
images, boxes, labels = next(iter(dataloaders['test']))
images = list(img.to(device) for img in images)

model.eval()
predictions = model(images)
predictions = predictions.remove_low_risk_box(predictions=predictions, threshold=0.5)
predictions = predictions.apply_nms(predictions, 0.5)

## Display the predictions
utils.display_images(Output1=utils.display_boundary(images[0], predictions[0]['boxes'], 
                                                    predictions[0]['labels'], predictions[0]['scores']),
                    Output2=utils.display_boundary(images[1], predictions[1]['boxes'], 
                                                   predictions[1]['labels'], predictions[1]['scores']))