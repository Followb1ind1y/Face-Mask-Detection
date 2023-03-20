"""
Contains various utility functions for PyTorch model training, saving and result display.
"""

import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Save the model to the target dir
def save_model(model: torch.nn.Module, target_dir: str, epoch: int):
    """
    Saves a PyTorch model to a target directory.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    check_point_name = f"model_epoch_{epoch}"
    model_save_path = target_dir_path / check_point_name

    # Save the model state_dict()
    #print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

# Plot the training curve
def plot_curve(results: dict, epochs: int):
    #train_ious, val_ious = np.array(results["train_iou"]), np.array(results["val_iou"])
    train_losses = np.array(results["train_loss"])

    plt.plot(np.arange(epochs, step=1), train_losses, label='Train loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()
    
def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def display_boundary(image, boxes, labels, score = None):

    label_to_name = {1: 'Masked', 2: 'No Mask', }
    label_to_color = {1: 'palegreen', 2: 'red'}

    transform = torchvision.transforms.ToPILImage()
    image = transform(image)
    boxes = boxes.tolist()
    labels = labels.tolist()

    img_bbox = ImageDraw.Draw(image)
    new_font = ImageFont.truetype(os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSansCondensed-Bold.ttf'), 10)

    for idx in range(len(boxes)):
        img_bbox.rectangle(boxes[idx], outline=label_to_color[labels[idx]], width=2)
        if score == None: 
            img_bbox.text((boxes[idx][0], boxes[idx][1]-15), label_to_name[labels[idx]], 
                          font=new_font, align ="left", fill=label_to_color[labels[idx]]) 
        else:
            img_bbox.text((boxes[idx][0], boxes[idx][1]-15), label_to_name[labels[idx]]+' '+ f"{score[idx].item():.2%}", 
                          font=new_font, align ="left", fill=label_to_color[labels[idx]])
    
    return image

# helper function for image visualization
def display_images(**images):
    """
    Plot images in one rown
    """
    num_images = len(images)
    plt.figure(figsize=(15,15))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, num_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=15)
        plt.imshow(image)
    plt.show()