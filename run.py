from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import numpy as np
from tqdm.notebook import tqdm

from model import *




def images_loader(image_names):

    res = []

    for i,name in enumerate(image_names):

        image = Image.open(name)

        if i == 0:
            imsize = np.array(image.size)[::-1]//10

        loader = transforms.Compose([transforms.Resize(imsize),  
                             transforms.CenterCrop(imsize),
                             transforms.ToTensor()])  
        image = loader(image).unsqueeze(0)
        res.append(image)

    return res


if __name__ == "__main__":

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	names = ["./input/1.jpg",
	         "./style/VanGog.jpeg",
	         "./style/Pic.jpg"]

	content_img, style1_img, style2_img = images_loader(names)

	model = MST(content_img, style1_img, style2_img, device)

	input_img = content_img.clone() #torch.rand(content_img.shape)#
	output = model.run(input_img)

	model.show()
	model.save(name='house')