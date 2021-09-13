from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
image_path = "dataset/train/bees_image/16838648_415acd9e3f.jpg"
img = Image.open(image_path)
image_array = np.array(img)
print(type(img))
print(image_array.shape)

writer.add_image('test', image_array, 2, dataformats="HWC")

# y = x
for i in range(99):
    writer.add_scalar("y=x*x", i * i, i)
writer.close()
