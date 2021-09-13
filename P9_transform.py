from torchvision import transforms
from PIL import Image
# Python 中的用法

img_path = "dataset/train/ants_image/6240329_72c01e663e.jpg"
img = Image.open(img_path)
tensor_tranform = transforms.ToTensor()
tensor_img = tensor_tranform(img)
print(type(tensor_img))