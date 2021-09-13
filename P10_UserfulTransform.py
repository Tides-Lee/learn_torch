from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img = Image.open("dataset/train/ants_image/6240329_72c01e663e.jpg")
writer = SummaryWriter("logs")

# ToTensor
tensor_transform = transforms.ToTensor()
img_tensor = tensor_transform(img)
print(type(img))
writer.add_image("test", img_tensor)

# Normalize
print(img_tensor[0][0][0])
tans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = tans_norm(img_tensor)
print(img_norm[0][0][0])

writer.add_image("Normalize", img_norm)

# ReSize
trans_size = transforms.Resize((512, 512))
img_resize = trans_size(img)
# print(img.size)
# print(img_resize.size)
# img_resize.show()
img_resize = tensor_transform(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)
writer.close()
