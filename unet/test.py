import torch
import cv2
import numpy as np
from unet import UNet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from PIL import Image

from torchvision import models, transforms
device = torch.device('cuda')
model = UNet()
model.load_state_dict(torch.load('Unetmodel1.pkl'))
model=model.to(device)
model.eval()

# for x in range(8001,8002):
img='./data/test/imgs1/{}.bmp'.format(10001)
img = cv2.imread(img, 0)
img = np.array(img).reshape(128, 128, 1)
img = img / 255.
img = np.transpose(img, [2, 0, 1]).reshape(1,1,128,128)
img = torch.tensor(img, dtype=torch.float32).to(device)

mask='./data/test/masks1/{}.bmp'.format(10001)
mask=cv2.imread(mask, 0)/255.


res=model(img).cpu().detach().numpy()[0,0,:,:]
cv2.imwrite('./test_pre10001.bmp', res*225)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(mask,cmap ='gray')

# plt.axis('off')
# plt.savefig('real_mask.png',bbox_inches='tight', pad_inches=0)
plt.title('real_mask')

plt.subplot(1,2,2)
plt.imshow(res,cmap='gray')
# plt.axis('off')
# plt.savefig('pre_mask.png',bbox_inches='tight', pad_inches=0)
plt.title('pre_mask')
plt.show()
rmse=np.sqrt(mean_squared_error(mask,res))
print(rmse)
