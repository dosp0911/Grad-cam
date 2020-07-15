
import torch
from torchvision.models import resnet18, resnet50
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

import matplotlib.pyplot as plt

import numpy as np
import cv2
from pathlib import Path


class GradCam:
	def __init__(self, model:torch.nn.Module, layer:torch.nn.Module):
		self.layer = layer
		self.model = model
		self.add_hooks()

	def add_hooks(self):
		self.layer.register_backward_hook(self.save_gradient)
		self.layer.register_forward_hook(self.save_feature_map)

	def save_gradient(self, m, grad_in, grad_out):
		self.grads = grad_out

	def save_feature_map(self, m, in_, out_):
		self.feature_map = out_

	def compute_score(self):
		weights = np.sum(self.grads[0].numpy(), axis=(2,3)) # weights: [N,C]
		batch_num = len(weights)
		cams = [np.zeros_like(self.feature_map[0,0,...].detach().numpy()) for i in range(batch_num)]
		for b in range(batch_num):
			for w, f in zip(weights[b], self.feature_map[b].detach().numpy()):
				cams[b] += (w * f)
				cams[b] = np.maximum(cams[b], 0)
			cams[b] = cv2.resize(np.array(cams[b]), self.input_size)
			cams[b] = (cams[b] - cams[b].min()) / (cams[b].max() - cams[b].min())

		return cams

	def __call__(self, x, y):
		self.model.eval()
		output = self.model(x)
		self.input_size = x.shape[-2:]
		index = np.argmax(output.cpu().data.numpy(), axis=1)
		one_hot = np.zeros((len(index), output.cpu().data.numpy().shape[-1]), dtype=np.float32)
		for i, idx in enumerate(index):
			one_hot[i][idx] = 1
		one_hot = torch.from_numpy(one_hot).requires_grad_(True)

		one_hot = torch.sum(one_hot * output)

		self.layer.zero_grad()
		self.model.zero_grad()

		one_hot.backward()
		self.cams = self.compute_score()
		self.targets = y

	def visualize(self, imgs, figsize=(5,10)):
		fig, axes = plt.subplots(len(imgs), 2, figsize=figsize)

		for i, (c, img, t) in enumerate(zip(self.cams, imgs, self.targets)):
			heatmap = cv2.applyColorMap(np.uint8(255 * c), cv2.COLORMAP_JET)
			heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
			temp_heatmap = heatmap.copy()
			heatmap[..., 2] = np.where(heatmap[..., 2] > 100, 0, heatmap[..., 2])  #CAM 일부분만 표시
			heatmap = heatmap / 255.0
			blend = heatmap + img
			blend = blend / blend.max() * 255
			plt.imsave(f'cam_{t}.jpg', np.uint8(blend))
			if len(imgs) == 1:
				axes[0].imshow(np.uint8(blend))
				axes[0].set_title(f'cam_{t}'.format(i))
				temp_heatmap = axes[1].imshow(np.uint8(temp_heatmap), cmap='jet')
				fig.colorbar(temp_heatmap, ax=axes[1])
				axes[1].set_title(f'cam_{t}_heatmap'.format(i))
			else:
				axes[i, 0].imshow(np.uint8(blend))
				axes[i, 0].set_title(f'cam_{t}'.format(i))
				temp_heatmap = axes[i, 1].imshow(np.uint8(temp_heatmap), cmap='jet')
				fig.colorbar(temp_heatmap, ax=axes[i, 1])
				axes[i, 1].set_title(f'cam_{t}_heatmap'.format(i))
		plt.show()
		# convert cam_nor into heat map


if __name__ == '__main__':
	y = ['alpaca', 'crocodile', 'dog', 'monkey']
	t_ = Compose([
	              # Resize(512),
	              ToTensor(),
		Normalize(mean=[0.485, 0.456, 0.406],
		          std=[0.229, 0.224, 0.225])
	])

	model = resnet50(True)
	layer = model.layer4[2].conv3

	g_cam = GradCam(model, layer)

	img_path = Path('images')
	img_arrs = [cv2.resize(plt.imread(str(p)), (256, 256)) / 255 for p in sorted(list(img_path.glob('*.jpg')))]
	imgs = [t_(img_arr).float().unsqueeze(0) for img_arr in img_arrs]
	x = torch.cat(imgs)
	g_cam(x.float(), y)
	g_cam.visualize(img_arrs)