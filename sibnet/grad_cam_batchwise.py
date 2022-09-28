# add batch
"""
Modified to take batch input.

Original Author: Jacob Gildenblat; github: https://github.com/jacobgil
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import Colormap
from matplotlib import cm
import torchvision.transforms as T
from torch.autograd import Variable

resize = T.Resize(size=256)


def model_flattening(module_tree):
    module_list = []
    children_list = list(module_tree.children())
    if len(children_list) == 0 :
        return [module_tree]
    else:
        for i in range(len(children_list)):
            module = model_flattening(children_list[i])
            module = [j for j in module]
            module_list.extend(module)
        return module_list

class FeatureExtractor(object):
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, output):
        features = []
        self.gradients = []
        # print(self.model)
        for name, module in self.model._modules.items():
            # print(output.shape, name)
            output = module(output)
            # output.requires_grad = True
            # print(name, output.requires_grad)
            if module == self.target_layers:
                # print('a')
                # print(name,(self.target_layers))
                output.register_hook(self.save_gradient)
                features += [output]
        return features, output


class ModelOutputs(object):
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        # output = self.model(x)
        # output = self.model.features.denseblock4.denselayer2.conv2(output)
        # output = self.model.features.norm5(output)
        # output = F.relu(output, inplace=True)
        # output = F.adaptive_avg_pool2d(output, (1, 1)).view(output.size(0), -1)
        # output = self.model.classifier(output)
        return target_activations, output


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      color_map: Colormap = cm.coolwarm,
                      name: str = None):
    """
    Reshape Overlay the GradCam output (mask) to the input img given color map.
    Args:
        img (): Expected to be within [0, 1]. Automatically normalized if dtype is uint8
        mask (): Output of the GradCam.
        color_map (): Matplotlib colormap
        name (): Name of the output img. Default is None, which disables the imwrite.

    Returns:

    """
    if img.dtype == np.uint8:
        img = img / 255.

    # BGR order
    heatmap = color_map(mask)[:, :, 0:3]
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    print('hm_max', heatmap.max())
    # norm to [0,1]

    cam = heatmap + np.float32(img)

    cam = cam / np.max(cam)

    cam *= 255
    out = np.uint8(cam)
    if name is not None:
        # applyColorMap returns a BGR out. So it is not necessary to convert the channel order while writing.
        cv2.imwrite(name, cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    return out, heatmap

    #   grad (32, 128, 8, 8)
    #   weight (32 128,)
    #   target (32 128, 8, 8)
    #    cam (32 8, 8)


class GradCam:
    def __init__(self, model, target_layer_names, cuda_id):
        self.model = model
        self.model.eval()
        self.device = torch.device(f'cuda:{cuda_id}' if cuda_id is not None and torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input_data):
        return self.model(input_data)

    def __call__(self, input_data: torch.Tensor, index=None, output_numpy=False, normalise=False):
        input_data = Variable(input_data)
        features, output = self.extractor(input_data.to(self.device))
        # print(output.shape)
        if index is None:
            index = np.argmax(output.cpu().data.numpy(), axis = -1)
        # print(index)
        one_hot = np.zeros((output.shape[0], output.size()[-1]), dtype=np.float32)
        one_hot[:, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(self.device) * output)
        self.model.zero_grad()
        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        weights = grads_val.mean(axis=(2, 3), keepdims=True)  # [0, :]
        weights = torch.from_numpy(weights).to(self.device)
        cam = F.relu((weights * target).mean(dim=1), inplace=True)
        if normalise:
            one_tensor = torch.ones_like(cam)
            min_val = cam.flatten(1, 2).min(dim=-1).values.unsqueeze(-1).unsqueeze(-1)*one_tensor
            max_val = cam.flatten(1, 2).max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)*one_tensor
            diff = max_val - min_val
            cam = (cam - min_val) / diff
        cam = torch.nan_to_num(cam, nan = 0)

        cam = resize(cam)
        if output_numpy:
            cam = cam.detach().cpu().numpy()

        return cam

