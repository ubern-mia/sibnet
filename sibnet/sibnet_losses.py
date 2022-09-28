import itertools
import numpy as np
import torch
import torch.nn as nn
from typing import List
from skimage.morphology import label, binary_erosion


class ClassDistinctivenessLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.cossim = nn.CosineSimilarity(dim=1)
        self.device = device

    def view_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(tensor.shape[0], -1)

    def forward(self, sal_tensor_list: [torch.Tensor]) -> torch.Tensor:
        loss_list = torch.Tensor([]).to(self.device)
        for sal_comb in itertools.combinations(sal_tensor_list, 2):
            loss_list = torch.cat((loss_list, torch.unsqueeze(torch.abs(self.cossim(self.view_tensor(sal_comb[0]), self.view_tensor(sal_comb[1]))).mean(), dim=0)))
        return torch.mean(loss_list)


class SpatialCoherence(nn.Module):
    def __init__(self, device, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.device = device
        # self.device = torch.device(f'cuda:{cuda_id}' if cuda_id is not None and torch.cuda.is_available() else 'cpu')

    def pixel_neighbor_squared_distance(self, x, y, image):
        dif = 0
        for i in range(torch.max(torch.Tensor([x - self.kernel_size // 2]), torch.Tensor([0])).int(),
                       torch.min(torch.Tensor([x + self.kernel_size // 2 + 1]), torch.Tensor([image.shape[0]])).int()):
            for j in range(torch.max(torch.Tensor([y - self.kernel_size // 2]), torch.Tensor([0])).int(),
                           torch.min(torch.Tensor([y + self.kernel_size // 2 + 1]), torch.Tensor([image.shape[1]])).int()):
                dif += torch.square(image[x, y] - image[i, j]).sum()  # sum in to aggregate if more than 3 channels
        return dif

    def sc_per_image_per_channel(self, image):
        sdif = 0
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                sdif += self.pixel_neighbor_squared_distance(x, y, image)
        return sdif

    @staticmethod
    def threshold_otsu_numpy(image=None, nbins=256):
        # Check if the image has more than one intensity value; if not, return that
        # value
        if image is not None:
            first_pixel = image.ravel()[0]
            if np.all(image == first_pixel):
                return first_pixel

        counts, bin_edges = np.histogram(image, bins=nbins, density=True)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        # class probabilities for all possible thresholds
        weight1 = np.cumsum(counts)
        weight2 = np.cumsum(counts[::-1])[::-1]
        # class means for all possible thresholds
        mean1 = np.cumsum(counts * bin_centers) / weight1
        mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

        # Clip ends to align class 1 and class 2 variables:
        # The last value of ``weight1``/``mean1`` should pair with zero values in
        # ``weight2``/``mean2``, which do not exist.
        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        idx = np.argmax(variance12)
        threshold = bin_centers[idx]

        return threshold

    def mask(self, sal_map_tensor, process=binary_erosion):
        sal_map_numpy = sal_map_tensor.detach().cpu().numpy()
        threshold = self.threshold_otsu_numpy(sal_map_numpy)
        binary_mask = (sal_map_numpy > threshold).astype(np.int32)
        processed_mask = process(binary_mask)
        labels = label(processed_mask, background=0)
        label_stack = np.zeros([1, np.max(labels)] + list(labels.shape))
        for i in range(1, np.max(labels)):
            label_stack[0, i - 1] = labels == i
        return torch.from_numpy(label_stack.astype(np.float32)).to(self.device)

    def forward(self, sal_tensor_list, device):
        """

        :param sal_tensor_list: a list of saliency tensors calculated for each class
        :return:
        """
        sum_sdiff = torch.Tensor(0).to(device)
        for sal_tensor in sal_tensor_list:
            for batch_index in range(sal_tensor.shape[0]):
                label_stack = self.mask(sal_tensor[batch_index])
                masked_sal_tensor = torch.squeeze(label_stack * sal_tensor[batch_index:batch_index + 1])
                # iterate over the connected components in the attr. map. maske_sal_tensor has shape
                # [conn_comp, img_x, img_y]
                for conn_comp_idx in range(masked_sal_tensor.shape[0]):
                    sum_sdiff += self.sc_per_image_per_channel(torch.squeeze(masked_sal_tensor[conn_comp_idx, :, :]))

        return sum_sdiff
