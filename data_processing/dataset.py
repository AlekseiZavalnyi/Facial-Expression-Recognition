import torch
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms import v2
import cv2
import numpy as np
import os
from tqdm import tqdm
from data_aug import FaceAugmentation

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, class_names, transform, n_dim, img_size=64):
        self.path = path
        self.classes = dict(zip(class_names, [i for i in range(len(class_names))]))
        self.transform = transform
        self.n_dim = n_dim
        self.img_size = img_size
        self.x = torch.ones([self.dataset_size(), self.n_dim, self.img_size, self.img_size])
        self.y = torch.full((self.x.size(0),), -1)

    def dataset_size(self):
        return sum([len(os.listdir(os.path.join(self.path, foldername))) for foldername in os.listdir(self.path)])

    @staticmethod
    def numpy_to_tensor(image):
        return torch.from_numpy(image).permute(2, 0, 1)

    @staticmethod
    def tensor_to_numpy(image):
        return image.permute(1, 2, 0).numpy()

    def load_and_preprocess(self, fa:FaceAugmentation, folder_path_to_write=None):
        idx = 0
        for emotion, code in self.classes.items():
            path_to_folder = os.path.join(self.path, emotion)
            for filename in tqdm(os.listdir(path_to_folder), desc=f'emotion {emotion}'):
                image = cv2.imread(os.path.join(path_to_folder, filename))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if fa.detector is not None and fa.predictor is not None:
                    image = fa.make_face_vertical(image)
                elif fa.detector is not None:
                    image = fa.crop_face(image)
                if image is None:
                    continue
                if fa.predictor is not None:
                    if image is not None:
                        e, m = fa.crop_eyes_and_mouth(image)
                        if e is not None:
                            em = np.vstack((e, m))
                            image_tensor = self.numpy_to_tensor(em)
                        else:
                            continue
                else:
                    image_tensor = self.numpy_to_tensor(image)

                if self.n_dim == 1:
                    image_tensor = rgb_to_grayscale(image_tensor)
                    image_tensor = image_tensor.unsqueeze(0)

                image_tensor = v2.Resize((self.img_size, self.img_size))(image_tensor)

                if folder_path_to_write is not None:
                    image = self.tensor_to_numpy(image_tensor).astype('uint8')
                    cv2.imwrite(f'{folder_path_to_write}/{emotion}/emotion_{idx}.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                else:
                    self.x[idx, ...] = image_tensor.to(torch.uint8)
                    self.y[idx] = code
                idx += 1

        if folder_path_to_write is None:
            self.x = self.x[:idx].to(torch.uint8)
            self.y = self.y[:idx].to(torch.uint8)

    def transform_batch(self, x):
        x = x.to(torch.float32)
        x /= 255.
        return self.transform(x)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        batch = self.x[idx, :].clone()
        batch = self.transform(batch)
        return batch, self.y[idx]