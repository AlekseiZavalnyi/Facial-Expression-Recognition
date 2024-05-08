from data.data_aug import FaceAugmentation
from data.dataset import Dataset
import dlib
import torch
import os

if __name__ == '__main__':
    class_names = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    path_to_dataset = 'path/to/unprepared_dataset'
    path_to_write_dataset = None
    face_detector = dlib.get_frontal_face_detector()
    landmarks_predictor = dlib.shape_predictor('path/to/predictor')
    fa = FaceAugmentation(face_detector, landmarks_predictor)

    write_dataset = False
    if write_dataset:
        path_to_write_dataset = 'path/to/write/augmented/dataset'

    ds = Dataset(path_to_dataset, class_names, transform=None, n_dim=3)
    ds.load_and_preprocess(fa)
    torch.save(ds.x.to(torch.uint8), 'path/to/tensor_data.pt')
    torch.save(ds.y.to(torch.uint8), 'path/to/tensor_targets.pt')