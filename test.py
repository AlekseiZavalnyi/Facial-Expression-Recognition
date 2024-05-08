import torch
import torch.nn as nn
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from tqdm import tqdm

from models import *

def test(model, test_dataloader, class_names, device):
    pred_y = []
    true_y = []
    pred_score = np.full((0, 7), -1)
    model.eval()
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for x, y in tqdm(test_dataloader):
            x = x.to(device)
            predictions = model(x).cpu()
            pred_y.append(torch.argmax(predictions, dim=-1).numpy().tolist())
            pred_score = np.vstack((pred_score, softmax(predictions).numpy()))
            true_y.append(y.cpu().numpy().tolist())

    pred_y = np.array(pred_y).ravel().tolist()
    true_y = np.array(true_y).ravel().tolist()

    conf_m = confusion_matrix(pred_y, true_y)
    class_rep = classification_report(true_y, pred_y, target_names=class_names)
    roc_auc = roc_auc_score(true_y, pred_score.tolist(), multi_class='ovo')
    return conf_m, class_rep, roc_auc


def top_two_accuracy(model, test_dataloader, device):
    model.eval()
    top_two = 0
    num_elements = 0
    with torch.no_grad():
        for x, y in tqdm(test_dataloader):
            x = x.to(device)
            true_y = y.cpu().numpy().tolist()
            num_elements += len(true_y)
            predictions = model(x).cpu().numpy().tolist()
            for prediction, y_i in zip(predictions, true_y):
                if y_i in np.argsort(prediction)[-2:].tolist():
                    top_two += 1

    return top_two / num_elements

if __name__ == '__main__':
    classes = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_mynet_model()
    model = model.to(device)
    path_to_dataset = 'path/to/augmented/dataset'

    test_transform = v2.Compose([v2.ToDtype(torch.float32, scale=True),
                                 v2.Grayscale(1),
                                 v2.Normalize(mean=(0.4809,), std=(0.2066,)
                                ])
    test_ds = ImageFolder(path_to_dataset, transform=test_transform)
    test_dataloader = DataLoader(val_ds, batch_size=8, shuffle=True, drop_last=True)

    conf_m, class_rep, roc_auc = test(model, test_dataloader, classes, device)
    print(f'{conf_m=} \n{class_rep=} \n{roc_auc=}')
    top_two_accuracy = top_two_accuracy(model, test_dataloader, device)
    print(f'{top_two_accuracy}')