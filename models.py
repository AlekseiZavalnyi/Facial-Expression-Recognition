import torch
import torch.nn as nn
import torchvision
from torchsummary import summary


class VggFace(nn.Module):

    def __init__(self):
        super(VggFace, self).__init__()
        self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=[7, 7], stride=(1, 1))
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(in_features=4096, out_features=7, bias=True)

    def forward(self, data):
        x1 = self.conv1_1(data)
        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu1_2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv2_1(x5)
        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu2_2(x8)
        x10 = self.pool2(x9)
        x11 = self.conv3_1(x10)
        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14)
        x16 = self.relu3_3(x15)
        x17 = self.pool3(x16)
        x18 = self.conv4_1(x17)
        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21)
        x23 = self.relu4_3(x22)
        x24 = self.pool4(x23)
        x25 = self.conv5_1(x24)
        x26 = self.relu5_1(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu5_2(x27)
        x29 = self.conv5_3(x28)
        x30 = self.relu5_3(x29)
        x31 = self.pool5(x30)
        x32 = self.fc6(x31)
        x33_preflatten = self.relu6(x32)
        x33 = x33_preflatten.view(x33_preflatten.size(0), -1)
        x34 = self.fc7(x33)
        x35 = self.relu7(x34)
        prediction = self.fc8(x35)
        return prediction


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1, padding_mode='reflect'), # ->32*32
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # ->16*16
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # ->8*8
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # ->4*4
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.25)
        )

        self.fcon_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256 * 4 * 4, out_features=7)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.fcon_layers(self.conv_layers(x))

    def predict_proba(self, x):
        return self.softmax(self.forward(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_resnet50_model(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads resnet50(pretrained for emotion recognition)
        model weights from the given path
    """
    net = torchvision.models.resnet50(weights='IMAGENET1K_V2')
    net.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.LeakyReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.25),

        nn.Linear(512, 128),
        nn.LeakyReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.25),

        nn.Linear(128, 7)
    )
    if weights_path:
        net.load_state_dict(torch.load(weights_path))
    return net


def get_mobilenet_model(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads mobilenet_v2(pretrained for emotion recognition)
        model weights from the given path
    """
    net = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V2')
    net.classifier = nn.Sequential(
        nn.Linear(1280, 512),
        nn.LeakyReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.25),
        nn.Linear(512, 7)
    )
    if weights_path:
        net.load_state_dict(torch.load(weights_path))
    return net


def get_vggface_model(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, vggface(pretrained for emotion recognition)
        model weights from the given path
    """
    net = VggFace()
    if weights_path:
        net.load_state_dict(torch.load(weights_path))
    return net


def get_mynet_model(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads mynet(pretrained for emotion recognition)
        model weights from the given path
    """
    net = MyNet()
    if weights_path:
        net.load_state_dict(torch.load(weights_path))
    return net

if __name__ == '__main__':
    model = get_mynet_model()
    print(summary(model, (1, 64, 64)))

    model = get_mobilenet_model()
    print(summary(model, (3, 224, 224)))

    model = get_resnet50_model()
    print(summary(model, (3, 224, 224)))

    model = get_vggface_model()
    print(summary(model, (3, 224, 224)))