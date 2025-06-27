from data import DatasetMerger, load_mnist, load_notmnist, load_cifar10
from torchvision.models import resnet18


def main():
    feature_extractor = resnet18()
