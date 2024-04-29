from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np


class Sub_Dataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class BatDataLoader:
    def __init__(self, config):
        self.config = config

        self.data_root = config['data_root']
        self.batch_size = config['batch_size']

        normalize = transforms.Normalize(mean=tuple(config['mean']), std=tuple(config['std'])) if not config['mean'] == 'None' else None

        self.transform = transforms.Compose([
            transforms.Resize(config['input_size']),
            transforms.RandomRotation(degrees=config['rotation_degrees']),
            transforms.RandomAdjustSharpness(sharpness_factor=config['sharpness_factor']),
            transforms.ColorJitter(brightness=config['brightness'], contrast=config['contrast'], saturation=config['saturation'], hue=config['hue']),
            transforms.ToTensor(),
            normalize if normalize is not None else lambda x: x,
        ])

        self.transform_train = transforms.Compose([
            transforms.Resize(config['input_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize if normalize is not None else lambda x: x,
        ])

    def create_loaders(self):
        dataset = datasets.ImageFolder(root=self.data_root)
        bat_indices = {bat: [] for bat in dataset.class_to_idx.keys()}

        # Group indices by class
        for idx, (_, label) in enumerate(dataset.imgs):
            class_name = dataset.classes[label]
            bat_indices[class_name].append(idx)

        # Stratified split
        np.random.seed(self.config['seed'])
        train_indices, val_indices, test_indices = [], [], []
        for indices in bat_indices.values():
            np.random.shuffle(indices)
            n_train = int(len(indices) * self.config['train_size'])
            n_val = int(len(indices) * self.config['val_size'])
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])

        # Create subsets for each dataset
        train_dataset = Sub_Dataset(Subset(dataset, train_indices), transform=self.transform_train)
        val_dataset = Sub_Dataset(Subset(dataset, val_indices), transform=self.transform)
        test_dataset = Sub_Dataset(Subset(dataset, test_indices), transform=self.transform)

        # DataLoader for each subset
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        return train_loader, val_loader, test_loader


# Example usage
if __name__ == '__main__':
    import yaml
    import matplotlib.pyplot as plt

    with open('../configs/resnet50.yaml', 'r') as f:
        config = yaml.safe_load(f)

    bat_loader = BatDataLoader(config)
    train_loader, val_loader, test_loader = bat_loader.create_loaders()

    plt.figure()
    # Iterate over one batch to see the data
    for images, labels in test_loader:
        print(images.shape, labels)
        plt.cla()
        plt.imshow(images[0].permute(1, 2, 0))
        plt.title(labels[0])
        plt.show()
