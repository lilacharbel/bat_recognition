from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import numpy as np


class BatDataLoader:
    def __init__(self, data_root, batch_size=256, resize=None):
        self.data_root = data_root
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize(resize) if resize else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])
        self.train_loader, self.val_loader, self.test_loader = self.create_loaders()

    def create_loaders(self):
        dataset = datasets.ImageFolder(root=self.data_root, transform=self.transform)
        bat_indices = {bat: [] for bat in dataset.class_to_idx.keys()}

        # Group indices by class
        for idx, (_, label) in enumerate(dataset.imgs):
            class_name = dataset.classes[label]
            bat_indices[class_name].append(idx)

        # Stratified split
        train_indices, val_indices, test_indices = [], [], []
        for indices in bat_indices.values():
            np.random.shuffle(indices)
            n_train = int(len(indices) * 0.7)
            n_val = int(len(indices) * 0.15)
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])

        # Create subsets for each dataset
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        # DataLoader for each subset
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


# Example usage
if __name__ == '__main__':
    data_root = '../data/processed_data/'
    bat_loader = BatDataLoader(data_root, batch_size=256)
    train_loader, val_loader, test_loader = bat_loader.create_loaders()

    # Iterate over one batch to see the data
    for images, labels in train_loader:
        print(images.shape, labels)