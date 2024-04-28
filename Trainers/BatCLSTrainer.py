import yaml
from torch import nn
import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from BatDataLoader import BatDataLoader
from models.FGVC_HERBS.builder import MODEL_GETTER
import torch.nn.functional as F
import munch
import timm
from transformers import AutoModel


#
# import warnings
# warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def suppression(target: torch.Tensor, threshold: torch.Tensor, temperature: float = 2):
    """
    target size: [B, S, C]
    threshold: [B',]
    """
    B = target.size(0)
    target = torch.softmax(target / temperature, dim=-1)
    # target = 1 - target
    return target

class BatCLSTrainer:

    def __init__(self, config):

        self.config = munch.Munch(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.load_model()

        bat_loader = BatDataLoader(config)
        self.train_loader, self.val_loader, self.test_loader = bat_loader.create_loaders()

        self.init_optimizer()

        self.init_metrics()

        self.loss_func = getattr(nn, config['loss'])()

        self.best_ckpt_metric = np.inf if self.config['checkpoint_metric_goal'] == 'minimize' else -np.inf

        self.epoch = 0


    def load_model(self):

        try:
            self.model = timm.create_model(model_name=self.config['model_name'], pretrained=self.config['pretrained'],
                                           num_classes=self.config['num_classes']).to(self.device)
        except:
            print('No pretrained model')


    def init_optimizer(self):
        self.optimizer = getattr(torch.optim, self.config['optimizer'])
        self.optimizer = self.optimizer(self.model.parameters(), lr=float(self.config['learning_rate']), weight_decay=float(self.config['weight_decay']))

    def init_metrics(self):
        self.train_metrics = {'train_loss': [], 'train_acc': []}
        self.val_metrics = {'val_loss': [], 'val_acc': []}

        self.epoch_train_metrics = {'train_loss': [], 'train_acc': []}
        self.epoch_val_metrics = {'val_loss': [], 'val_acc': []}

    def batch_to_device(self, batch):
        # move all data to device
        for k in batch:
            if isinstance(batch[k], torch.Tensor) \
                    and batch[k].device.type != self.device:
                batch[k] = batch[k].to(self.device)

        return batch

    def save_checkpoint(self, epoch_performance):

        # latest checkpoint
        torch.save(self.model.state_dict(), os.path.join(self.config['experiment_dir'], 'latest_checkpoint.pt'))
        print(f'latest checkpoint saved')

        # best checkpoint
        if epoch_performance > self.best_ckpt_metric:
            torch.save(self.model.state_dict(), os.path.join(self.config['experiment_dir'], 'best_checkpoint.pt'))
            self.best_ckpt_metric = epoch_performance
            print(f'best checkpoint saved with {self.config["checkpoint_metric"]} of {epoch_performance}')

    def load_checkpoint(self, version='latest'):
        self.model.load_state_dict(torch.load(os.path.join(self.config['experiment_dir'], f'{version}_checkpoint.pt')))

    def calc_metrics(self, outputs, labels):
        preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        return accuracy_score(labels, preds)

    def update_epoch_train_metrics(self, outputs, labels):
        accuracy = self.calc_metrics(outputs, labels)
        self.epoch_train_metrics['train_acc'].append(accuracy)

    def update_epoch_val_metrics(self, outputs, labels):
        accuracy = self.calc_metrics(outputs, labels)
        self.epoch_val_metrics['val_acc'].append(accuracy)

    def compute_metrics(self):
        # train metrics
        for k, v in self.epoch_train_metrics.items():
            self.epoch_train_metrics[k] = np.mean(self.epoch_train_metrics[k])

        # val metrics
        for k, v in self.epoch_val_metrics.items():
            self.epoch_val_metrics[k] = np.mean(self.epoch_val_metrics[k])

    def reset_metrics(self):
        for k, v in self.epoch_train_metrics.items():
            self.epoch_train_metrics[k] = []
        for k, v in self.epoch_val_metrics.items():
            self.epoch_val_metrics[k] = []

    def train_step(self, batch):
        self.model.train()

        imgs, labels = batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(imgs)
        loss = self.loss_func(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            outs = self.model(imgs)
            self.update_epoch_train_metrics(outs, labels)
            self.epoch_train_metrics['train_loss'].append(loss.cpu().detach().numpy())

    def val_step(self, batch):

        with torch.no_grad():
            self.model.eval()

            imgs, labels = batch
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            output = self.model(imgs)
            loss = self.loss_func(output, labels)

        self.update_epoch_val_metrics(output, labels)
        self.epoch_val_metrics['val_loss'].append(loss)

    def test_step(self, batch):

        with torch.no_grad():
            self.model.eval()

            imgs, labels = batch
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            output = self.model(imgs)

            return output, labels

    def train(self):

        for epoch in range(self.config['n_epochs']):

            # train
            for batch in tqdm(self.train_loader, desc=f'train epoch {epoch}', leave=True):
                self.train_step(batch)

            # val
            for batch in tqdm(self.val_loader, desc=f'validation epoch {epoch}', leave=True):
                self.val_step(batch)

            self.compute_metrics()

            print('-------------------')
            print(f'epoch: {epoch}/{self.config["n_epochs"]}')
            for k, v in self.epoch_train_metrics.items():
                self.train_metrics[k].append(v)
                print(f'{k}: {v}')

            print('---')
            for k, v in self.epoch_val_metrics.items():
                self.val_metrics[k].append(v)
                print(f'{k}: {v}')
            print('---')

            # save checkpoint
            epoch_performance = self.epoch_val_metrics[f'val_{self.config["checkpoint_metric"]}']
            self.save_checkpoint(epoch_performance)
            print('-------------------')

            # reset metrics
            self.reset_metrics()

        # plot and save convergence curves

        plots_dir = os.path.join(self.config['experiment_dir'], 'train_plots')
        os.makedirs(plots_dir, exist_ok=True)

        for metric in self.train_metrics.keys():
            metric_name = metric.split('_')[1]
            plt.figure()
            plt.plot(self.train_metrics[f'train_{metric_name}'], label=f'train')
            plt.plot(self.val_metrics[f'val_{metric_name}'], label=f'val')
            metric_name = 'Accuracy' if  metric_name == 'acc' else metric_name
            plt.title(f'{self.config["training_name"]} - {metric_name}')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.legend(loc='upper left')
            plt.savefig(os.path.join(plots_dir, f'{metric}_plot'))
        plt.show()

    def test(self):

        # load model checkpoint
        self.load_checkpoint(version='best')

        # init dataset
        all_outputs = []
        all_labels = []
        for batch in tqdm(self.test_loader, desc='test', leave=True):
            outputs, labels = self.test_step(batch)
            all_outputs.append(outputs)
            all_labels.append(labels)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        all_preds = torch.argmax(torch.softmax(all_outputs, dim=1), dim=1)

        test_plot_dir = os.path.join(self.config['experiment_dir'], 'test_plots')
        os.makedirs(test_plot_dir, exist_ok=True)

        # calc accuracy and plot confusion matrix
        # all_outputs = all_outputs.cpu().detach().numpy()
        all_labels = all_labels.cpu().detach().numpy()
        all_preds = all_preds.cpu().detach().numpy()

        acc = accuracy_score(all_labels, all_preds)

        print(f'test accuracy: {acc}')

        # confusion matrix
        plt.figure(figsize=(20, 20))
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(all_labels), yticklabels=np.unique(all_labels))
        plt.title('{} \nConfusion Matrix - {} \nAccuracy: {:.2f}%'.format(self.config['training_name'], 'test', acc * 100))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(test_plot_dir, f'confusion_matrix.png'))
        plt.show()
