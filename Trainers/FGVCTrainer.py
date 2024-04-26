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
from models.builder import MODEL_GETTER
import torch.nn.functional as F

import sys
sys.path.append('../FGVC-HERBS')
from eval import evaluate
import munch

def suppression(target: torch.Tensor, threshold: torch.Tensor, temperature: float = 2):
    """
    target size: [B, S, C]
    threshold: [B',]
    """
    B = target.size(0)
    target = torch.softmax(target / temperature, dim=-1)
    # target = 1 - target
    return target

class FGVCTrainer:

    def __init__(self, config):

        self.config = munch.Munch(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = MODEL_GETTER[config['model_name']](
            use_fpn=config['use_fpn'],
            fpn_size=config['fpn_size'],
            use_selection=config['use_selection'],
            num_classes=config['num_classes'],
            num_selects=config['num_selects'],
            use_combiner=config['use_combiner'],
        ).to(self.device) # about return_nodes, we use our default setting

        bat_loader = BatDataLoader(config)
        self.train_loader, self.val_loader, self.test_loader = bat_loader.create_loaders()

        self.init_optimizer()

        self.init_metrics()

        self.loss_func = getattr(nn, config['loss']) #nn.CrossEntropyLoss()

        self.best_ckpt_metric = np.inf if self.config['checkpoint_metric_goal'] == 'minimize' else -np.inf

        self.epoch = 0

        # todo
        if self.config['use_amp']:
            self.scaler = torch.cuda.amp.GradScaler()
            self.amp_context = torch.cuda.amp.autocast

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

    def train_step(self, batch, batch_id):
        self.model.train()

        imgs, labels = batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        with self.amp_context():

            outs = self.model(imgs)

            loss = 0.
            for name in outs:

                if "FPN1_" in name:
                    if self.config['lambda_b0'] != 0:
                        aux_name = name.replace("FPN1_", "")
                        gt_score_map = outs[aux_name].detach()
                        thres = torch.Tensor(self.model.selector.thresholds[aux_name])
                        gt_score_map = suppression(gt_score_map, thres, self.temperature)
                        logit = F.log_softmax(outs[name] / self.temperature, dim=-1)
                        loss_b0 = nn.KLDivLoss()(logit, gt_score_map)
                        loss += self.config['lambda_b0'] * loss_b0
                    else:
                        loss_b0 = 0.0

                elif "select_" in name:
                    if not self.config['use_selection']:
                        raise ValueError("Selector not use here.")
                    if self.config['lambda_s'] != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, self.config['num_classes']).contiguous()
                        loss_s = nn.CrossEntropyLoss()(logit, labels.unsqueeze(1).repeat(1, S).flatten(0))
                        loss += self.config['lambda_s'] * loss_s
                    else:
                        loss_s = 0.0

                elif "drop_" in name:
                    if not self.config['use_selection']:
                        raise ValueError("Selector not use here.")

                    if self.config['lambda_n'] != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, self.config['num_classes']).contiguous()
                        n_preds = nn.Tanh()(logit)
                        labels_0 = torch.zeros([self.config['batch_size'] * S, self.config['num_classes']]) - 1
                        labels_0 = labels_0.to(self.device)
                        loss_n = nn.MSELoss()(n_preds, labels_0)
                        loss += self.config['lambda_n'] * loss_n
                    else:
                        loss_n = 0.0

                elif "layer" in name:
                    if not self.config['use_fpn']:
                        raise ValueError("FPN not use here.")
                    if self.config['lambda_b'] != 0:
                        ### here using 'layer1'~'layer4' is default setting, you can change to your own
                        loss_b = nn.CrossEntropyLoss()(outs[name].mean(1), labels)
                        loss += self.config['lambda_b'] * loss_b
                    else:
                        loss_b = 0.0

                elif "comb_outs" in name:
                    if not self.config['use_combiner']:
                        raise ValueError("Combiner not use here.")

                    if self.config['lambda_c'] != 0:
                        loss_c = nn.CrossEntropyLoss()(outs[name], labels)
                        loss += self.config['lambda_c'] * loss_c

                elif "ori_out" in name:
                    loss_ori = F.cross_entropy(outs[name], labels)
                    loss += loss_ori

            if batch_id < len(self.train_loader) - self.n_left_batchs:
                loss /= self.config['update_freq']
            else:
                loss /= self.n_left_batchs

        """ = = = = calculate gradient = = = = """
        if self.config['use_amp']:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        """ = = = = update model = = = = """
        if (batch_id + 1) % self.config['update_freq'] == 0 or (batch_id + 1) == len(self.train_loader):
            if self.config['use_amp']:
                self.scaler.step(self.optimizer)
                self.scaler.update()  # next batch
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        self.model.eval()
        with torch.no_grad():
            outs = self.model(imgs)['comb_outs']
            self.update_epoch_train_metrics(outs, labels)
            self.epoch_train_metrics['train_loss'].append(loss.cpu().detach().numpy())

    def val_step(self, batch):

        with torch.no_grad():
            self.model.eval()

            imgs, labels = batch
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            output = self.model(imgs)['comb_outs']

        # loss = self.loss_func(output, labels)

        self.update_epoch_val_metrics(output, labels)
        self.epoch_val_metrics['val_loss'].append(0)

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

            self.temperature = 0.5 ** (epoch // 10) * self.config['temperature']
            self.n_left_batchs = len(self.train_loader) % self.config['update_freq']

            # train
            for batch_in, batch in enumerate(tqdm(self.train_loader, desc=f'train epoch {epoch}', leave=True)):
                self.train_step(batch, batch_in)

            # val
            # todo: use their eval function
            # self.model.eval()
            # with torch.no_grad():
            #     acc, eval_name, accs = evaluate(self.config, self.model, self.val_loader, self.device)
            #     self.val_metrics['val_acc'].append(acc)

            for batch_in, batch in enumerate(tqdm(self.val_loader, desc=f'validation epoch {epoch}', leave=True)):
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
        self.load_checkpoint(version='latest')

        results = {}
        for partition in ['train', 'test']:

            # init dataset
            test_loader = self.dataset.get_dataloader(partition)

            all_outputs = []
            all_labels = []
            for batch in tqdm(test_loader, desc=partition, leave=True):
                outputs, labels = self.test_step(batch)
                all_outputs.append(outputs)
                all_labels.append(labels)

            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            all_preds = torch.argmax(torch.softmax(all_outputs, dim=1), dim=1)

            test_plot_dir = os.path.join(self.config['experiment_dir'], 'test_plots')
            os.makedirs(test_plot_dir, exist_ok=True)

            # calc accuracy and plot confusion matrix
            all_outputs = all_outputs.cpu().detach().numpy()
            all_labels = all_labels.cpu().detach().numpy()
            all_preds = all_preds.cpu().detach().numpy()

            acc = accuracy_score(all_labels, all_preds)

            print(f'{partition} accuracy: {acc}')
            results[partition] = float(acc) * 100


            # confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(all_labels, all_preds)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(all_labels), yticklabels=np.unique(all_labels))
            plt.title('{} \nConfusion Matrix - {} \nAccuracy: {:.2f}%'.format(self.config['training_name'], partition, acc * 100))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(test_plot_dir, f'confusion_matrix_{partition}.png'))
            plt.show()

        # save final results dict

        with open(os.path.join(self.config['experiment_dir'], 'results.yml'), 'w') as f:
            yaml.dump(results, f)
