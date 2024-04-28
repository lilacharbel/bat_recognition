import os
import argparse

import torch
import matplotlib.pyplot as plt
import yaml

from BatDataLoader import BatDataLoader


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import timm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_grad_cam_visualization(input_tensor, model):

    target_layers = [model.layer4]

    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor, targets=None, aug_smooth=True)

    grayscale_cam = grayscale_cam.transpose(1, 2, 0)
    norm_inputs = input_tensor[0].numpy().transpose(1, 2, 0)
    norm_inputs = (norm_inputs - norm_inputs.min()) / (norm_inputs.max() - norm_inputs.min())
    visualization = show_cam_on_image(norm_inputs, grayscale_cam, use_rgb=True)

    return visualization


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize SwinT Large")
    parser.add_argument('--experiment', default=None)

    # parser.add_argument("-pr", "--pretrained_root", type=str, help="contain {pretrained_root}/best.pt, {pretrained_root}/config.yaml")
    # parser.add_argument("-img", "--image", type=str)
    # parser.add_argument("-sn", "--save_name", type=str)
    # parser.add_argument("-lb", "--label", type=int)
    parser.add_argument("-usl", "--use_label", default=False, type=bool)
    parser.add_argument("-sum_t", "--sum_features_type", default="softmax", type=str)
    args = parser.parse_args()

    # load config file
    model_dir = os.path.join('../experiments', args.experiment)

    # load config file
    with open(os.path.join(model_dir, 'config.yml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    gc_dir = os.path.join(model_dir, 'grad_cam_plots')
    os.makedirs(gc_dir, exist_ok=True)

    # load model
    model = timm.create_model(model_name=config['model_name'], pretrained=config['pretrained'], num_classes=config['num_classes'])
    model.load_state_dict(torch.load(os.path.join(config['experiment_dir'], f'best_checkpoint.pt')))

    # det dataloaders
    bat_loader = BatDataLoader(config)
    train_loader, val_loader, test_loader = bat_loader.create_loaders()

    model.eval()

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        for i in range(inputs.shape[0]):
            input_tensor = inputs[i:i+1, ...]
            label = targets[i:i+1].item()

            visualization = get_grad_cam_visualization(input_tensor, model)
            grad_cam_figure = plt.figure()
            plt.imshow(visualization)
            plt.title(label)
            grad_cam_figure.savefig(os.path.join(gc_dir, f'grad_cam_{i}.png'))

        break