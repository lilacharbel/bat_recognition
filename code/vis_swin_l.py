import torch
import warnings
torch.autograd.set_detect_anomaly(True)
warnings.simplefilter("ignore")
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import argparse
import timm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from models.FGVC_HERBS.builder import MODEL_GETTER
import yaml
from BatDataLoader import BatDataLoader

# from utils.config_utils import load_yaml
# from vis_utils import ImgLoader, get_cdict

global module_id_mapper
global features
global grads

def get_cdict():
    _jet_data = {
              # 'red':   ((0.00, 0, 0),
              #          (0.35, 0, 0),
              #          (0.66, 1, 1),
              #          (0.89, 1, 1),
              #          (1.00, 0.5, 0.5)),
              'red':   ((0.00, 0, 0),
                       (0.35, 0.5, 0.5),
                       (0.66, 1, 1),
                       (0.89, 1, 1),
                       (1.00, 0.8, 0.8)),
             'green': ((0.000, 0, 0),
                       (0.125, 0, 0),
                       (0.375, 1, 1),
                       (0.640, 1, 1),
                       (0.910, 0.3, 0.3),
                       (1.000, 0, 0)),
             # 'blue':  ((0.00, 0.5, 0.5),
             #           (0.11, 1, 1),
             #           (0.34, 1, 1),
             #           (0.65, 0, 0),
             #           (1.00, 0, 0))}
             'blue':  ((0.00, 0.30, 0.30),
                       (0.25, 0.8, 0.8),
                       (0.34, 0.8, 0.8),
                       (0.65, 0, 0),
                       (1.00, 0, 0))
             }
    return _jet_data

def forward_hook(module: nn.Module, inp_hs, out_hs):
    global features, module_id_mapper
    layer_id = len(features) + 1
    module_id_mapper[module] = layer_id
    features[layer_id] = {}
    features[layer_id]["in"] = inp_hs
    features[layer_id]["out"] = out_hs
    # print('forward_hook, layer_id:{}, hs_size:{}'.format(layer_id, out_hs.size()))

def backward_hook(module: nn.Module, inp_grad, out_grad):
    global grads, module_id_mapper
    layer_id = module_id_mapper[module]
    grads[layer_id] = {}
    grads[layer_id]["in"] = inp_grad
    grads[layer_id]["out"] = out_grad
    # print('backward_hook, layer_id:{}, hs_size:{}'.format(layer_id, out_grad[0].size()))


def build_model(pretrainewd_path: str,
                img_size: int, 
                fpn_size: int, 
                num_classes: int,
                num_selects: dict,
                use_fpn: bool = True, 
                use_selection: bool = True,
                use_combiner: bool = True, 
                comb_proj_size: int = None):
    from models.FGVC_HERBS.pim_module.pim_module_eval import PluginMoodel

    model = \
        PluginMoodel(img_size = img_size,
                     use_fpn = use_fpn,
                     fpn_size = fpn_size,
                     proj_type = "Linear",
                     upsample_type = "Conv",
                     use_selection = use_selection,
                     num_classes = num_classes,
                     num_selects = num_selects, 
                     use_combiner = use_combiner,
                     comb_proj_size = comb_proj_size)

    if pretrainewd_path != "":
        ckpt = torch.load(pretrainewd_path)
        model.load_state_dict(ckpt)
    
    model.eval()

    ### hook original layer1~4
    model.backbone.layers[0].register_forward_hook(forward_hook)
    model.backbone.layers[0].register_full_backward_hook(backward_hook)
    model.backbone.layers[1].register_forward_hook(forward_hook)
    model.backbone.layers[1].register_full_backward_hook(backward_hook)
    model.backbone.layers[2].register_forward_hook(forward_hook)
    model.backbone.layers[2].register_full_backward_hook(backward_hook)
    model.backbone.layers[3].register_forward_hook(forward_hook)
    model.backbone.layers[3].register_full_backward_hook(backward_hook)
    ### hook original FPN layer1~4
    model.fpn_down.Proj_layer1.register_forward_hook(forward_hook)
    model.fpn_down.Proj_layer1.register_full_backward_hook(backward_hook)
    model.fpn_down.Proj_layer2.register_forward_hook(forward_hook)
    model.fpn_down.Proj_layer2.register_full_backward_hook(backward_hook)
    model.fpn_down.Proj_layer3.register_forward_hook(forward_hook)
    model.fpn_down.Proj_layer3.register_full_backward_hook(backward_hook)
    model.fpn_down.Proj_layer4.register_forward_hook(forward_hook)
    model.fpn_down.Proj_layer4.register_full_backward_hook(backward_hook)
    ### hook original FPN_UP layer1~4
    model.fpn_up.Proj_layer1.register_forward_hook(forward_hook)
    model.fpn_up.Proj_layer1.register_full_backward_hook(backward_hook)
    model.fpn_up.Proj_layer2.register_forward_hook(forward_hook)
    model.fpn_up.Proj_layer2.register_full_backward_hook(backward_hook)
    model.fpn_up.Proj_layer3.register_forward_hook(forward_hook)
    model.fpn_up.Proj_layer3.register_full_backward_hook(backward_hook)
    model.fpn_up.Proj_layer4.register_forward_hook(forward_hook)
    model.fpn_up.Proj_layer4.register_full_backward_hook(backward_hook)

    return model

def cal_backward(args, out, sum_type: str = "softmax"):
    assert sum_type in ["none", "softmax"]

    target_layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'FPN1_layer1', 'FPN1_layer2', 'FPN1_layer3', 'FPN1_layer4', 'comb_outs']

    sum_out = None
    for name in target_layer_names:

        if name != "comb_outs":
            tmp_out = out[name].mean(1)
        else:
            tmp_out = out[name]
        
        if sum_type == "softmax":
            tmp_out = torch.softmax(tmp_out, dim=-1)

        if sum_out is None:
            sum_out = tmp_out
        else:
            sum_out = sum_out + tmp_out # note that use '+=' would cause inplace error

    with torch.no_grad():
        if args.use_label:
            print("use label as target class")
            pred_score = torch.softmax(sum_out, dim=-1)[0][args.label]
            backward_cls = args.label
        else:
            pred_score, pred_cls = torch.max(torch.softmax(sum_out, dim=-1), dim=-1)
            pred_score = pred_score[0]
            pred_cls = pred_cls[0]
            backward_cls = pred_cls

    print(sum_out.size())
    print("pred: {}, gt: {}, score:{}".format(backward_cls, args.label, pred_score))
    sum_out[0, backward_cls].backward()

@torch.no_grad()
def get_grad_cam_weights(grads):
    weights = {}
    for grad_name in grads:
        _grad = grads[grad_name]['out'][0][0]
        L, C = _grad.size()
        H = W = int(L ** 0.5)
        _grad = _grad.view(H, W, C).permute(2, 0, 1)
        C, H, W = _grad.size()
        weights[grad_name] = _grad.mean(1).mean(1)
        print(weights[grad_name].max())

    return weights

@torch.no_grad()
def plot_grad_cam(features, weights):
    act_maps = {}
    for name in features:
        hs = features[name]['out'][0]
        L, C = hs.size()
        H = W = int(L ** 0.5)
        hs = hs.view(H, W, C).permute(2, 0, 1)
        C, H, W = hs.size()
        w = weights[name]
        w = w.view(-1, 1, 1).repeat(1, H, W)
        weighted_hs = F.relu(w * hs)
        a_map = weighted_hs
        a_map = a_map.sum(0)
        # a_map /= abs(a_map).max()
        act_maps[name] = a_map
    return act_maps


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    global module_id_mapper, features, grads
    module_id_mapper, features, grads = {}, {}, {}

    """
    Please add 
    pretrained_path to yaml file.
    """
    # ===== 0. get setting =====
    parser = argparse.ArgumentParser("Visualize SwinT Large")
    parser.add_argument('--experiment', default=None)

    # parser.add_argument("-pr", "--pretrained_root", type=str, help="contain {pretrained_root}/best.pt, {pretrained_root}/config.yaml")
    # parser.add_argument("-img", "--image", type=str)
    # parser.add_argument("-sn", "--save_name", type=str)
    # parser.add_argument("-lb", "--label", type=int)
    parser.add_argument("-usl", "--use_label", default=False, type=bool)
    parser.add_argument("-sum_t", "--sum_features_type", default="none", type=str)
    args = parser.parse_args()


    # load config file
    model_dir = os.path.join('../experiments', args.experiment)

    # load config file
    with open(os.path.join(model_dir, 'config.yml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    gc_dir = os.path.join(model_dir, 'grad_cam_plots')
    os.makedirs(gc_dir, exist_ok=True)


    bat_loader = BatDataLoader(config)
    train_loader, val_loader, test_loader = bat_loader.create_loaders()

    for batch in test_loader:

        imgs, labels = batch

        # ===== 1. build model and restore checkpoint =====
        model = build_model(pretrainewd_path = model_dir + "/best_checkpoint.pt",
                            img_size = config['input_size'][0],
                            fpn_size = config['fpn_size'],
                            num_classes = config['num_classes'],
                            num_selects = config['num_selects'])


        # ===== 2. load image =====

        for i in range(imgs.shape[0]):

            # imgs = imgs.to(device)
            # labels = labels.to(device)
            img = imgs[i]
            args.label = labels[i]


            # ===== 3. forward and backward =====
            img = img.unsqueeze(0) # add batch size dimension
            out = model(img)

            cal_backward(args, out, sum_type=args.sum_features_type)

            # ===== 4. check result =====
            grad_weights = get_grad_cam_weights(grads)
            act_maps = plot_grad_cam(features, grad_weights)

            # ===== 5. show =====
            # cv2.imwrite("./vis_imgs/{}_ori.png".format(args.save_name), ori_img)
            sum_act = None
            resize = torchvision.transforms.Resize(tuple(config['input_size']))
            for name in act_maps:
                layer_name = "layer: {}".format(name)
                _act = act_maps[name]
                _act /= _act.max()
                r_act = resize(_act.unsqueeze(0))
                act_m = _act.numpy() * 255
                act_m = act_m.astype(np.uint8)
                act_m = cv2.resize(act_m, tuple(config['input_size']))
                # cv2.namedWindow(layer_name, 0)
                # cv2.imshow(layer_name, act_m)
                if sum_act is None:
                    sum_act = r_act
                else:
                    sum_act *= r_act

            sum_act /= sum_act.max()
            sum_act = torchvision.transforms.functional.adjust_gamma(sum_act, 1.0)
            sum_act = sum_act.numpy()[0]

            # sum_act *= 255
            # sum_act = sum_act.astype(np.uint8)

            plt.cla()
            cdict = get_cdict()
            cmap = matplotlib.colors.LinearSegmentedColormap("jet_revice", cdict)
            plt.imshow(img[0].permute([1, 2, 0]))
            plt.imshow(sum_act, alpha=0.5, cmap=cmap) # , alpha=0.5, cmap='jet'
            plt.axis('off')
            # plt.savefig("./{}.jpg".format(args.save_name),
            #     bbox_inches='tight', pad_inches=0.0, transparent=True)
            plt.savefig(os.path.join(gc_dir, f'grad_cam_{i}.png'))

            # plt.show()
            print('.')

        break

    # cv2.namedWindow("ori", 0)
    # cv2.imshow("ori", ori_img)
    # cv2.namedWindow("heat", 0)
    # cv2.imshow("heat", sum_act)
    # cv2.waitKey(0)