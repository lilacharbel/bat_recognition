import os
import yaml
from argparse import ArgumentParser
from Trainer import Trainer

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, default='../configs/FGVC-HERBS.yaml', help='yaml file to initialize params')
    parser.add_argument('--partition', type=str, default='train')
    args = parser.parse_args()

    with open(os.path.join(args.config_file)) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.partition == 'train':
        all_experiments = os.listdir('../experiments')
        all_experiments = [exp for exp in all_experiments if exp.startswith(config['training_name'])]

        exp_num = 1 if len(all_experiments) == 0 else max([int(exp.split('_')[-1]) for exp in all_experiments]) + 1

        config['experiment_dir'] = os.path.join('../experiments', f'{config["training_name"]}_{exp_num}')
        config['training_name'] = f'{config["training_name"]}_{exp_num}'

        os.makedirs(config['experiment_dir'], exist_ok=True)

        # save config file
        with open(os.path.join(config['experiment_dir'], 'config.yml'), 'w') as f:
            yaml.dump(config, f)

        print('---------------')
        print(config['training_name'])
        print('---------------')

        # train
        trainer = Trainer(config)
        trainer.train()

    if args.partition == 'test':
        print('---------------')
        print(model_name)
        print('---------------')

        model_dir = os.path.join(folder_dir, 'experiments', model_name)

        # load config file
        with open(os.path.join(folder_dir, model_dir, 'config.yml')) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        trainer = FashionMNISTTrainer(config)
        trainer.test()