import os
import yaml
from argparse import ArgumentParser
import Trainers as Trainers
import importlib

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, default='../configs/swint.yaml', help='yaml file to initialize params')
    parser.add_argument('--partition', type=str, default='train')
    parser.add_argument('--experiment', default=None)
    args = parser.parse_args()

    with open(os.path.join(args.config_file)) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.partition == 'train':
        os.makedirs('../experiments', exist_ok=True)

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
        trainer = getattr(importlib.import_module(f"Trainers.{config['trainer']}"), config['trainer'])(config)
        trainer.train()

    if args.partition == 'test':
        print('---------------')
        print(args.experiment)
        print('---------------')

        model_dir = os.path.join('../experiments', args.experiment)

        # load config file
        with open(os.path.join(model_dir, 'config.yml')) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        trainer = getattr(importlib.import_module(f"Trainers.{config['trainer']}"), config['trainer'])(config)

        trainer.test()