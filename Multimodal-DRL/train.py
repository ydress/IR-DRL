import argparse
from Config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Train a ur5 robot')
    
    parser.add_argument('env', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    
def main():
    args = parse_args()

    cfg = Config(config_path='config.test_config_file')
    
    #  # work_dir is determined in this priority: CLI > segment in file > filename
    # if args.work_dir is not None:
    # # update configs according to CLI args if args.work_dir is not None
    #     cfg.work_dir = args.work_dir
    # elif cfg.get('work_dir', None) is None:
    #     # use config filename as default work_dir if cfg.work_dir is None
    #     cfg.work_dir = osp.join('./work_dirs',
    #                             osp.splitext(osp.basename(args.config))[0])
        
    # if args.resume_from is not None:
    #     cfg.resume_from = args.resume_from
     
    model = cfg.get_model()
    
    print(model)
    
    model.learn(
        total_timesteps=cfg.train_cfg['total_timesteps'],
        n_eval_episodes=cfg.train_cfg['n_eval_episodes'],
        #callback=callback
    )   
    
    
if __name__ == '__main__':
    main()