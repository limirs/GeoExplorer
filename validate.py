import os

import torch
from models.ppo import PPO
from data_utils import Sequence

from utils import generate_config, generate_config_unseen, seed_everything
from config import cfg

device = torch.device('cuda:0')


GOAL_PATCH_LIST = list(range(25)) * 16

if __name__=='__main__':

    if not os.path.exists(os.path.join(cfg.train.ckpt_folder, cfg.train.expt_folder)):
        os.makedirs(os.path.join(cfg.train.ckpt_folder, cfg.train.expt_folder))

    import json

    with open(os.path.join(cfg.train.ckpt_folder, cfg.train.expt_folder, "config.json"), "w") as f:
        json.dump(cfg, f)

    seed_everything(cfg.train.hparams.random_seed)

    # initialize a PPO agent
    ppo_agent = PPO(cfg.train.hparams.lr_actor,
    cfg.train.hparams.lr_critic,
    cfg.train.hparams.lr_llm,
    cfg.train.hparams.gamma,
    cfg.train.hparams.K_epochs,
    cfg.train.hparams.eps_clip,
    cfg.train.hparams.lr_gamma).cuda()

    ppo_agent.load_state_dict(torch.load(cfg.train.checkpoint_path))
    
    ppo_agent.eval()

    valid_path = cfg.data.test_path

    if cfg.dataset == 'swisstopo-unseen':
        ground_path = cfg.data.ground_embeds_path
        text_path = cfg.data.text_embeds_path
    
    import time

    start = time.time()

    print("chcekpoint", cfg.train.checkpoint_path)
    print("data", cfg.data.test_path)
    print("dataset", cfg.dataset)

    for d in range(cfg.min_c, cfg.max_c):
        seed_everything(cfg.train.hparams.random_seed)
        

        if cfg.dataset == 'swisstopo-unseen':
            config = generate_config_unseen(cfg.data.test_path, GOAL_PATCH_LIST, patch_size=cfg.data.patch_size, dist=d, n_config_per_img=cfg.num_config_per_img)
            # print("===================aerial view====================")
            # cur_val_success, res = ppo_agent.validate_unseen(config, valid_path, n_config_per_img=cfg.num_config_per_img)
            # print(f"dist={d}")
            # print("===================ground view====================")
            # cur_val_success = ppo_agent.validate_ground_unseen(config, valid_path, ground_path, n_config_per_img=cfg.num_config_per_img)
            # print(f"dist={d}")
            cur_val_success = ppo_agent.validate_text_unseen(config, valid_path, text_path, n_config_per_img=cfg.num_config_per_img)
            print(f"dist={d}")
        
        else:    
            config = generate_config(cfg.data.test_path, patch_size=cfg.data.patch_size, dist=d, n_config_per_img=cfg.num_config_per_img)
            cur_val_success, res = ppo_agent.validate(config, valid_path, n_config_per_img=cfg.num_config_per_img)
            print(f"dist={d}", f"success_ratio: {cur_val_success/cfg.sample_number}")
        