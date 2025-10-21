from easydict import EasyDict as edict

action_list = {0:"up",
1:"right",
2:"down",
3:"left"
}

cfg = edict()



cfg.data = edict()
cfg.data.patch_size = 5
cfg.data.min_budget = 10
cfg.data.max_budget = 11
cfg.data.budget_step = 2
cfg.min_c = 4
cfg.max_c = 9
cfg.dataset = 'swisstopo-unseen' # 'xbd', 'xbd-post', 'mmgag', 'swisstopo', 'masa', 'swisstopo-unseen'
cfg.reward = 'in'
cfg.factor = 1

# set sample number for each dataset
if cfg.dataset == 'masa' or cfg.dataset == 'masa-budget':
    cfg.sample_number = 895
elif cfg.dataset == 'swisstopo':
    cfg.sample_number = 500
elif cfg.dataset == 'swisstopo-unseen':
    cfg.sample_number = 15*25

# set the number of configurations per image
if cfg.dataset == 'swisstopo-unseen':
    cfg.num_config_per_img = 1
else:
    cfg.num_config_per_img = 5


# set paths for each dataset
# masa
# note we used resize
if cfg.dataset == 'masa':
    if cfg.data.patch_size == 5:
        cfg.data.train_path = 'data/masa/sat_train_grid_5.npy'
        cfg.data.val_path = 'data/masa/sat_val_grid_5.npy'
        cfg.data.test_path = 'data/masa/sat_test_grid_5.npy'
    elif cfg.data.patch_size == 10:
        cfg.data.train_path = 'data/masa/sat_train_grid_10.npy'
        cfg.data.val_path = 'data/masa/sat_val_grid_10.npy'
        cfg.data.test_path = 'data/masa/sat_test_grid_10.npy'

#swissview
elif cfg.dataset == 'swissview':
    cfg.data.test_path = 'data/swissview/sat_swissview_test_3_grid_5.npy'

elif cfg.dataset == 'swissview-unseen':
    cfg.data.ground_embeds_path = 'data/swissview/grd_swissview_unseen_shifted_v2.npy'
    cfg.data.text_embeds_path = 'data/swissview/text_knowledge_swissview.npy'
    #cfg.data.text_embeds_path = 'data/swissview/text_content_swissview.npy'
    cfg.data.test_path = 'data/swissview/sat_swissview_unseen_shifted_grid_5.npy'


#===========pretrain============
cfg.pretrain = edict()
cfg.pretrain.ckpt_folder = "checkpoint"
cfg.pretrain.expt_folder = "pretrain_falcon"
cfg.pretrain.expt_name ="sat2cap_optimal_action_falcon.pt"
cfg.pretrain.log_name = "expt_logs.txt"
cfg.pretrain.min_seq_length = 6

cfg.pretrain.hparams = edict()
cfg.pretrain.hparams.accelerator='gpu'
cfg.pretrain.hparams.lr = 1e-5
cfg.pretrain.hparams.warmup = 5
cfg.pretrain.hparams.devices = 1
cfg.pretrain.hparams.epochs = 300
cfg.pretrain.hparams.weight_decay = 0.0001

# ==============train================
# checkpoint from the author
# falcon-pretrain
# gomaa-geo-falcon-pretrain

cfg.train = edict()
cfg.train.ckpt_folder = "checkpoint"
cfg.train.expt_folder = "test" # name for the experiment
cfg.train.load_from_checkpoint = False

cfg.train.llm_checkpoint = "checkpoint/geoexplorer/state_action.pt.ckpt"


#geoexplorer
#cfg.train.checkpoint_path = "checkpoint/geoexplorer/geoexplorer.pt"


cfg.train.expt_name ="ppo_falcon.pt"
cfg.train.expt_name_tmp ="ppo_falcon_"
cfg.train.log_name = "expt_logs.txt"
cfg.train.llm_model = "tiiuae/falcon-7b"
cfg.train.num_actions = 4
cfg.train.llm_hidden_dim = 1152


cfg.train.hparams = edict()
cfg.train.hparams.max_ep_len = cfg.data.min_budget
cfg.train.hparams.max_training_timesteps = int(1e8)
cfg.train.hparams.log_freq = cfg.train.hparams.max_ep_len * 2
cfg.train.hparams.save_model_freq = int(2e4)
cfg.train.hparams.update_timestep = cfg.train.hparams.max_ep_len * 64
cfg.train.hparams.K_epochs = 4
cfg.train.hparams.eps_clip = 0.2
cfg.train.hparams.gamma = 0.93
cfg.train.hparams.lr_actor = 0.0001
cfg.train.hparams.lr_critic = 0.0001
cfg.train.hparams.lr_llm = 0.0001
cfg.train.hparams.lr_gamma = 0.9999
cfg.train.hparams.random_seed = 42
