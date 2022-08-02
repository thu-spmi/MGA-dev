import logging, time, os

class _Config:
    def __init__(self):
        self.init()

    def init(self):
        self.data_path = './data/multi-woz-2.1-processed'# for multi-woz
        self.dbs = {
            'attraction': 'db/attraction_db_processed.json',
            'hospital': 'db/hospital_db_processed.json',
            'hotel': 'db/hotel_db_processed.json',
            'police': 'db/police_db_processed.json',
            'restaurant': 'db/restaurant_db_processed.json',
            'taxi': 'db/taxi_db_processed.json',
            'train': 'db/train_db_processed.json',
        }
        self.domain_file_path = 'data/multi-woz-2.1-processed/domain_files.json'
        self.slot_value_set_path = 'db/value_set_processed.json'
        self.exp_path = 'to be generated'
        self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        
        #key training settings
        self.multi_card = False
        self.save_type='max_score'#'min_loss'/'max_reward'
        self.mixed_train=False
        self.dataset=1 # 0 for multiwoz2.0, 1 for multiwoz2.1
        self.example_log=True
        
        #semi-supervise settings
        self.spv_proportion=100
        self.posterior_train=False
        self.gen_db=False #critical setting. Only True when we use posterior model to generate belief state.
        #VLtrain:
        self.VL_with_kl=True 
        self.PrioriModel_path='to be generated'
        self.PosteriorModel_path='to be generated'
        #STtrain:
        self.fix_ST=True # whether add straight through trick
        self.ST_resp_only=True #whether calculate cross-entropy on response only
        #evaluation:
        self.fast_validate=True
        self.eval_batch_size=32
        self.model_path = 'distilgpt2'
        #self.model_path = 'experiments_21/base/best_loss_model'#use pretraining
        self.val_set='test'
        self.col_samples=True #collect wrong predictions samples
        self.test_data_path=''

        #additional data setting
        self.len_limit=True
        self.post_loss_weight=0.5
        self.kl_loss_weight=0.5
        self.debugging=False        
        self.divided_path='to be generated'
        self.gradient_checkpoint=False

        # experiment settings
        self.exp_domains = ['all'] # hotel,train, attraction, restaurant, taxi
        self.log_path = ''
        self.mode = 'train'
        self.cuda = True
        self.cuda_device = [0]
        self.pos_device = [0]
        self.exp_no = ''
        self.seed = 11
        self.save_log = True # tensorboard 

        # training settings
        self.lr = 1e-4
        self.warmup_steps = -1 
        self.warmup_ratio= 0.2
        self.weight_decay = 0.0 
        self.gradient_accumulation_steps = 4
        self.batch_size = 8
        self.lr_decay = 0.5
        self.use_scheduler=True
        self.epoch_num = 40
        self.early_stop=False
        self.early_stop_count = 5
        self.weight_decay_count = 4
        self.only_target_loss = True #only used in training
        # evaluation settings
        self.eval_load_path = 'to be generated'
        self.model_output = 'to be generated'

        # sequence settings
        self.input_history = False
        self.only_dst = False
        # settings for further study
        
        self.rl_train = False
        self.score=True
        self.train_us=False
        self.use_previous_context = True
        self.save_first_epoch=True 
        self.sample_type='top1'#'topk'
        self.topk_num=10#only when sample_type=topk  
        self.use_pretraining = False    

    def _init_logging_handler(self, mode):
        stderr_handler = logging.StreamHandler()
        if not os.path.exists('./log'):
            os.mkdir('./log')
        if self.save_log and ('test'  not in mode):
            file_handler = logging.FileHandler('./log/log_{}_{}_sd{}.txt'.format(mode, self.exp_no, self.seed))
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        elif 'test' in mode and os.path.exists(self.eval_load_path):
            eval_log_path = os.path.join(self.eval_load_path, 'eval_log.json')
            file_handler = logging.FileHandler(eval_log_path)
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        else:
            logging.basicConfig(handlers=[stderr_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()

