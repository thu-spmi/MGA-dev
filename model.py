import os
import warnings
import shutil
import random
import time
import logging
import json

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel # BertTokenizer for Chinese gpt2
from torch.optim import Adam
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from reader import MultiWozReader
from utils.utils import add_torch_input
import inference

warnings.filterwarnings("ignore")

class Model(object):
    
    def __init__(self,cfg):
        self.cfg = cfg
        self.device = cfg.cuda_device
        if not cfg.multi_card:
            if isinstance(self.device,list):
                self.device = self.device[0]

        tokenizer_path=cfg.model_path
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)#related with model
        
        self.reader = MultiWozReader(self.tokenizer)
        self.model=GPT2LMHeadModel.from_pretrained(cfg.model_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        if cfg.gradient_checkpoint:
            self.model.config.gradient_checkpointing=True
            
        self.model.to(self.device)
        logging.info("Model loaded from {}".format(cfg.model_path))

        if cfg.save_log :
            log_path='./log/log_{}_lr{}'.format(cfg.exp_no,cfg.lr) if cfg.dataset==1 else './log/log_{}'.format(cfg.exp_no)
            if os.path.exists(log_path):
                shutil.rmtree(log_path)
                os.mkdir(log_path)
            else:
                os.mkdir(log_path)
            self.tb_writer = SummaryWriter(log_dir=log_path)
        else:
            self.tb_writer = None

        if 'test' not in cfg.mode:
            json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
        self.global_output=4
    
    def train(self): 
        cfg = self.cfg       
        num_dials = len(self.reader.train)
        all_batches = self.reader.get_batches('train') 
        train_data=self.reader.train
        random.shuffle(train_data)

        set_stats = self.reader.set_stats['train']
        num_turns=set_stats['num_turns']
        optimizer, scheduler = self.get_sep_optimizers(num_turns,self.model)

        # log info
        logging.info("***** Running turn-level training *****")
        logging.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
                     set_stats['num_training_steps_per_epoch'])
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        logging.info("  Num Dialogs = %d", set_stats['num_dials'])
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info("  Gradient Accumulation steps = %d",
                     cfg.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d",
                     set_stats['num_training_steps_per_epoch']*cfg.epoch_num // cfg.gradient_accumulation_steps)

        log_inputs = 4
        global_step = 0

        min_loss = 1000
        min_eval_loss=1000
        max_score=0
        early_stop_count=cfg.early_stop_count
        epoch_th=0.1*cfg.epoch_num if 'distilgpt2' in cfg.model_path else -1#early epochs skipped evaluation step to save time 
        warmup_epochs=cfg.warmup_steps*cfg.gradient_accumulation_steps*cfg.batch_size//num_dials \
            if cfg.warmup_steps>=0 else int(cfg.epoch_num*cfg.warmup_ratio)
        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss = 0.0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()
            random.shuffle(all_batches)

            for batch_idx, batch0 in enumerate(all_batches):
                dial_batch=self.reader.transpose_batch(batch0)
                pv_batch = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    first_turn = (turn_num == 0)
                    inputs, labels = self.reader.convert_batch_turn(turn_batch, pv_batch, first_turn, posterior=cfg.posterior_train)
                    pv_batch = self.reader.get_pv_batch(pv_batch, turn_batch['user'],
                        turn_batch['resp'], turn_batch['bspn'],turn_batch['dspn'])
                    try:  # avoid OOM
                        self.model.train()
                        if log_inputs > 0:  # log inputs for the very first two turns
                            logging.info('Input examples:\n{}'.format(self.tokenizer.decode(inputs['contexts'][0])))
                            log_inputs-=1
                        inputs = add_torch_input(inputs,self.device)
                        outputs = self.model(inputs['contexts_tensor'])
                        if cfg.only_target_loss:
                            labels = add_torch_input(labels,self.device)    
                            loss = self.calculate_loss_and_accuracy(outputs, labels=labels['contexts_tensor'])
                        else:
                            loss = self.calculate_loss_and_accuracy(outputs, labels=inputs['contexts_tensor'])
                        loss.backward()
                        tr_loss += loss.item()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                        epoch_step += 1

                        # step, wrt gradient_accumulation_steps, clip grad norm
                        if epoch_step % cfg.gradient_accumulation_steps == 0 or(
                            batch_idx==len(all_batches)-1 and turn_num==len(dial_batch)-1):
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            # global_step: actual step the optimizer took
                            global_step += 1
                            
                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            max_length = max(inputs['lengths'])
                            oom_time += 1
                            logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                                oom_time, cfg.batch_size, max_length))
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            logging.info(str(exception))
                            raise exception
            
            logging.info('Epoch:{}, Train epoch time: {:.2f} min, epoch loss: {:.4f}'.format(
                epoch, (time.time()-btm)/60, tr_loss))
            if cfg.save_type=='min_loss':
                eval_loss=self.eval(model=self.model)
                logging.info('model evaluation loss:{}'.format(eval_loss))
                if self.tb_writer:
                    self.tb_writer.add_scalar('loss_eval',eval_loss,epoch)
                if eval_loss<min_eval_loss:
                    min_eval_loss=eval_loss
                    self.save_model(path='best_loss_model',model=self.model)
                    early_stop_count=cfg.early_stop_count
                else:
                    if epoch>=warmup_epochs:#early stop after warm up
                        early_stop_count-=1
                        logging.info('early stop count:%d'%early_stop_count)
                        if early_stop_count==0 and cfg.early_stop:
                            logging.info('early stopped')
                            break
            elif cfg.save_type=='max_score' :
                if cfg.posterior_train and epoch>epoch_th:
                    eval_result=inference.validate_pos(data='dev')
                    self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                    self.tb_writer.add_scalar('act_F1',eval_result['act_F1'],epoch)
                    self.tb_writer.add_scalar('db_acc',eval_result['db_acc'],epoch)
                    score=eval_result['joint_acc']
                else:
                    if epoch>epoch_th:
                        eval_result=inference.validate_fast(self,data='dev')
                        self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                        self.tb_writer.add_scalar('match',eval_result['match'],epoch)
                        self.tb_writer.add_scalar('success',eval_result['success'],epoch)
                        self.tb_writer.add_scalar('bleu',eval_result['bleu'],epoch)
                        self.tb_writer.add_scalar('combined_score',eval_result['score'],epoch)
                        score=eval_result['score']
                    else:
                         score = 0
                if score>=max_score:
                    early_stop_count=cfg.early_stop_count
                    max_score=score
                    self.save_model(model=self.model)
                else:
                    if epoch>=warmup_epochs:
                        early_stop_count-=1
                        logging.info('early stop count:%d'%early_stop_count)
                        if early_stop_count==0 and cfg.early_stop:
                            logging.info('early stopped')
                            break
    

    def get_sep_optimizers(self, num_dials, model, num_batches=None):
        cfg = self.cfg
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        if not num_batches:
            num_training_steps = num_dials*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.batch_size)
        else:
            num_training_steps = num_batches*cfg.epoch_num
        num_warmup_steps = cfg.warmup_steps if cfg.warmup_steps >= 0 else int(num_training_steps*cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,\
            num_training_steps=num_training_steps) if cfg.use_scheduler else None
        logging.info('Training steps:{}, warmup steps:{}, steps per epoch:{}'.format(num_training_steps, 
            num_warmup_steps, num_batches))
        return optimizer, scheduler

    #use celoss in calculating, can be substituted by auto function in gpt2
    def calculate_loss_and_accuracy(self, outputs, labels):
        lm_logits = outputs[0]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        pad_id = self.tokenizer.encode('<pad>')[0]
        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        not_ignore = shift_labels.ne(pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
        return loss
  
    def save_model(self, path=None, model=None):
        if not path:
            save_path = os.path.join(self.cfg.exp_path, 'best_model')
        else:
            save_path = os.path.join(self.cfg.exp_path, path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info('Saving model checkpoint to %s', save_path)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def eval(self,data='dev',model=None):
        model.eval()
        all_batches = self.reader.get_batches(data)
        total_batch=len(all_batches)
        total_loss=0
        with torch.no_grad():
            data_iterator = self.reader.get_data_iterator(all_batches)
            for batch_idx, dial_batch in enumerate(data_iterator):
                pv_batch = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    first_turn = (turn_num == 0)
                    inputs, labels = self.reader.convert_batch_turn(turn_batch, pv_batch, first_turn, posterior=self.cfg.posterior_train)
                    pv_batch = self.reader.get_pv_batch(pv_batch, turn_batch['user'],
                        turn_batch['resp'], turn_batch['bspn'], turn_batch['dspn'])
                    inputs = add_torch_input(inputs,self.device)#B,T
                    labels = add_torch_input(labels,self.device)#B,T
                    outputs = model(inputs['contexts_tensor'])
                    loss=self.calculate_loss_and_accuracy(outputs,labels['contexts_tensor'])
                    total_loss+=loss.item()
        return total_loss/total_batch

