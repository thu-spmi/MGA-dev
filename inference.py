import warnings
import copy
import os
import time
import logging
import json
import numpy as np

import utils.utils as utils
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from mwzeval.metrics import Evaluator
from config import global_config as config # only tokenizer is used
from reader import MultiWozReader


warnings.filterwarnings("ignore")
tokenizer = GPT2Tokenizer.from_pretrained(config.model_path)
reader = MultiWozReader(tokenizer)
global_output = 2

def generate_batch(model, contexts, max_len, eos_id, beam=1):#a general way to generate, may be substitued by the generate function of huggingface, performance difference need to be evaluation
    # generate by batch
    # contexts: a list of ids
    # max_len: the max generated length
    # eos_id: the end id
    # return: a batch of ids with pre pad 
    batch_size=len(contexts)
    end_flag=np.zeros(batch_size)
    if beam>1:
        beam_box=[beam]*batch_size
        beam_result=[[] for _ in range(batch_size)]
        max_prob=[-float('inf')]*batch_size
    past_key_values=None
    inputs,attentions=reader.batch_align(contexts,left_len=max_len,return_attn=True)
    inputs=torch.tensor(inputs).to(model.device)
    attentions=torch.tensor(attentions).to(model.device)
    model.eval()
    with torch.no_grad():
        for i in range(max_len):
            if beam==1:
                position_ids = attentions.long().cumsum(-1) - 1
                position_ids.masked_fill_(attentions == 0, 1)
                if past_key_values is not None:
                    position_ids=position_ids[:, -1].unsqueeze(-1)
                if inputs.size(0)==0:
                    raise ValueError(contexts, inputs.cpu().list(), attentions)
                outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                        return_dict=True,use_cache=True,past_key_values=past_key_values)

                past_key_values=outputs.past_key_values

                preds=outputs.logits[:,-1,:].argmax(-1)#B
                if i==0:
                    gen_tensor=preds.unsqueeze(1)
                else:
                    gen_tensor=torch.cat([gen_tensor,preds.unsqueeze(1)],dim=1)
                attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                inputs=preds.unsqueeze(1)
                end_flag+=(preds.cpu().numpy()==eos_id).astype(float)
                if sum(end_flag==0)==0:
                    break
            else:
                if i==0:
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)
                    past_key_values=[outputs.past_key_values]*beam
                    log_prob=F.log_softmax(outputs.logits[:, -1, :], -1) # B, V
                    beam_prob, beam_idx=torch.topk(log_prob, beam, -1) # B, beam
                    gen_tensor=beam_idx.unsqueeze(-1)# B, beam, 1
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    position_ids=position_ids[:, -1].unsqueeze(-1)
                    pv_beam_prob=beam_prob #B, beam
                    pv_beam_idx=beam_idx#B, beam
                else:
                    for j in range(beam):
                        inputs=pv_beam_idx[:,j].unsqueeze(-1) # B, 1
                        outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values[j])
                        past_key_values[j]=outputs.past_key_values
                        log_prob=F.log_softmax(outputs.logits[:, -1, :], -1) # B, V
                        beam_prob, beam_idx=torch.topk(log_prob, beam, -1) # B, beam
                        if j==0:
                            prob_pool= beam_prob+pv_beam_prob[:, j].unsqueeze(-1).expand(-1, beam) # B, beam
                            id_pool=beam_idx
                        else:
                            prob_pool=torch.cat([prob_pool, beam_prob+pv_beam_prob[:, j].unsqueeze(-1).expand(-1, beam)],-1) # B, beam*beam
                            id_pool=torch.cat([id_pool, beam_idx], -1)# B, beam*beam
                    beam_prob, temp_id=torch.topk(prob_pool, beam, -1) #B, beam
                    beam_idx=torch.gather(id_pool, -1, temp_id)
                    temp_id=temp_id//beam
                    new_past_key_values=copy.deepcopy(past_key_values)
                    for b in range(batch_size):
                        gen_tensor[b, :, :]=gen_tensor[b, :, :].index_select(0, temp_id[b, :])
                        for t in range(beam):
                            for l in range(6):
                                new_past_key_values[t][l][:, b, :,:,:]=past_key_values[temp_id[b, t]][l][:, b, :, :, :]
                    past_key_values=new_past_key_values
                    gen_tensor=torch.cat([gen_tensor, beam_idx.unsqueeze(-1)],-1) #B, beam, T
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    position_ids=position_ids[:, -1].unsqueeze(-1)
                    pv_beam_prob=beam_prob #B, beam
                    pv_beam_idx=beam_idx
                for m in range(batch_size):
                    for n, gen in enumerate(gen_tensor.cpu().tolist()[m]):
                        if eos_id in gen:
                            beam_box[m]-=1
                            avg_prob=pv_beam_prob[m][n]/len(gen)
                            beam_result[m].append((gen, avg_prob))
                            pv_beam_prob[m][n]=-float('inf')
                # we do not break during beam search        
    if beam==1:
        return gen_tensor.cpu().tolist()
    else:
        for i, tup in enumerate(beam_result):
            beam_list=sorted(tup, key=lambda item:item[1], reverse=True)
            beam_result[i]=[item[0] for item in beam_list[:beam]]
        return beam_result      

def gen_hidden_state(model, turn_batch, pv_batch, turn_num, posterior=True, validate=False):#used only in semi-supervision.input batches are turn_batches
    model.eval()
    max_len_b=60
    max_len_a=20
    with torch.no_grad():
        # generate bspn
        contexts=reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_bspn', posterior=posterior)
        bspn_batch=generate_batch(model, contexts, max_len_b, reader.eos_b_id)
        bs_gen, db_gen=get_bspn(bspn_batch,return_db=True,data=turn_batch,turn_num=turn_num)
        # generate aspn
        contexts=reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_ar', 
            bspn_gen=bs_gen,db_gen=db_gen, posterior=posterior)
        aspn_batch=generate_batch(model, contexts, max_len_a, reader.eos_a_id)
        aspn_gen=get_aspn(aspn_batch)
        if validate:# generate hidden states for validation
            turn_batch['bspn_gen']=bs_gen
            turn_batch['db_gen']=db_gen
            turn_batch['aspn_gen']=aspn_gen
        else:# generate hidden states for training
            turn_batch['bspn']=bs_gen
            turn_batch['db']=db_gen
            turn_batch['aspn']=aspn_gen
        pv_batch=reader.get_pv_batch(pv_batch, turn_batch['user'], turn_batch['resp'], bs_gen)
    return turn_batch, pv_batch

def get_bspn(bs_tensor,return_db=False,data=None,turn_num=None):#used only in semi-supervision
    #return db, data and turn_num must be together
    #data represents one batch
    if not isinstance(bs_tensor,list):
        bs_batch=bs_tensor.cpu().tolist()
    else:
        bs_batch=bs_tensor
    bs_gen=[]
    db_gen=[]
    eos_b_id=reader.eos_b_id
    sos_b_id=reader.sos_b_id
    for i,bs in enumerate(bs_batch):
        if eos_b_id in bs:
            bs=[sos_b_id]+bs[:bs.index(eos_b_id)+1]
        else:
            bs[-1]=eos_b_id
            bs=[sos_b_id]+bs
        if bs.count(sos_b_id)>1:
            last=bs[::-1].index(sos_b_id)+1
            bs=bs[-last:]

        bs_gen.append(bs)
        if return_db:
            db_result=reader.bspan_to_DBpointer(reader.tokenizer.decode(bs), data['turn_domain'][i])
            db = reader.tokenizer.convert_tokens_to_ids(reader.tokenizer.tokenize('<sos_db> '+ db_result + ' <eos_db>'))
            db_gen.append(db)
    if return_db:
        return bs_gen,db_gen
    else:
        return bs_gen

def get_aspn(aspn_tensor):#used only in semi-supervision
    if not isinstance(aspn_tensor, list):
        aspn_batch=aspn_tensor.cpu().tolist()
    else:
        aspn_batch=aspn_tensor
    aspn_gen=[]
    eos_a_id=reader.eos_a_id
    sos_a_id=reader.sos_a_id
    for i ,aspn in enumerate(aspn_batch):
        if eos_a_id in aspn:
            aspn=[sos_a_id]+aspn[:aspn.index(eos_a_id)+1]
        else:
            aspn[-1]=eos_a_id
            aspn=[sos_a_id]+aspn
        if aspn.count(sos_a_id)>1:
            last=aspn[::-1].index(sos_a_id)+1
            aspn=aspn[-last:]
        aspn_gen.append(aspn)
    return aspn_gen

def get_resp(resp_tensor):
    if not isinstance(resp_tensor,list):
        resp_batch=resp_tensor.cpu().tolist()
    else:
        resp_batch=resp_tensor
    resp_gen=[]
    eos_r_id=reader.eos_r_id
    sos_r_id=reader.sos_a_id
    for i,resp in enumerate(resp_batch):
        if eos_r_id in resp:
            resp=[sos_r_id]+resp[:resp.index(eos_r_id)+1]
        else:
            resp[-1]=eos_r_id
            resp=[sos_r_id]+resp
        if resp.count(sos_r_id)>1:
            last=resp[::-1].index(sos_r_id)+1  #[::-1] means reverse the list
            resp=resp[-last:]
        resp_gen.append(resp)
    return resp_gen

def validate_fast(Modelclass,data='dev'):
    cfg = Modelclass.cfg
    model = Modelclass.model
    model.eval()
    reader = Modelclass.reader
    eval_data = reader.get_eval_data(data)
    if cfg.debugging:
        eval_data=eval_data[:32]
    origin_batch_size=cfg.batch_size
    cfg.batch_size=cfg.eval_batch_size
    batches = reader.get_batches('test',data=eval_data)
    result_path=os.path.join(cfg.eval_load_path,'result.json')
    
    if os.path.exists(result_path) and cfg.mode=='test':
        results=json.load(open(result_path, 'r'))
        e = Evaluator(True,True,False,True)  #args.bleu, args.success, args.richness, args.dst
        eval_results, _ , _ , _ = e.evaluate(results['dialogs'])
        bleu = eval_results['bleu']['mwz22']
        success = eval_results['success']['success']['total']
        match = eval_results['success']['inform']['total']
        joint_acc =  eval_results['dst']['joint_accuracy']#slot_f1
        #bleu, success, match = self.evaluator.validation_metric(results)
        score = 0.5 * (success + match) + bleu
        logging.info('[Std] validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))
        eval_results = {}
        eval_results['bleu'] = bleu
        eval_results['success'] = success
        eval_results['match'] = match
        eval_results['score'] = score
        eval_results['joint_acc']=joint_acc
        eval_results['result'] = 'validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    score: %.2f   joint: %.3f' % (match, success, bleu, score, joint_acc)
        return eval_results
    
    result_collection = {}
    st=time.time()
    for batch in batches:
        try:
            if batch==[]:
                continue
            batch=generate_batch_turn_level(Modelclass,batch)
            for dialog in batch:
                result_collection.update(reader.inverse_transpose_turn(dialog))
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logging.info("WARNING: ran out of memory during validation and batch will be divided by half, batch size:{}, turn num:{}"\
                    .format(len(batch),len(batch[0])))
                if hasattr(torch.cuda, 'empty_cache'):
                    with torch.cuda.device(Modelclass.device):
                        torch.cuda.empty_cache()
                #divide the batch in half if out of memory
                batches.insert(0,batch[:len(batch)//2])
                batches.insert(1,batch[len(batch)//2:])
            else:
                logging.info(str(exception))
                raise exception
    results, field = reader.wrap_result_lm(result_collection)
    logging.info('Inference time:{:.3f} min'.format((time.time()-st)/60))

    dials=utils.pack_dial(results)
    input_data={}
    for dial_id in dials:
        input_data[dial_id]=[]
        dial=dials[dial_id]
        length=len(dial)
        for l in range(length):
            for turn in dial:
                if int(turn['turn_num'])==l:
                    if turn['user']=='':
                        continue
                    entry={}
                    entry['response']=turn['resp_gen']
                    entry['state']=reader.bspan_to_constraint_dict(turn['bspn_gen'])
                    input_data[dial_id].append(entry)
    e = Evaluator(True,True,False,True)#args.bleu, args.success, args.richness, args.dst
    eval_results,match_list, success_list, bleu_list = e.evaluate(input_data)
    result = {'dialogs':input_data,'eval':eval_results,'match':match_list,'success':success_list,'bleu':bleu_list}
    json.dump(result, open(result_path, 'w'), indent=2)
    #eval_results = e.evaluate(input_data)
    bleu = eval_results['bleu']['mwz22']
    success = eval_results['success']['success']['total']
    match = eval_results['success']['inform']['total']
    joint_acc = joint_acc =  eval_results['dst']['joint_accuracy']#slot_f1
    #bleu, success, match = self.evaluator.validation_metric(results)
    score = 0.5 * (success + match) + bleu
    logging.info('[Std] validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))

    eval_results = {}
    eval_results['bleu'] = bleu
    eval_results['success'] = success
    eval_results['match'] = match
    eval_results['score'] = score
    eval_results['joint_acc']=joint_acc
    cfg.batch_size=origin_batch_size
    return eval_results

def generate_batch_turn_level(Modelclass, batch, posterior=False):
    batch=reader.transpose_batch(batch)
    model = Modelclass.model
    model.eval()
    max_len_b=75
    max_len_resp=80
    batch_size=len(batch[0]['dial_id'])
    contexts=[[] for i in range(batch_size)]
    bs_gen=[]
    db_gen=[]
    resp_gen=[]
    pv_batch=None
    with torch.no_grad():
        for turn_num, turn_batch in enumerate(batch):
            contexts=reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_bspn', posterior=posterior)
            if global_output>0 :
                logging.info(Modelclass.tokenizer.decode(contexts[0]))
                global_output-=1

            bspn_batch=generate_batch(model, contexts, max_len_b, reader.eos_b_id)# generate bspn
            bs_gen, db_gen=get_bspn(bspn_batch,return_db=True,data=turn_batch,turn_num=turn_num)
            
            contexts=reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_ar', 
                bspn_gen=bs_gen,db_gen=db_gen, posterior=posterior)
            resp_tensor=generate_batch(model, contexts, max_len_resp, reader.eos_r_id)# generate aspn and resp
            aspn_gen=get_aspn(resp_tensor)
            resp_gen=get_resp(resp_tensor)

            pv_batch=reader.get_pv_batch(pv_batch, turn_batch['user'], resp_gen, bs_gen, db_gen)
            turn_batch['bspn_gen']=bs_gen
            turn_batch['db_gen']=db_gen
            turn_batch['aspn_gen']=aspn_gen
            turn_batch['resp_gen']=resp_gen
    return reader.inverse_transpose_batch(batch)

def validate_pos(Modelclass,data='dev'):
    cfg = Modelclass.cfg
    model = Modelclass.model
    result_path=os.path.join(cfg.eval_load_path,'result.json')
    if os.path.exists(result_path) and 'test' in cfg.mode:
        results=json.load(open(result_path, 'r'))
        eval_result={}
        #eval_result['act_F1']=evaluator.aspn_eval(results)
        cnt=0
        recall_cnt=0
        correct_cnt=0
        recall_correct_cnt=0
        for turn in results:
            if turn['bspn_gen']!='':
                bspn_gen=turn['bspn_gen'].split(' ')
                bspn=turn['bspn'].split(' ')
                for word in bspn_gen:
                    cnt = cnt + 1
                    if word in bspn:
                        correct_cnt = correct_cnt + 1
                for word in bspn:
                    recall_cnt =  recall_cnt + 1
                    if word in bspn_gen:
                        recall_correct_cnt = recall_correct_cnt + 1
        recall = recall_correct_cnt/recall_cnt
        precision = correct_cnt/cnt
        eval_result['slot_F1'] = 2*recall*precision / (recall+precision)
        # TODO: slot accuracy
        #eval_result['joint_acc'], eval_result['db_acc'] = compute_jacc(results,return_db=True)
        logging.info('Joint acc:{:.3f}, Act_F1:{:.3f}, slot_acc:{:.3f}'.\
            format(eval_result['joint_acc'],eval_result['act_F1'],eval_result['slot_F1']))
        return eval_result
    reader._load_data()
    eval_data = reader.get_eval_data(data)
    cfg.batch_size=cfg.eval_batch_size
    batches=reader.get_batches('test',data=eval_data)
    results=[]
    st=time.time()
    result_collection={}
    for batch_idx, batch in enumerate(batches):
        pv_batch=None
        dial_batch=reader.transpose_batch(batch)
        for turn_num, turn_batch in enumerate(dial_batch):
            turn_batch, pv_batch=gen_hidden_state(model,turn_batch, pv_batch, turn_num, posterior=True, validate=True)
        
        dial_batch=reader.inverse_transpose_batch(dial_batch)
        for dialog in dial_batch:
            result_collection.update(reader.inverse_transpose_turn(dialog))
    
    results, field = reader.wrap_result_lm(result_collection)
    logging.info('Inference time:{:.3f} min'.format((time.time()-st)/60))
    json.dump(results, open(result_path, 'w'), indent=2)
    eval_result={}
    #eval_result['act_F1']=evaluator.aspn_eval(results)

    logging.info('Joint acc:{:.3f}, Act_F1:{:.3f}, DB_acc:{:.3f}'.\
        format(eval_result['joint_acc'],eval_result['act_F1'],eval_result['db_acc']))
    return eval_result

