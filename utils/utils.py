
import os
import json
import logging

import torch
import pickle
import numpy as np

import utils.MWZontology as ontology
from config import global_config as cfg

def add_torch_input(inputs, device):
        # to tensor and to device
        if 'contexts_np' not in inputs:
            inputs['contexts_np'],_=padSeqs_gpt(inputs['contexts'], cfg.pad_id)
        contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
        contexts_tensor = contexts_tensor.to(device)
        inputs['contexts_tensor'] = contexts_tensor
        return inputs

def padSeqs_gpt(sequences, pad_id, maxlen=None):
    lengths = []
    for x in sequences:
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_mexlen = np.max(lengths)
    if seq_mexlen > 1024: # gpt2.n_ctx / maxlen = 1024
        maxlen = 1024
    else:
        maxlen = seq_mexlen
    
    x = (np.ones((num_samples, maxlen)) * pad_id)
    for idx, s in enumerate(sequences):
        if not len(s):
            print('empty list was found in padSeqs')
        trunc = s[-maxlen:]
        trunc = np.asarray(trunc)
        x[idx, :len(trunc)] = trunc
            
    return x, lengths

def parse_arg_cfg(config,args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(config, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k == 'cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(config, k, v)
        try:
            if len(cfg.device)>1:
                cfg.device = cfg.device[0] if not cfg.multi_card else cfg.device
            if len(cfg.pos_device)>1:
                cfg.pos_device = cfg.pos_device[0] if not cfg.multi_card else cfg.pos_device
        except:
            pass
    return config

def dict2str(dict):
        s = ''
        for k,v in dict.items():
            s += '{} : {}\n'.format(k,v)
        return s

def pack_dial(data):
    dials = {}
    for turn in data:
        dial_id = turn['dial_id']
        if dial_id not in dials:
            dials[dial_id] = []
        dials[dial_id].append(turn)
    return dials

def aspan_to_act_list( aspan):
    aspan = aspan.split() if isinstance(aspan, str) else aspan
    acts = []
    domain = None
    conslen = len(aspan)
    for idx, cons in enumerate(aspan):
        #cons =vocab.decode(cons) if type(cons) is not str else cons
        if cons == '<eos_a>':
            break
        if '[' in cons and cons[1:-1] in ontology.dialog_acts:
            domain = cons[1:-1]

        elif '[' in cons and cons[1:-1] in ontology.dialog_act_params:
            if domain is None:
                continue
            vidx = idx+1
            if vidx == conslen:
                acts.append(domain+'-'+cons[1:-1]+'-none')
                break
            vt = aspan[vidx]
            #vt = self.vocab.decode(vt) if type(vt) is not str else vt
            no_param_act = True
            while vidx < conslen and vt != '<eos_a>' and '[' not in vt:
                no_param_act = False
                acts.append(domain+'-'+cons[1:-1]+'-'+vt)
                vidx += 1
                if vidx == conslen:
                    break
                vt = aspan[vidx]
                #vt = self.vocab.decode(vt) if type(vt) is not str else vt
            if no_param_act:
                acts.append(domain+'-'+cons[1:-1]+'-none')
    return acts

# io functions
def save_json(obj, save_path, indent=2):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(load_path, lower=True):
    with open(load_path, "r", encoding="utf-8") as f:
        obj = f.read()

        if lower:
            obj = obj.lower()

        return json.loads(obj)


def save_pickle(obj, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(load_path):
    with open(load_path, "rb") as f:
        return pickle.load(f)


def save_text(obj, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for o in obj:
            f.write(o + "\n")


def load_text(load_path, lower=True):
    with open(load_path, "r", encoding="utf-8") as f:
        text = f.read()
        if lower:
            text = text.lower()
        return text.splitlines()


