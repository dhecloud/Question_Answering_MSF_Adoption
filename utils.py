import datetime
import os
import numpy as np
import pandas as pd
import bcolz
import pickle 
import spacy
import json
from fastText import load_model
import random 
import string
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_default_args(args):
    if not args.name:   
        now = datetime.datetime.now()
        args.name = now.strftime("%Y-%m-%d-%H-%M")
    args.expr_dir = os.path.join(args.save_dir, args.name)

def random_word(N):
    return ''.join(
        random.choices(
            string.ascii_uppercase + string.ascii_lowercase + string.digits,
            k=N
        )
    )

def save_plt(array, name, args):
    colors = ['blue','red','green','pink','purple']
    plt.cla()
    plt.clf()
    plt.close()
    for i in range(len(array)):
        np.savetxt(os.path.join(args.expr_dir,name[i]+'.txt'), array[i], fmt='%f')
        plt.plot(array[i],color=colors[i], label=name[i])
        plt.xlabel('epoch')
        plt.legend()
    plt.savefig(os.path.join(args.expr_dir, name[i]+'.png'))
    plt.cla()
    plt.clf()
    plt.close()

    
def txt_to_csv(path='data/', files=['questions_tokens', 'answers_tokens']):
    data_list = []
    for file in files:
        data = pd.read_csv(path+file+'.txt', sep="\t",header = None)
        data_list.append(data)
        data.to_csv(path + file + '.csv', header=False, index=False)
        
    data = pd.concat(data_list, axis=1)
    data.to_csv('data/combined.csv', header=False, index=False)

def normalizeStringInDF(s):
    s = s.str.normalize('NFC')
    s = s.str.lower()
    s = s.str.replace(r"([.!?])", r" \1")
    s = s.str.replace(r"[^a-zA-Z0-9.!?]+", r" ")
    s = s.str.replace(r"<a.*</a>", 'url')
    return s

def parseAugmentedPermutations():
    stime = time.time()
    augments = {}
    files = ['output_success_'+ str(x) + '.txt' for x in range(0,95)]
    for file in files:
        with open('data/ntu_faq_permutations/'+file, 'r', encoding='utf-8') as f:
            print("processing", file, "...")
            lines = f.readlines()
            start_idxs = []
            for i, line in enumerate(lines):
                # lines[i] = normalizeString(line)
                # lines[i] = unicodeToAscii(line.lower().strip())
                # lines[i] = re.sub(r"([.!?])", r" \1", lines[i])
                # lines[i] = re.sub(r"[^a-zA-Z.!?]+", r" ", lines[i])
                while lines[i][0] == ' ' or lines[i][0] == "'":
                    if len(lines[i]) > 2:
                        lines[i] = lines[i][1:]
                if 'Permutations of ' in lines[i]:
                    start_idxs.append(i)
            start_idxs.append(i+1)
            for i, idx in enumerate(start_idxs):
                if idx != start_idxs[-1]:
                    key = lines[idx][17:-3]
                    while key[0] == ' ' or key[0] == "'":
                        if len(key) > 2:
                            key = key[1:]
                    if "==========" not in lines[start_idxs[i+1]-1]:
                        continue
                    assert("==================" in lines[start_idxs[i+1]-1] )
                    augments[key] = lines[idx+2:start_idxs[i+1]-1]
    etime = time.time() - stime 
    print("Took", etime, "to parse all augmented data!")
    # print(augments)
    return augments

def augments_into_csv(augments, data, sample=400, num_permutations=2):
    cls_list = []
    qns_list = []
    cls_to_qns = {}
    num_keys = len(list(augments.keys()))
    found = 0
    for key in augments.keys():
        size = len(augments[key])
        idx = data[data.question.str.contains(key,regex=False)==True].index
        if len(idx) == 1:
            idx = idx[0]
            cls_to_qns[idx] = augments[key]
            # if len(augments[key]) >= sample:
            #     cls_list += [idx for x in range(sample)]
            #     rndm_qns = random.sample(augments[key], sample)
            #     qns_list += rndm_qns
            # else:
            #     cls_list += [idx for x in range(len(augments[key]))]
            #     qns_list += augments[key]
            found += 1
    print(found, "keys found out of ", num_keys)
    for i in range(data.shape[0]):
        for j in range(sample):
            new_cls = [i]
            if i in cls_to_qns:
                q1 = random.choice(cls_to_qns[i]).strip()
            else:
                q1 = data.loc[i, 'question']
            new_qn = q1
            for k in range(num_permutations-1):
                rndm_cls = random.choice(list(cls_to_qns.keys()))
                new_qn = new_qn + ' ' + random.choice(cls_to_qns[rndm_cls]).strip()
                new_cls.append(rndm_cls)
            print(new_cls,new_qn)
            cls_list.append(new_cls)
            qns_list.append(new_qn)
    new = pd.DataFrame({'class':cls_list, 'question':qns_list})
    return new
if __name__ == '__main__':
    # wm = create_embeddings(["i","need","to","poop" , "asdfsdf"])
    # print(wm.shape)
    txt_to_csv()