from test_BERT import *
import torch
import torch.nn as nn
import time
import pandas as pd
from torch.nn.functional import softmax


def load_checkpoint(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])


    return model
    
def normalizeStringInDF(s):
    s = s.str.normalize('NFC')
    s = s.str.replace(r"([.!?])", r" \1")
    s = s.str.replace(r"[^a-zA-Z0-9.!?]+", r" ")
    s = s.str.replace(r"<a.*</a>", 'url')
    return s

def init_model_and_stuff():
    with open('data/questions_and_answers_ntu.json','r') as f:
        data = f.readline()
    data = pd.read_json(data, orient='index').apply(lambda x: normalizeStringInDF(x))
    # data = pd.read_csv('data/ntu_faq.csv', encoding='utf-8').apply(lambda x: normalizeStringInDF(x))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='./.pytorch_pretrained_bert', do_lower_case=True)
    model = QAC_BERT(1,2,3, load_pretrained=False)
    model = load_checkpoint('experiments/test2/9_checkpoint.pth.tar', model)
    print('loaded')
    return data, tokenizer, model.cuda()
    
if __name__ == '__main__':
    MAX_SEQ_LEN = 200
    correct_top5,correct_top3,correct_top1,total = 0,0,0,0
    data, tokenizer, model = init_model_and_stuff()
    questions = data.loc[:,'question'].to_frame()
    answers = data.loc[:,'answer'].to_frame()
    model.eval()
    with torch.no_grad():
        while 1:
            input_str = input("\n\nEnter Input String: ")
        # for i, strin in questions.iterrows():
            # i = int(i[2:])-1
            # input_str = strin[0]
            # print(i, input_str)
            stime = time.time()
            input_example = InputExample(guid=0, text_a=input_str)
            tokens = ["[CLS]"] + tokenizer.tokenize(input_example.text_a) + ["[SEP]"]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(tokens)
            padding = [0] * (MAX_SEQ_LEN - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            assert len(input_ids) == MAX_SEQ_LEN
            assert len(input_mask) == MAX_SEQ_LEN
            assert len(segment_ids) == MAX_SEQ_LEN
            labels_ids = []
            input_feat = InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=labels_ids)
            logits = model(torch.tensor(input_ids).unsqueeze(0).cuda(), torch.tensor(segment_ids).unsqueeze(0).cuda(), torch.tensor(input_mask).unsqueeze(0).cuda(), labels=None)
            pred = softmax(logits)
            vals, topk = torch.topk(pred, 5)
            print(vals)
            # print(i)
            print(topk)
            topk = topk[0].cpu().numpy()
                
            print("\n\nTop 3 most probable answers: ")
            print('\n\n1.', answers.iloc[topk[0],0], 'for\n', questions.iloc[topk[0],0])
            print('\n\n2.', answers.iloc[topk[1],0], 'for\n', questions.iloc[topk[1],0])
            print('\n\n3.', answers.iloc[topk[2],0], 'for\n', questions.iloc[topk[2],0])
            print("time taken for one search: ", time.time()-stime, "seconds")
            
        #     if i in topk:
        #         correct_top5 +=1
        #     if i in topk[:3]:
        #         correct_top3 +=1
        #     if i in topk[:1]:
        #         correct_top1 +=1
        #     total += 1
        # 
        # correct_top5 /= total
        # correct_top3 /= total
        # correct_top1 /= total
        # print('top1:', correct_top1)
        # print('top3:', correct_top3)
        # print('top5:', correct_top5)
        # 
                
            
            
    
    