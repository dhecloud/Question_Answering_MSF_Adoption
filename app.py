from flask import Flask, request, jsonify
from test_BERT import *
import torch
import torch.nn as nn
import time
import pandas as pd
import json
app = Flask(__name__)

def load_checkpoint(path, model):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    return model
    
def normalizeStringInDF(s):
    s = s.str.normalize('NFC')
    s = s.str.replace(r"([.!?])", r" \1")
    s = s.str.replace(r"[^a-zA-Z0-9.!?]+", r" ")
    s = s.str.replace(r"<a.*</a>", 'url')
    return s


def init_model_and_stuff():
    answers = pd.read_csv('data/answers.txt', sep='\n', header=None, encoding='utf-8').apply(lambda x: normalizeStringInDF(x))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='./.pytorch_pretrained_bert', do_lower_case=True)
    model = QAC_BERT(1,2,3, load_pretrained=False)
    model = load_checkpoint('0_checkpoint.pth.tar', model)
    print('loaded')
    return answers, tokenizer, model

project_name = 'baby_bonus_faq'
answers, tokenizer, model = init_model_and_stuff()
model.eval()
MAX_SEQ_LEN = 512
ACCEPT_THRESHOLD = 1/294
# @app.route('/')
# def helloworld():
#     return "Use post methods to localhost:5000/predict to ask questions"

    
@app.route("/" + project_name + "-service", methods=['POST'])
def predict():
    if request.method == 'POST':
        tmp =request.form
        q = tmp['input']
        input_example = InputExample(guid=0, text_a=q)
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
        with torch.no_grad():
            logits = model(torch.tensor(input_ids).unsqueeze(0), torch.tensor(segment_ids).unsqueeze(0), torch.tensor(input_mask).unsqueeze(0), labels=None)
        pred = logits.sigmoid()
        _, topk = torch.topk(pred, 5)
        topk = topk[0].cpu().numpy()
    return jsonify({'text':answers.iloc[topk[0],0]})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')