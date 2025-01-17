import flask
from flask import Flask
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, onnx
import json
import torch

'''
https://huggingface.co/Helsinki-NLP/opus-mt-en-zh
'''
# torch.set_num_threads(4)
device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained("shanxi_model")
model = AutoModelForSeq2SeqLM.from_pretrained("shanxi_model").to(device)
with torch.inference_mode():
    model.eval()

app = Flask(__name__)


@app.route('/translation', methods=['POST', 'GET'])
def transaltion():
    input_text = flask.request.args.get('text')
    # input_text = [input_text]
    with torch.no_grad():
        translated = model.generate(**tokenizer(str(input_text), return_tensors="pt", padding=True).to(device))
        res = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return json.dumps({'translation': res}, ensure_ascii=False)


if __name__ == '__main__':
    # 启动Flask应用，启用多线程来提高并发处理能力
    app.run(host='0.0.0.0', port=5153, threaded=True)

'''
请求方式
'''
# http://localhost:5153/translation?text=hello world

# import requests
#
# url = 'http://localhost:5153/translation'
# data = {'text':'hello world'}
#
# response = requests.get('{}?text={}'.format(url, data.get('text')))
#
# if response.status_code == 200:
#     translation = response.json()
#     print(translation)
# else:
#     print("Error:", response.status_code)


'''
json数据格式的部署方式
'''
# import flask
# from flask import Flask
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,onnx
# import json
# import torch
#
# '''
# https://huggingface.co/Helsinki-NLP/opus-mt-en-zh
# '''
# #torch.set_num_threads(4)
# device = 'cpu'
# tokenizer = AutoTokenizer.from_pretrained("shanxi_model")
# model = AutoModelForSeq2SeqLM.from_pretrained("shanxi_model").to(device)
# with torch.inference_mode():
#     model.eval()
#
# app = Flask(__name__)
#
# @app.route('/translation', methods=['POST', 'GET'])
# def transaltion():
#     input_text = flask.request.get_json().get('text')
#     # input_text = [input_text]
#     with torch.no_grad():
#         translated = model.generate(**tokenizer(str(input_text), return_tensors="pt", padding=True).to(device))
#         res = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
#         return json.dumps({'translation':res}, ensure_ascii=False)
#
#
# if __name__ == '__main__':
#     # 启动Flask应用，启用多线程来提高并发处理能力
#     app.run(host='0.0.0.0', port=5153, threaded=True)
#
'''
请求方式
'''
# import requests
#
# url = 'http://localhost:5153/translation'
# data = {'text':'hello world'}
#
# response = requests.post(url, json=data)
#
# if response.status_code == 200:
#     translation = response.json()
#     print(translation)
# else:
#     print("Error:", response.status_code)
