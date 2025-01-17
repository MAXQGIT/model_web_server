from sanic import Sanic
from sanic.response import json
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import asyncio

# pip install gunicorn uvicorn sanic
# Initialize the model and tokenizer
device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained("test_model")
model = AutoModelForSeq2SeqLM.from_pretrained("test_model").to(device)
with torch.inference_mode():
    model.eval()

# Initialize Sanic app
app = Sanic("TranslationApp")


# Define request body with Pydantic (we will validate data manually in Sanic)
class TranslationRequest(BaseModel):
    text: str


'''post请求方式'''
@app.post("/translation")
async def translation(request):
    # Manual validation of incoming request body
    data = request.json
    try:
        request_data = TranslationRequest(**data)
    except Exception as e:
        return json({"error": "Invalid request data", "message": str(e)}, status=400)

    input_text = request_data.text

    # Perform translation
    with torch.no_grad():
        translated = model.generate(
            **tokenizer(str(input_text), return_tensors="pt", padding=True).to(device)
        )
        res = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return json({'translation':res})
'''get请求方式'''


# @app.get("/translation")
# async def translation(request):
#     # 从查询参数中获取 'text' 参数
#     input_text = request.args.get('text')
#
#     # 如果没有提供 'text' 参数，返回错误信息
#     if not input_text:
#         return json({"error": "Missing 'text' query parameter"}, status=400)
#
#     # 使用 tokenizer 进行文本编码
#     inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
#
#     # Perform translation
#     with torch.no_grad():
#         translated = model.generate(
#             **inputs  # 解包字典参数
#         )
#         res = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
#
#     # 返回 JSON 格式的翻译结果
#     return json({'translation': res})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5153)

'''
get请求方式
'''
# http://localhost:5153/translation?text=Hello world

'''
post请求方式
'''

# curl -X POST http://localhost:5153/translation -H "Content-Type: application/json" -d '{"text": "Hello, world!"}'
'''
post请求方式
'''
# import requests
# import sys
# url = "http://localhost:5153/translation"
# # data = sys.argv[1]
#
# data = {"text": "hello world"}
#
# response = requests.post(url, json=data)
# if response.status_code == 200:
#     translation = response.json()[translation]
#     print(translation)
# else:
#     print("Error:", response.status_code)
