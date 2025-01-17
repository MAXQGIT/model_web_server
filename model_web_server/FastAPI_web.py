from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import uvicorn

# Initialize the model and tokenizer
device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained("test_model")
model = AutoModelForSeq2SeqLM.from_pretrained("test_model").to(device)
with torch.inference_mode():
    model.eval()

# FastAPI app initialization
app = FastAPI()


# Define request body with Pydantic
class TranslationRequest(BaseModel):
    text: str


'''
post方法
'''


@app.post("/translation")
async def translation(request: TranslationRequest):
    input_text = request.text

    # Perform translation
    with torch.no_grad():
        translated = model.generate(
            **tokenizer(str(input_text), return_tensors="pt", padding=True).to(device)
        )
        res = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return {"translation": res}


'''
GET方法
'''


# @app.get("/translation")
# async def translation(request: Request):
#     # Get the 'text' parameter from the query string
#     input_text = request.query_params.get("text")
#
#     # If 'text' parameter is missing, return an error message
#     if not input_text:
#         return {"error": "Missing 'text' query parameter"}
#
#     # Perform translation
#     with torch.no_grad():
#         translated = model.generate(
#             **tokenizer(str(input_text), return_tensors="pt", padding=True).to(device)
#         )
#         res = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
#
#     # Return the result as JSON
#     return {"translation": res}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5153)

'''
get请求方式
'''
# http://localhost:5153/translation?text=Hello world
'''
post请求方式
'''
# curl -X POST \
#   http://localhost:5153/translation \
#   -H "Content-Type: application/json" \
#   -d '{"text": "hello world"}'


'''
请求方式
'''
# import requests
# import sys
# url = "http://localhost:5153/translation"
#
# data = {"text": "hello world"}
#
# response = requests.post(url, json=data)
#
# if response.status_code == 200:
#     translation = response.json()['translation']
#     print(translation)
# else:
#     print("Error:", response.status_code)