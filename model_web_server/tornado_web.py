import tornado.ioloop
import tornado.web
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import asyncio
import json

# Initialize the model and tokenizer
device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained("test_model")
model = AutoModelForSeq2SeqLM.from_pretrained("test_model").to(device)
model.eval()


# Define async function for translation
async def generate_translation(input_text: str):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    translated = await asyncio.to_thread(model.generate, **inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


# Define the Tornado request handler
class MainHandler(tornado.web.RequestHandler):
    '''post请求方式'''

    # async def post(self):
    #     data = json.loads(self.request.body)
    #     input_text = data.get('text')
    #
    #     # Get translation result
    #     result = await generate_translation(input_text)
    #     self.write({"translation": result})

    '''get请求方式'''
    async def get(self):
        input_text = self.get_argument('text', None)
        result = await generate_translation(input_text)
        self.write({"translation": result})


# Create Tornado application
def make_app():
    return tornado.web.Application([
        (r"/translation", MainHandler),
    ])


if __name__ == "__main__":
    # Run the Tornado application
    app = make_app()
    app.listen(5153)  # Port number to listen on
    tornado.ioloop.IOLoop.current().start()


'''
get请求方式
'''
# http://localhost:5153/translation?text=hello world

'''
POST请求方式
'''
# import requests
#
# url = 'http://localhost:5153/translation'
# data = {'text': 'Hello, world!'}
#
# response = requests.post(url, json=data)
#
# if response.status_code == 200:
#     translation = response.json().get('translation')
#     print(translation)
# else:
#     print("Error:", response.status_code)
