import transformers
from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/emotion')
def detect_emotion():
    if request.args.get("text") is None:
        return "No text provided"
    else:
        return jsonify(emotion(request.args.get("text")))


@app.route("/chat")
def chat():
    if request.args.get("text") is None:
        return "No text provided"
    else:
        result_str = str(chatbot(transformers.Conversation(request.args.get("text")), pad_token_id=50256))
        bot_resp = result_str.find("bot >>")
        return result_str[bot_resp + 6:]


if __name__ == '__main__':
    app.run()
