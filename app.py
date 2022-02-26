from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/emotion')
def detect_emotion():
    if request.args.get("text") is None:
        return "No text provided"
    else:
        return jsonify(emotion(request.args.get("text")))


if __name__ == '__main__':
    app.run()
