
# wrapper for text sentiment (using transformers pipeline)
from transformers import pipeline

class TextSentiment:
    def __init__(self, model_name=None):
        # if model_name provided, pipeline will load it; otherwise default
        self.pipe = pipeline('sentiment-analysis', model=model_name) if model_name else pipeline('sentiment-analysis')

    def predict(self, text):
        return self.pipe(text)[0]
