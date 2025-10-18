import gradio as gr
from src.scripts.model_load import predict as model

def predict(text):
    return model([text])[0]

app = gr.Interface(fn=predict, inputs="text", outputs="text")
