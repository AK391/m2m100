import soundfile as sf
import torch
from transformers import Wav2Vec2ForMaskedLM, Wav2Vec2Tokenizer
import gradio as gr

# load pretrained model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h")

# load audio

def wav2vec2(audio):
    audio_input, _ = sf.read(audio.name)
    input_values = tokenizer(audio_input, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription

inputs = gr.inputs.Audio(label=None, type="file")
outputs =  gr.outputs.Textbox(label="Output Text")

title = "Wav2vec2"
description = "demo for OpenAI GPT-2. To use it, simply add your text, or click one of the examples to load them and optionally add a text label seperated by commas to help clip classify the image better. Read more at the links below."
article = "<p style='text-align: center'><a href='https://openai.com/blog/clip/'>CLIP: Connecting Text and Images</a> | <a href='https://github.com/openai/CLIP'>Github Repo</a></p>"

gr.Interface(wav2vec2, inputs, outputs, title=title, description=description, article=article).launch()