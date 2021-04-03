import soundfile as sf
import torch
from transformers import Wav2Vec2ForMaskedLM, Wav2Vec2Tokenizer
import gradio as gr

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h")

def wav2vec2(audio):
    audio_input, _ = sf.read(audio.name)
    input_values = tokenizer(audio_input, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription

inputs = gr.inputs.Audio(label="Input Audio", type="file")
outputs =  gr.outputs.Textbox(label="Output Text")

title = "wav2vec 2.0"
description = "demo for Facebook AI wav2vec 2.0. To use it, simply upload your audio, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2006.11477'>wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations</a> | <a href='https://github.com/pytorch/fairseq'>Github Repo</a></p>"

gr.Interface(wav2vec2, inputs, outputs, title=title, description=description, article=article).launch()