from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import gradio as gr

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

def translate(text):
  tokenizer.src_lang = "en"
  encoded_hi = tokenizer(text, return_tensors="pt")
  generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("fr"))
  return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

inputs = gr.inputs.Textbox(lines=5, label="Input Text")
outputs = gr.outputs.Textbox(label="Output Text")

title = "m2m100"
description = "demo for Facebook m2m100 english to french. To use it, simply add your text, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2010.11125'>Beyond English-Centric Multilingual Machine Translation</a> | <a href='https://github.com/pytorch/fairseq'>Github Repo</a></p>"

gr.Interface(translate, inputs, outputs, title=title, description=description, article=article).launch()