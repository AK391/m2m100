from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import gradio as gr

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

def translate(text, inlang, outlang):
  tokenizer.src_lang = inlang
  encoded_hi = tokenizer(text, return_tensors="pt")
  generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id(outlang))
  return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

inputs = gr.inputs.Textbox(lines=5, label="Input Text")
outputs = gr.outputs.Textbox(label="Output Text")

title = "m2m100"
description = "demo for Facebook m2m100 english to french. To use it, simply add your text, or click one of the examples to load them. Read more at the links below."
article = """<p style='text-align: center'><a href='https://arxiv.org/abs/2010.11125'>Beyond English-Centric Multilingual Machine Translation</a> | <a href='https://github.com/pytorch/fairseq'>Github Repo</a> | Languages covered
Afrikaans (af), Amharic (am), Arabic (ar), Asturian (ast), Azerbaijani (az), Bashkir (ba), Belarusian (be), Bulgarian (bg), Bengali (bn), Breton (br), Bosnian (bs), Catalan; Valencian (ca), Cebuano (ceb), Czech (cs), Welsh (cy),
Danish (da), German (de), Greeek (el), English (en), Spanish (es), Estonian (et), Persian (fa), Fulah (ff), Finnish (fi), French (fr), Western Frisian (fy), Irish (ga),
Gaelic; Scottish Gaelic (gd), Galician (gl), Gujarati (gu), Hausa (ha), Hebrew (he), Hindi (hi), Croatian (hr), Haitian;
Haitian Creole (ht), Hungarian (hu), Armenian (hy), Indonesian (id), Igbo (ig), Iloko (ilo), Icelandic (is), Italian (it), Japanese (ja), Javanese (jv), Georgian (ka),
Kazakh (kk), Central Khmer (km), Kannada (kn), Korean (ko), Luxembourgish; Letzeburgesch (lb), Ganda (lg), Lingala (ln), Lao (lo), Lithuanian (lt), Latvian (lv), Malagasy (mg),
Macedonian (mk), Malayalam (ml), Mongolian (mn), Marathi (mr), Malay (ms), Burmese (my), Nepali (ne), Dutch; Flemish (nl), Norwegian (no), Northern Sotho (ns), Occitan (post 1500) (oc),
Oriya (or), Panjabi; Punjabi (pa), Polish (pl), Pushto; Pashto (ps), Portuguese (pt), Romanian; Moldavian; Moldovan (ro), Russian (ru), Sindhi (sd), Sinhala; Sinhalese (si), Slovak (sk), Slovenian (sl), Somali (so), Albanian (sq),
Serbian (sr), Swati (ss), Sundanese (su), Swedish (sv), Swahili (sw), Tamil (ta), Thai (th), Tagalog (tl), Tswana (tn), Turkish (tr), Ukrainian (uk), Urdu (ur), Uzbek (uz), Vietnamese (vi), Wolof (wo), Xhosa (xh), Yiddish (yi),
Yoruba (yo), Chinese (zh), Zulu (zu)</p>"""

examples = [
  ["Life is like a box of chocolate."],
  ["How many miles is it from Earth to Neptune?"]
]

gr.Interface(translate, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()