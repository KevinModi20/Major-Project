import gradio as gr
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
model = MBartForConditionalGeneration.from_pretrained(
    "facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained(
    "facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")


def greet(Input):
    model_inputs = tokenizer(Input, return_tensors="pt")
# translate from English to Hindi
    generated_tokens = model.generate(
        **model_inputs, forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"])
    translation = tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True)
    return translation


demo = gr.Interface(fn=greet, inputs=gr.Textbox(
    lines=1, placeholder=" "), outputs="text",)
demo.launch(share=True)
