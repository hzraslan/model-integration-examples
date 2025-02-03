from transformers import  pipeline
import gradio as gr

pipe = pipeline("automatic-speech-recognition", model="hzraslan/wsper")

def inference(speech_file):
  return pipe(speech_file)["text"]

gr.Interface(inference,gr.Audio(source="microphone", type="filepath"),"text").launch(share=True)