## Multi-Modal Image and Text Generation
Multi-Modal Image and Text Generation is a model that generates images and text based on the input text. The model is based on the Gemini gemini-1.5-flash  and llama-3.2-1b-preview models.
Libraries used:
- **Gradio** for the UI
- **Groq** cloud API for pulling model
- **Google GenAI** for image to text conversion
- **Pillow** for image processing

To run this example, you need to install the following libraries:
- go to groq console create an api key and inspect available models [here](https://console.groq.com/playground)
- install groq using ```pip install groq```
 - install gradio using ```pip install gradio```
 - install google genai  using ```pip install google-generativeai```
 - install PILL using ```pip install pillow```
 - run the script using ```python main.py```