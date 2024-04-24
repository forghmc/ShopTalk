import gradio as gr
import asyncio, httpx
import async_timeout
from loguru import logger
from typing import Optional, List
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

class Message(BaseModel):
    role: str
    content: str

async def generate_image(description: str, delay: int = 30) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": description,
        "n": 1,  # Number of images to generate
        "size": "1024x1024"  # Image size
    }
    try:
        async with httpx.AsyncClient(timeout=delay) as client:
            response = await client.post("https://api.openai.com/v1/images/generations", headers=headers, json=data)
            if response.status_code == 200:
                image_data = response.json()['data'][0]
                return image_data['url']
    except asyncio.TimeoutError:
        logger.error("Timeout!")
    return None

async def predict(input, history, image_display, chatbot):
    history.append(Message(role="user", content=input))
    image_url = await generate_image(input)
    
    if image_url:
        response = f"Based on your input '{input}', here is the image:"
        chatbot.append(f"{response} [See image below]")
        return [image_url, history]
    else:
        response = "Sorry, could not generate an image at this time."
        chatbot.append(response)
        return [None, history]

def clear_all(chatbot, image_display):
    chatbot.clear()
    image_display.clear()
    return [], []

with gr.Blocks() as demo:
    logger.info("Starting Demo...")
    chatbot = gr.Chatbot(label="Visual Chat")
    image_display = gr.Image()
    state = gr.State([])

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter a detailed description (e.g., 'two cats with breed name')")
        txt.submit(predict, inputs=[txt, state, image_display, chatbot], outputs=[image_display, state])

    clear = gr.Button("Clear")
    clear.click(clear_all, inputs=[chatbot, image_display], outputs=[chatbot, image_display])

demo.launch(server_port=8080)
