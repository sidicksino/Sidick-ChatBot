"""Local test UI (Gradio). For the portfolio, server.py + widget/sidick-widget.js are used instead."""
import gradio as gr

from chatbot import SidickBot

bot = SidickBot()


def respond(message, history):
    text = ""
    for piece in bot.stream_answer(message, history):
        text += piece
        yield text


gr.ChatInterface(
    respond,
    title="Chat with Sidick",
    description="Ask me about my skills, projects and background.",
    examples=["Who are you?", "What do you know how to build?", "Tell me about SinoCoffee"],
).launch()
