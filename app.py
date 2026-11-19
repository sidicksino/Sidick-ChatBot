import gradio as gr

from chatbot import SidickBot

bot = SidickBot()


def respond(message, history):
    return bot.answer(message, history)


gr.ChatInterface(
    respond,
    type="messages",
    title="Chat with Sidick",
    description="Ask me about my skills, projects and background.",
    examples=["Who are you?", "What do you know how to build?", "Tell me about SinoCoffee"],
).launch()
