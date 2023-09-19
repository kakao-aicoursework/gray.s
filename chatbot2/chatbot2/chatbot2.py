import openai
import os
from datetime import datetime

import pynecone as pc
from pynecone.base import Base


openai.api_key = os.getenv("OPENAI_API_KEY")


def just_chat_chatgpt(text, historys) -> str:

    # system instruction 만들고
    system_instruction = f"assistant는 친절한 우리들의 이웃이에요."

    before_chat = []
    for history in historys:
        before_chat.append({"role": history.role, "content":history.text})
    # messages를만들고
    prompt = []
    prompt.append({"role": "system", "content": system_instruction})
    prompt.extend(before_chat)

    # API 호출
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=prompt)
    answer_text = response['choices'][0]['message']['content']
    # Return
    return answer_text


class Message(Base):
    role: str
    text: str
    created_at: str


class State(pc.State):
    """The app state."""

    text: str = ""
    messages: list[Message] = []

    @pc.var
    def get_messages(self) -> list[Message]:
        return self.messages


    def post(self):
        self.messages = self.messages + [
            Message(
                role='user',
                text=self.text,
                created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
            )
        ]

        if self.text.strip():
            answer_text = just_chat_chatgpt(self.text, self.messages)
            self.messages = self.messages + [
                Message(
                    role='assistant',
                    text=answer_text,
                    created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
                    )
                ]



def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("ChatBot 🗺", font_size="2rem"),
        pc.text(
            "just chat",
            margin_top="0.5rem",
            color="#666",
        ),
    )


def down_arrow():
    return pc.vstack(
        pc.icon(
            tag="arrow_down",
            color="#666",
        )
    )


def text_box(text):
    return pc.text(
        text,
        background_color="#fff",
        padding="1rem",
        border_radius="8px",
    )


def message(message):
    return pc.box(
        pc.vstack(
            text_box(message.role + " : " + message.text),
            pc.box(
                pc.text(" · ", margin_x="0.3rem"),
                pc.text(message.created_at),
                display="flex",
                font_size="0.8rem",
                color="#666",
            ),
            spacing="0.3rem",
            align_items="left",
        ),
        background_color="#f5f5f5",
        padding="1rem",
        border_radius="8px",
    )


def smallcaps(text, **kwargs):
    return pc.text(
        text,
        font_size="0.7rem",
        font_weight="bold",
        text_transform="uppercase",
        letter_spacing="0.05rem",
        **kwargs,
    )


def output():
    return pc.box(
        pc.box(
            smallcaps(
                "Output",
                color="#aeaeaf",
                background_color="white",
                padding_x="0.1rem",
            ),
            position="absolute",
            top="-0.5rem",
        ),
        pc.text(State.output),
        padding="1rem",
        border="1px solid #eaeaef",
        margin_top="1rem",
        border_radius="8px",
        position="relative",
    )


def index():
    return pc.container(
        header(),
        pc.vstack(
            pc.foreach(State.messages, message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        pc.input(
            placeholder="Chat anything",
            on_blur=State.set_text,
            margin_top="1rem",
            border_color="#eaeaef"
        ),
        pc.button("Post", on_click=State.post, margin_top="1rem"),
        padding="2rem",
        max_width="600px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="ChatBot")
app.compile()
