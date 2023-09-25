import openai
from datetime import datetime
import os
import pynecone as pc
from pynecone.base import Base
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.agents.tools import Tool
from langchain.agents import initialize_agent
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

openai.api_key = os.getenv("OPENAI_API_KEY")

class CustomVectorDB():
    CHROMA_PERSIST_DIR = "chroma-persist"
    CHROMA_COLLECTION_NAME = "dosu-bot"
    _db = None
    _retriever = None

    def __init__(self):
        self._db = Chroma(
            persist_directory=self.CHROMA_PERSIST_DIR,
            embedding_function=OpenAIEmbeddings(),
            collection_name=self.CHROMA_COLLECTION_NAME,
        )
        self._retriever = self._db.as_retriever()


    def query(self, query: str, use_retriever: bool = True) -> list[str]:
        if use_retriever:
            docs = self._retriever.get_relevant_documents(query)
        else:
            docs = self._db.similarity_search(query)

        str_docs = [doc.page_content for doc in docs]

        return str_docs


template = """
<이전대화>
{before_chat}
</이전대화>

<질문>
{query}
</질문>

<참고자료>
{document}
</참고자료>

너는 카카오에 고용된 직원으로, <GUIDELINE>을 반드시 준수해야해.
필요 시 <참고자료>와 <이전대화>를 활용해 대화에서 <질문>에 대한 대답을 생성해 줘.
<GUIDELINE>
1. 절대 <질문>에 없는 내용에 대해 미리 대답하지 말 것.
2. 절대 참고자료를 직접 확인하라는 의미의 말을 하지 말 것.
</GUIDELINE>
"""

llm = ChatOpenAI(temperature=0.0, max_tokens=500, model="gpt-3.5-turbo-16k")
custom_vectordb = CustomVectorDB()

def create_chain(llm, output_key='output'):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=template,
        ),
        output_key=output_key,
        verbose=True,
    )
chain = create_chain(llm)


def chat_with_agent(text, func_messages) -> str:
    # 무한루프(이전 출력의 결과를 질문으로 받아들이고 반복적으로 대답) 및 오류로 인해 사용 불가ㅠㅜ
    tools =[
        Tool(
            name="search_vectordb",
            func=custom_vectordb.query,
            description="카카오싱크의 메뉴얼을 참조 할 때 유용합니다.",
        ),
        Tool(
            name="before_chat",
            func=func_messages,
            description="이전 대회를 참고해서 대답해야 할 때 유용합니다.",
        ),
    ]

    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    result = agent.run(text)
    print("\n=== result ===" )
    print(result)
    return result


def just_chat(text, func_messages) -> str:
    before_chat = func_messages(text)

    result = chain(dict(
        before_chat=before_chat,
        query=text,
        document=custom_vectordb.query(text)
    ))

    # Return
    return result['output']


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

    def func_messages(self, query):
        before_chat = ""
        for history in self.messages:
            before_chat = before_chat + f"{history.role} : {history.text}\n"

        return before_chat

    def post(self):
        self.messages = self.messages + [
            Message(
                role='user',
                text=self.text,
                created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
            )
        ]

        if self.text.strip():
            answer_text = just_chat(self.text, self.func_messages)
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
