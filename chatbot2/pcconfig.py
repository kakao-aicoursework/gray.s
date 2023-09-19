import pynecone as pc

class ChatbotConfig(pc.Config):
    pass

config = ChatbotConfig(
    app_name="chatbot2",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)