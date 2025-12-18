from langchain_classic.memory import ConversationBufferMemory

def get_memory_module():
    return ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )
