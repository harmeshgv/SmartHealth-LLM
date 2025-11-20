from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

class ConversationMemory:
    """
    A class to manage the conversation history for a single user session.
    
    This class encapsulates the logic for storing and retrieving chat messages,
    ensuring a clean separation of concerns from the main application logic.
    """
    def __init__(self):
        """Initializes the ConversationMemory with an empty chat history."""
        self.chat_history: list[BaseMessage] = []

    def add_message(self, message: BaseMessage):
        """Adds a message (either from a human or AI) to the history."""
        if not isinstance(message, BaseMessage):
            raise TypeError("Message must be an instance of BaseMessage (e.g., HumanMessage, AIMessage)")
        self.chat_history.append(message)

    def add_user_message(self, content: str):
        """Convenience method to add a user (human) message."""
        self.add_message(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        """Convenience method to add an AI-generated message."""
        self.add_message(AIMessage(content=content))

    def get_history(self) -> list[BaseMessage]:
        """Returns the full conversation history."""
        return self.chat_history

    def clear(self):
        """Clears the entire conversation history."""
        self.chat_history = []
