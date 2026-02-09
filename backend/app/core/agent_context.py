from app.memory.long_term_file_memory import LongTermFileMemory
from app.memory.memory import ShortTermMemory
import uuid

class AgentContext:
    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"user-{uuid.uuid4()}"

        self.tools = {}

        self.long_memory = LongTermFileMemory()

        self.short_memory = ShortTermMemory(limit=5)

