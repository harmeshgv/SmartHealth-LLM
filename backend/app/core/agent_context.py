from app.memory.long_term_file_memory import LongTermFileMemory
from app.memory.memory import ShortTermMemory

class AgentContext:
    def __init__(self, user=None):
        self.user = user
        self.session_id = f"user-{user.id}" if user else "anonymous"

        self.tools = {}

        # LONG TERM: persisted to disk
        self.long_memory = LongTermFileMemory()

        # SHORT TERM: only last 5 in RAM
        self.short_memory = ShortTermMemory(limit=5)

        self.debug = True
