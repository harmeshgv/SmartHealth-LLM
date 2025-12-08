class ShortTermMemory:
    def __init__(self, limit=5):
        self.limit = limit
        self.buffer = {}

    async def save(self, session_id, user_message, agent_output):
        if session_id not in self.buffer:
            self.buffer[session_id] = []

        self.buffer[session_id].append({
            "user_message": user_message,
            "agent_output": agent_output
        })

        self.buffer[session_id] = self.buffer[session_id][-self.limit:]

    async def get(self, session_id):
        return self.buffer.get(session_id, [])
