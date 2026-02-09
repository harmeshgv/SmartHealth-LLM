import json
import os
from typing import List

class LongTermFileMemory:
    def __init__(self, base_dir="memory_store"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _filepath(self, session_id):
        return os.path.join(self.base_dir, f"{session_id}.json")

    async def save(self, session_id, user_message, agent_output):
        filepath = self._filepath(session_id)

        # Load existing memory
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
        else:
            data = []

        # Append new entry
        data.append({
            "user_message": user_message,
            "agent_output": agent_output
        })

        # Save back to file
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    async def get(self, session_id) -> List[dict]:
        filepath = self._filepath(session_id)

        if not os.path.exists(filepath):
            return []

        with open(filepath, "r") as f:
            return json.load(f)

    async def clear(self, session_id):
        filepath = self._filepath(session_id)
        if os.path.exists(filepath):
            os.remove(filepath)
