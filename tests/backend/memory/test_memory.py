import unittest
from app.memory.memory import ShortTermMemory

class TestShortTermMemory(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.memory = ShortTermMemory(limit=3)
        self.session_id_1 = "test_session_1"
        self.session_id_2 = "test_session_2"

    async def test_save_and_get_memory(self):
        # Test getting from a non-existent session
        self.assertEqual(await self.memory.get(self.session_id_1), [])

        # Save first message to session 1
        await self.memory.save(self.session_id_1, "user1", "agent1")
        
        # Verify content of session 1
        content1 = await self.memory.get(self.session_id_1)
        self.assertEqual(len(content1), 1)
        self.assertEqual(content1[0], {"user_message": "user1", "agent_output": "agent1"})

        # Save a message to session 2 to ensure separation
        await self.memory.save(self.session_id_2, "user_other", "agent_other")
        content2 = await self.memory.get(self.session_id_2)
        self.assertEqual(len(content2), 1)
        
        # Verify session 1 is unaffected
        self.assertEqual(len(await self.memory.get(self.session_id_1)), 1)


    async def test_memory_limit(self):
        # Save 4 messages to a buffer with a limit of 3
        await self.memory.save(self.session_id_1, "u1", "a1")
        await self.memory.save(self.session_id_1, "u2", "a2")
        await self.memory.save(self.session_id_1, "u3", "a3")
        await self.memory.save(self.session_id_1, "u4", "a4")

        # Get the memory and verify it only contains the last 3 messages
        content = await self.memory.get(self.session_id_1)
        self.assertEqual(len(content), 3)
        self.assertEqual(content[0]["user_message"], "u2")
        self.assertEqual(content[1]["user_message"], "u3")
        self.assertEqual(content[2]["user_message"], "u4")
    
    async def test_get_empty_memory(self):
        # Test getting memory from a session that has not been saved to
        content = await self.memory.get("non_existent_session")
        self.assertEqual(content, [])


if __name__ == '__main__':
    unittest.main()
