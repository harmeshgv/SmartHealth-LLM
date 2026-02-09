import unittest
import os
import json
import tempfile
import shutil
from app.memory.long_term_file_memory import LongTermFileMemory

class TestLongTermFileMemory(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Create a temporary directory for the memory store
        self.test_dir = tempfile.mkdtemp()
        self.memory = LongTermFileMemory(base_dir=self.test_dir)
        self.session_id = "test_session_123"

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    async def test_initialization_creates_directory(self):
        # Check that the base directory was created
        self.assertTrue(os.path.isdir(self.test_dir))

    async def test_save_and_get_memory(self):
        # 1. Test getting memory from a session that doesn't exist
        memory_content = await self.memory.get(self.session_id)
        self.assertEqual(memory_content, [])

        # 2. Save a first message
        user_message1 = "Hello, world!"
        agent_output1 = "Hi there!"
        await self.memory.save(self.session_id, user_message1, agent_output1)

        # 3. Get the memory and verify its content
        memory_content = await self.memory.get(self.session_id)
        self.assertEqual(len(memory_content), 1)
        self.assertEqual(memory_content[0]["user_message"], user_message1)
        self.assertEqual(memory_content[0]["agent_output"], agent_output1)

        # 4. Save a second message
        user_message2 = "How are you?"
        agent_output2 = "I'm doing well, thanks!"
        await self.memory.save(self.session_id, user_message2, agent_output2)

        # 5. Get the memory again and verify it has been appended
        memory_content = await self.memory.get(self.session_id)
        self.assertEqual(len(memory_content), 2)
        self.assertEqual(memory_content[1]["user_message"], user_message2)
        self.assertEqual(memory_content[1]["agent_output"], agent_output2)

    async def test_clear_memory(self):
        # 1. Save some data
        await self.memory.save(self.session_id, "Some message", "Some reply")
        
        # Verify the file exists
        filepath = self.memory._filepath(self.session_id)
        self.assertTrue(os.path.exists(filepath))

        # 2. Clear the memory
        await self.memory.clear(self.session_id)

        # 3. Verify the file is gone
        self.assertFalse(os.path.exists(filepath))

        # 4. Get the memory and verify it's empty
        memory_content = await self.memory.get(self.session_id)
        self.assertEqual(memory_content, [])

if __name__ == '__main__':
    unittest.main()
