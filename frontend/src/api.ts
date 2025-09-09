// frontend/src/api.ts
import axios from "axios";

const API_URL = "http://127.0.0.1:8000"; // FastAPI backend URL

export const askBackend = async (query: string) => {
  try {
    const response = await axios.post(`${API_URL}/ask`, { query });
    return response.data.answer;
  } catch (error) {
    console.error("Error calling backend:", error);
    throw error;
  }
};
