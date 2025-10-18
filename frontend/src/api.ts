import axios from "axios";

const API_URL = "https://harmesh95-smart-health-llm.hf.space"; // FastAPI backend URL

// ---- Health check (auto run once) ----
let backendHealthy = false;

export const checkHealth = async (): Promise<boolean> => {
  try {
    const response = await axios.get(`${API_URL}/`);
    backendHealthy = response.data.status === "ok";
    console.log("✅ Backend health:", backendHealthy);
    return backendHealthy;
  } catch (error) {
    console.error("❌ Backend not reachable:", error);
    backendHealthy = false;
    return false;
  }
};

// Automatically check once when file is imported
checkHealth();

// ---- Setup LLM ----
export const setupLLM = async (apiKey: string, baseUrl: string, model: string) => {
  if (!backendHealthy) {
    await checkHealth();
    if (!backendHealthy) throw new Error("Backend not reachable");
  }

  try {
    const response = await axios.post(`${API_URL}/setup_llm`, {
      api_key: apiKey,
      provider:baseUrl,
      model:model,
    });
    // Extract user_id from backend message like: "LLM set up for user <id>"
    return response.data.message.match(/user (\S+)/)?.[1] || "";
  } catch (error) {
    console.error("Error setting up LLM:", error);
    throw error;
  }
};

// ---- Ask backend ----
export const askBackend = async (message: string, userId: string) => {
  if (!backendHealthy) {
    await checkHealth();
    if (!backendHealthy) throw new Error("Backend not reachable");
  }

  try {
    const response = await axios.post(`${API_URL}/ask`, {
      user_id: userId,
      message,
    });
    return response.data.answer;
  } catch (error) {
    console.error("Error calling backend:", error);
    throw error;
  }
};
