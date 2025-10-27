// api.ts
import axios from "axios";

const API_URL = "http://localhost:8000"; // Remove the trailing slash

// ---- Health check (auto run once) ----
let backendHealthy = false;

export const checkHealth = async (): Promise<boolean> => {
  try {
    const response = await axios.get(`${API_URL}/`); // Keep slash here for root endpoint
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

// ---- Ask backend ----
// api.ts - Ensure your backend call handles images correctly
export const askBackend = async (message: string, userId: string, image?: string) => {
  if (!backendHealthy) await checkHealth();
  if (!backendHealthy) throw new Error("Backend not reachable");

  try {
    const payload: any = {
      user_id: userId,
      message: message || "Analyze this image", // Ensure message is never empty for images
    };

    if (image) {
      payload.image = image;
    }

    const response = await axios.post(`${API_URL}/ask`, payload);
    return response.data.answer;
  } catch (error) {
    console.error("Error calling backend:", error);
    throw error;
  }
};
