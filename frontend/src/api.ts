// api.ts
import axios from "axios";

const API_URL = "https://harmesh95-smart-health-llm.hf.space";

// Connection status types
export type ConnectionStatus = "connected" | "connecting" | "error";

// Global state
let backendHealthy = false;
let connectionStatus: ConnectionStatus = "connecting";

// Event listeners for connection status changes
const statusListeners: ((status: ConnectionStatus) => void)[] = [];

// Notify all listeners about status change
const notifyStatusChange = (status: ConnectionStatus) => {
  connectionStatus = status;
  statusListeners.forEach(listener => listener(status));
};

// ---- Health check ----
export const checkHealth = async (): Promise<boolean> => {
  notifyStatusChange("connecting");

  try {
    const response = await axios.get(`${API_URL}/`, {
      timeout: 10000, // 10 second timeout
      headers: {
        'Accept': 'application/json',
      }
    });

    // Handle different response formats
    backendHealthy = response.data?.status === "ok" ||
                    response.data?.status === "healthy" ||
                    response.status === 200;

    console.log("✅ Backend health:", backendHealthy, response.data);
    notifyStatusChange(backendHealthy ? "connected" : "error");
    return backendHealthy;
  } catch (error: any) {
    console.error("❌ Backend not reachable:", error.message);
    backendHealthy = false;
    notifyStatusChange("error");
    return false;
  }
};

// ---- Subscribe to connection status changes ----
export const onConnectionStatusChange = (listener: (status: ConnectionStatus) => void) => {
  statusListeners.push(listener);
  // Immediately call with current status
  listener(connectionStatus);

  // Return unsubscribe function
  return () => {
    const index = statusListeners.indexOf(listener);
    if (index > -1) {
      statusListeners.splice(index, 1);
    }
  };
};

// ---- Get current connection status ----
export const getConnectionStatus = (): ConnectionStatus => {
  return connectionStatus;
};

// ---- Ask backend ----
export const askBackend = async (message: string, userId: string, image?: string) => {
  // Check health if not already known to be healthy
  if (!backendHealthy) {
    await checkHealth();
  }

  if (!backendHealthy) {
    throw new Error("Backend service is currently unavailable. Please try again later.");
  }

  try {
    const payload: any = {
      user_id: userId,
      message: message || "Analyze this image",
    };

    if (image) {
      payload.image = image;
    }

    const response = await axios.post(`${API_URL}/ask`, payload, {
      timeout: 30000, // 30 second timeout for AI responses
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      }
    });

    // Update connection status to connected on successful request
    notifyStatusChange("connected");

    return response.data.answer;
  } catch (error: any) {
    console.error("Error calling backend:", error);

    // Update connection status based on error type
    if (error.code === 'ECONNREFUSED' || error.code === 'NETWORK_ERROR') {
      backendHealthy = false;
      notifyStatusChange("error");
    } else if (error.response?.status >= 500) {
      // Server error
      notifyStatusChange("error");
    }

    // Provide more specific error messages
    if (error.response) {
      // Server responded with error status
      throw new Error(`Server error: ${error.response.status} - ${error.response.data?.error || 'Unknown error'}`);
    } else if (error.request) {
      // Request made but no response received
      throw new Error("No response from server. Please check your internet connection.");
    } else {
      // Something else happened
      throw new Error(`Request failed: ${error.message}`);
    }
  }
};

// ---- Test connection (for manual testing) ----
export const testConnection = async (): Promise<{ success: boolean; message: string }> => {
  try {
    const isHealthy = await checkHealth();
    return {
      success: isHealthy,
      message: isHealthy ? "✅ Backend is connected and healthy" : "❌ Backend is not healthy"
    };
  } catch (error: any) {
    return {
      success: false,
      message: `❌ Connection test failed: ${error.message}`
    };
  }
};

// Automatically check health when the module is loaded
// But don't block the application startup
setTimeout(() => {
  checkHealth().catch(console.error);
}, 1000);

// Export for debugging
export const getBackendHealth = () => backendHealthy;