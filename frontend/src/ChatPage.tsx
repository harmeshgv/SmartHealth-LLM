import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import { askBackend, onConnectionStatusChange, getConnectionStatus, checkHealth } from "./api";
import "highlight.js/styles/github-dark.css";

interface Message {
  type: "user" | "bot";
  content: string;
  isImage?: boolean;
  previewUrl?: string;
}

interface ChatPageProps {
  onBack: () => void;
}

type ConnectionStatus = "connected" | "connecting" | "error";

export default function ChatPage({ onBack }: ChatPageProps) {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>(getConnectionStatus());
  const [userId] = useState<string>(() => {
    return `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  });

  const bottomRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Subscribe to connection status changes
  useEffect(() => {
    const unsubscribe = onConnectionStatusChange((status) => {
      setConnectionStatus(status);
    });

    // Manual health check on component mount
    checkHealth().catch(console.error);

    return unsubscribe;
  }, []);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Convert file to Base64
  const toBase64 = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = (error) => reject(error);
    });

  // Handle sending text messages
  const handleSend = async () => {
    if (!query.trim() || loading) return;

    setLoading(true);
    const userMessage = query.trim();
    setMessages((prev) => [...prev, { type: "user", content: userMessage }]);
    setQuery("");

    try {
      const answer = await askBackend(userMessage, userId);
      setMessages((prev) => [...prev, { type: "bot", content: answer }]);
    } catch (error: any) {
      console.error("API Error:", error);
      setMessages((prev) => [
        ...prev,
        {
          type: "bot",
          content: `⚠️ **Error**\n\n${error.message || "Something went wrong. Please try again."}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Handle uploading images
  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0 || loading) return;

    setLoading(true);
    try {
      for (const file of Array.from(files)) {
        const previewUrl = URL.createObjectURL(file);
        setMessages(prev => [
          ...prev,
          {
            type: "user",
            content: `📷 Image: ${file.name}`,
            isImage: true,
            previewUrl
          },
        ]);

        const base64Image = await toBase64(file);
        const answer = await askBackend("", userId, base64Image);

        setMessages(prev => [...prev, { type: "bot", content: answer }]);
      }
    } catch (error: any) {
      console.error("Image upload error:", error);
      setMessages(prev => [
        ...prev,
        {
          type: "bot",
          content: `⚠️ **Upload Error**\n\n${error.message || "Failed to process the image. Please try again."}`,
        },
      ]);
    } finally {
      setLoading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const onEnterPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && query.trim() && !loading) {
      handleSend();
    }
  };

  const getStatusText = () => {
    switch (connectionStatus) {
      case "connected": return "Connected";
      case "connecting": return "Connecting...";
      case "error": return "Connection Failed";
      default: return "Unknown";
    }
  };

  const handleRetryConnection = async () => {
    await checkHealth();
  };

  return (
    <div className="app-bg">
      <div className="chat-page-container">
        {/* Back Button */}
        <button className="back-btn" onClick={onBack}>
          ← Back to Home
        </button>

        {/* Connection Status */}
        <div
          className={`connection-status status-${connectionStatus} ${connectionStatus === 'error' ? 'clickable' : ''}`}
          onClick={connectionStatus === 'error' ? handleRetryConnection : undefined}
          title={connectionStatus === 'error' ? 'Click to retry connection' : ''}
        >
          <div className="status-dot"></div>
          <span>{getStatusText()}</span>
        </div>

        {/* Chat Title */}
        <h1 className="chat-title">Medical Assistant</h1>

        {/* Messages List */}
        <div className="chat-list">
          {messages.length === 0 ? (
            <div className="welcome-message">
              <p>
                👋 Welcome! I'm your medical assistant. Ask me about symptoms,
                diseases, or upload medical images for analysis.
              </p>
              <p
                style={{
                  fontSize: "0.8em",
                  opacity: 0.7,
                  marginTop: "10px",
                }}
              >
                Session ID: {userId}
              </p>
              {connectionStatus === "error" && (
                <p style={{ color: '#ef4444', marginTop: '10px' }}>
                  ⚠️ Backend connection failed. Some features may not work.
                </p>
              )}
            </div>
          ) : (
            messages.map((msg, index) => (
              <div
                key={index}
                className={`chat-bubble ${
                  msg.type === "user" ? "right-bubble" : "left-bubble"
                } ${msg.isImage ? "img-bubble" : ""}`}
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                {msg.isImage && msg.previewUrl ? (
                  <div>
                    <img
                      src={msg.previewUrl}
                      alt="uploaded"
                      style={{
                        maxWidth: "100%",
                        borderRadius: "12px",
                        marginBottom: "8px"
                      }}
                    />
                    <div style={{ fontSize: "0.9em", opacity: 0.8 }}>
                      {msg.content}
                    </div>
                  </div>
                ) : msg.type === "bot" ? (
                  <div className="markdown-content">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeHighlight]}
                    >
                      {msg.content}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <span>{msg.content}</span>
                )}
              </div>
            ))
          )}

          {/* Typing Indicator */}
          {loading && (
            <div className="chat-bubble left-bubble">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input Bar */}
        <div className={`chat-input-bar ${loading ? "disabled" : ""}`}>
          <button
            className="icon-btn"
            onClick={triggerFileInput}
            type="button"
            disabled={loading}
            title="Upload image"
          >
            📷
          </button>

          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onEnterPress}
            placeholder="Ask about symptoms, diseases, or upload an image..."
            disabled={loading}
            autoFocus
          />

          <button
            className="icon-btn"
            onClick={handleSend}
            disabled={loading || !query.trim()}
            title="Send message"
          >
            {loading ? "⏳" : "📨"}
          </button>

          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            onChange={handleImageUpload}
            style={{ display: 'none' }}
          />
        </div>
      </div>
    </div>
  );
}