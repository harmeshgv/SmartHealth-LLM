// ChatPage.tsx
import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import { askBackend } from "./api";
import "highlight.js/styles/github.css";

interface Message {
  type: "user" | "bot";
  content: string;
  isImage?: boolean;
  previewUrl?: string;
}

interface ChatPageProps {
  onBack: () => void;
}

export default function ChatPage({ onBack }: ChatPageProps) {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [, setShowUpload] = useState(false);
  const [userId] = useState<string>(() => {
    return `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  });
  const bottomRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
    if (!query.trim()) return;
    setLoading(true);
    setMessages((prev) => [...prev, { type: "user", content: query.trim() }]);
    const messageToSend = query.trim();
    setQuery("");

    try {
      const answer = await askBackend(messageToSend, userId);
      setMessages((prev) => [...prev, { type: "bot", content: answer }]);
    } catch (error: any) {
      setMessages((prev) => [
        ...prev,
        {
          type: "bot",
          content:
            "⚠️ Error: " +
            (error.response?.data?.error || "Contacting backend failed"),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Handle uploading one or multiple images
  const handleImage = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setLoading(true);
    try {
      for (const file of Array.from(files)) {
        const previewUrl = URL.createObjectURL(file);
        setMessages(prev => [
          ...prev,
          { type: "user", content: `📷 ${file.name}`, isImage: true, previewUrl },
        ]);

        const base64Image = await toBase64(file);
        const answer = await askBackend("", userId, base64Image);

        setMessages(prev => [...prev, { type: "bot", content: answer }]);
      }
    } catch (error: any) {
      setMessages(prev => [
        ...prev,
        {
          type: "bot",
          content: "⚠️ Error: " + (error.response?.data?.error || "Processing failed"),
        },
      ]);
    } finally {
      setLoading(false);
      setShowUpload(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const onEnterPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && query.trim() && !loading) handleSend();
  };

  return (
    <div className="app-bg">
      <div className="chat-page-container">
        <button className="back-btn" onClick={onBack}>
          ← Back
        </button>

        <h1 className="chat-title">Medical Assistant</h1>

        <div className="chat-list">
          {messages.length === 0 ? (
            <div className="welcome-message">
              <p>
                Welcome! I'm your medical assistant. Ask me about symptoms,
                diseases, or upload an image for analysis.
              </p>
              <p
                style={{
                  fontSize: "0.8em",
                  opacity: 0.7,
                  marginTop: "10px",
                }}
              >
                User ID: {userId}
              </p>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div
                key={index}
                className={`chat-bubble ${
                  msg.type === "user" ? "right-bubble" : "left-bubble"
                } ${msg.isImage ? "img-bubble" : ""}`}
              >
                {msg.isImage && msg.previewUrl ? (
                  <img
                    src={msg.previewUrl}
                    alt="uploaded"
                    style={{ maxWidth: "200px", borderRadius: "10px" }}
                  />
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

        <div className={`chat-input-bar ${loading ? "disabled" : ""}`}>
          <div className="input-wrapper">
            <button
              className="icon-btn"
              onClick={triggerFileInput}
              type="button"
              disabled={loading}
            >
              📷
            </button>

            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={onEnterPress}
              placeholder="Ask about a disease or symptoms..."
              disabled={loading}
              autoFocus
            />

            <button
              className="icon-btn"
              onClick={handleSend}
              disabled={loading || !query.trim()}
            >
              📨
            </button>

            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={handleImage}
              style={{ display: 'none' }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}