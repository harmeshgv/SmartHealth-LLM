import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import { askBackend, setupLLM } from "./api";
import SetupForm from "./SetupForm";
import "highlight.js/styles/github.css";

interface Message {
  type: "user" | "bot";
  content: string;
  isImage?: boolean;
}

interface ChatPageProps {
  onBack: () => void;
}

export default function ChatPage({ onBack }: ChatPageProps) {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [llmSetup, setLlmSetup] = useState(false);
  const [userId, setUserId] = useState<string>("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ---- Handle LLM setup ----
  const handleSetup = async (apiKey: string, provider: string, model: string) => {
    setLoading(true);
    try {
      const uid = await setupLLM(apiKey, provider, model);
      if (uid) {
        setUserId(uid);
        setLlmSetup(true);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // ---- Handle sending messages ----
  const handleSend = async () => {
    if (!query.trim() || !userId) return;
    setLoading(true);
    setMessages(prev => [...prev, { type: "user", content: query.trim() }]);
    const messageToSend = query.trim();
    setQuery("");

    try {
      const answer = await askBackend(messageToSend, userId);
      setMessages(prev => [...prev, { type: "bot", content: answer }]);
    } catch {
      setMessages(prev => [
        ...prev,
        { type: "bot", content: "⚠️ Error contacting backend." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // ---- Image Upload ----
  const toBase64 = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = error => reject(error);
    });

  const handleImage = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !userId) return;

    setLoading(true);
    setMessages(prev => [
      ...prev,
      { type: "user", content: "📷 Image uploaded", isImage: true },
    ]);

    try {
      const base64Image = await toBase64(file);
      const answer = await askBackend(base64Image, userId);
      setMessages(prev => [...prev, { type: "bot", content: answer }]);
    } catch {
      setMessages(prev => [
        ...prev,
        { type: "bot", content: "⚠️ Error processing image." },
      ]);
    } finally {
      setLoading(false);
      setShowUpload(false);
    }
  };

  const onEnterPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && query.trim() && !loading) handleSend();
  };

  return (
    <div className="chat-page-container">
      {!llmSetup ? (
        <SetupForm onSetup={handleSetup} />
      ) : (
        <>
          <button className="back-btn" onClick={onBack}>
            ← Back
          </button>
          <h2 className="chat-title">💬 Smart Health Assistant</h2>
          <div className="chat-list">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`chat-bubble ${
                  msg.type === "user" ? "right-bubble" : "left-bubble"
                } ${msg.isImage ? "img-bubble" : ""}`}
              >
                {msg.type === "bot" ? (
  <div className="markdown-content">
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight]}
    >
      {msg.content}
    </ReactMarkdown>
  </div>
) : (
  msg.content
)}

              </div>
            ))}
            <div ref={bottomRef} />
          </div>
          <div className={`chat-input-bar ${loading ? "disabled" : ""}`}>
            <button className="icon-btn" onClick={() => setShowUpload(s => !s)}>
              📎
            </button>
            <input
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
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
            {showUpload && (
              <input
                type="file"
                accept="image/*"
                className="upload-input"
                onChange={handleImage}
              />
            )}
          </div>
        </>
      )}
    </div>
  );
}
