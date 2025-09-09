import React, { useState, useRef, useEffect } from "react";
import { askBackend } from "./api";

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
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setMessages(prev => [...prev, { type: "user", content: query.trim() }]);
    setQuery("");
    const answer = await askBackend(query.trim());
    setMessages(prev => [...prev, { type: "bot", content: answer }]);
    setLoading(false);
  };

  const toBase64 = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = error => reject(error);
    });

  const handleImage = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setMessages(prev => [...prev, { type: "user", content: "📷 Image uploaded", isImage: true }]);

    try {
      const base64Image = await toBase64(file);
      const answer = await askBackend(base64Image); // Your API must accept base64 string
      setMessages(prev => [...prev, { type: "bot", content: answer }]);
    } catch {
      setMessages(prev => [...prev, { type: "bot", content: "Error processing image." }]);
    } finally {
      setLoading(false);
      setShowUpload(false);
    }
  };

  const onEnterPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && query.trim() && !loading) {
      handleSend();
    }
  };

  return (
    <div className="chat-page-container">
      <button className="back-btn" onClick={onBack}>
        ← Back
      </button>
      <h2 className="chat-title">What's on your mind today?</h2>
      <div className="chat-list">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={
              "chat-bubble " +
              (msg.type === "user" ? "right-bubble" : "left-bubble") +
              (msg.isImage ? " img-bubble" : "")
            }
          >
            {msg.content}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <div className={"chat-input-bar " + (loading ? "disabled" : "")}>
        <button
          className="icon-btn"
          onClick={() => setShowUpload(show => !show)}
        >
          +
        </button>
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={onEnterPress}
          placeholder="Ask anything"
          disabled={loading}
          autoFocus
        />
        <button
          className="icon-btn mic"
          disabled
          title="Voice coming soon"
        >
          <i className="fa fa-microphone" />
        </button>
        <button
          className="icon-btn"
          onClick={handleSend}
          disabled={loading || !query.trim()}
        >
          <i className="fa fa-paper-plane" />
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
    </div>
  );
}
