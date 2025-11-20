import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { askBackend, onConnectionStatusChange, checkHealth } from "./api";
import "./App.css";

// --- SVGs ---
const BotIcon = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2 2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z"/><path d="M4 11a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-7z"/><rect x="8" y="13" width="2" height="2"/><rect x="14" y="13" width="2" height="2"/></svg>;
const UserIcon = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>;
const PlusIcon = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>;
const SendIcon = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>;
const ClipIcon = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>;
const ExitIcon = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>;
const XCircleIcon = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.47 2 2 6.47 2 12s4.47 10 10 10 10-4.47 10-10S17.53 2 12 2zm5 13.59L15.59 17 12 13.41 8.41 17 7 15.59 10.59 12 7 8.41 8.41 7 12 10.59 15.59 7 17 8.41 13.41 12 17 15.59z"/></svg>;


// Helper to convert file to base64
const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = (error) => reject(error);
  });
};

export default function ChatPage({ onBack, isDark, toggleTheme }: any) {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("connecting");
  
  // New state for staging uploads
  const [stagedImage, setStagedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  const bottomRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const sub = onConnectionStatusChange(setStatus);
    checkHealth().catch(() => {});
    return sub;
  }, []);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);

  const handleSend = async () => {
    if (!query.trim() && !stagedImage) return;

    const text = query.trim();
    const imageToSend = stagedImage;
    const previewUrlToSend = imagePreview;

    // Add user's message to chat log immediately
    setMessages(prev => [...prev, { 
      type: "user", 
      content: text,
      isImage: !!imageToSend,
      previewUrl: previewUrlToSend
    }]);

    // Clear inputs
    setQuery("");
    setStagedImage(null);
    setImagePreview(null);
    setLoading(true);

    try {
      let imageBase64: string | undefined = undefined;
      if (imageToSend) {
        imageBase64 = await fileToBase64(imageToSend);
      }

      // Replace "user-123" with real logic if needed
      const res = await askBackend(text, "user-123", imageBase64);
      setMessages(prev => [...prev, { type: "bot", content: res }]);
    } catch (e) {
      setMessages(prev => [...prev, { type: "bot", content: "⚠️ System Error. Please retry." }]);
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setStagedImage(file);
      setImagePreview(URL.createObjectURL(file));
    }
    // Reset file input value to allow re-selection of the same file
    if (fileRef.current) {
      fileRef.current.value = "";
    }
  };

  const cancelImage = () => {
    setStagedImage(null);
    setImagePreview(null);
  }

  return (
    <div className="chat-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <button className="new-chat" onClick={() => setMessages([])}>
          <PlusIcon /> New Consultation
        </button>

        <div style={{marginTop: 'auto'}}>
          <div className="icon-btn" style={{cursor:'default'}}>
            <span style={{color: status === 'connected' ? '#22c55e' : '#ef4444'}}>●</span>
            {status === 'connected' ? 'System Operational' : 'Offline'}
          </div>
          <button className="icon-btn" onClick={toggleTheme}>
            {isDark ? "Switch to Light" : "Switch to Dark"}
          </button>
          <button className="icon-btn" onClick={onBack}>
            <ExitIcon /> End Session
          </button>
        </div>
      </aside>

      {/* Main Area */}
      <main className="main-area">
        <div className="chat-scroll">
          <div className="chat-width">
            {messages.length === 0 ? (
              <div style={{textAlign:'center', marginTop:'10vh', opacity:0.5}}>
                <div style={{fontSize:'3rem', marginBottom:'20px'}}>🩺</div>
                <h2>How can I assist you today?</h2>
                <p>Describe symptoms or upload a report.</p>
              </div>
            ) : (
              messages.map((m, i) => (
                <div key={i} className="msg">
                  <div className={`avatar ${m.type}`}>{m.type === 'bot' ? <BotIcon/> : <UserIcon/>}</div>
                  <div className="bubble">
                    {m.isImage && <img src={m.previewUrl} alt="upload" style={{maxWidth:'250px', borderRadius:'12px', border:'1px solid var(--border)', marginBottom:'10px'}} />}
                    {m.type === 'bot' ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown> : m.content}
                  </div>
                </div>
              ))
            )}
            {loading && (
              <div className="msg">
                <div className="avatar bot"><BotIcon/></div>
                <div className="bubble" style={{fontStyle:'italic'}}>Analyzing...</div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </div>
        
        {/* Input Area */}
        <div className="input-area">
            {imagePreview && (
              <div className="image-preview">
                <img src={imagePreview} alt="Staged upload" />
                <button onClick={cancelImage} className="cancel-img-btn">
                  <XCircleIcon />
                </button>
              </div>
            )}
            <div className="input-float">
                <button className="send-icon" style={{background:'transparent', color:'var(--text-muted)'}} onClick={() => fileRef.current?.click()}>
                    <ClipIcon />
                </button>
                <input
                    placeholder="Type a symptom or health question..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
                />
                <button className="send-icon" onClick={handleSend} disabled={(!query.trim() && !stagedImage) || loading}>
                    <SendIcon />
                </button>
            </div>
        </div>

        <input type="file" hidden ref={fileRef} onChange={handleFileSelect} accept="image/*" />
      </main>
    </div>
  );
}