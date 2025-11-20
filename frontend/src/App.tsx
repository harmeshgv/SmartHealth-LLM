import React, { useState, useEffect } from "react";
import ChatPage from "./ChatPage";
import "./App.css";

// --- SVGs ---
const LogoIcon = () => <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2v20M2 12h20" /></svg>; // Medical Cross
const SunIcon = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>;
const MoonIcon = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>;
const ArrowRight = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>;

export default function App() {
  const [started, setStarted] = useState(false);
  const [isDark, setIsDark] = useState(true);

  // Theme Toggle
  useEffect(() => {
    document.body.className = isDark ? 'dark' : 'light';
  }, [isDark]);

  if (started) {
    return <ChatPage onBack={() => setStarted(false)} isDark={isDark} toggleTheme={() => setIsDark(!isDark)} />;
  }

  return (
    <div className="landing-wrap">
      <nav className="nav">
        <div className="logo"><LogoIcon /> MediChat AI</div>
        <button className="icon-btn" style={{width:'auto'}} onClick={() => setIsDark(!isDark)}>
          {isDark ? <SunIcon /> : <MoonIcon />}
        </button>
      </nav>

      <main className="hero">
        <div className="pill">AI-POWERED DIAGNOSTICS</div>
        <h1>
          Healthcare <br />
          <span className="highlight">Simplified.</span>
        </h1>
        <p className="subtext">
          Instant symptom analysis, lab report interpretation, and personalized health guidance. Powered by advanced medical LLMs.
        </p>
        <button className="cta-btn" onClick={() => setStarted(true)}>
          Start Consultation
        </button>
      </main>
    </div>
  );
}