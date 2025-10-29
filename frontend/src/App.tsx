// App.tsx

import React, { useState, useEffect } from "react";
import ChatPage from "./ChatPage";

// In JSX:
import "./App.css";

export default function App() {
  const [started, setStarted] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(true);

  // Apply/remove dark/light CSS class on body
  useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add("dark");
      document.body.classList.remove("light");
    } else {
      document.body.classList.add("light");
      document.body.classList.remove("dark");
    }
  }, [isDarkMode]);

  return (
    <div className="app-bg">
      {!started ? (
        <>
          {/* Navbar */}
          <div className="nav-bar">
            <div className="logo">⚕️ Smart Health</div>

            <div className="nav-links">
              <a href="/">Home</a>
              <a href="/docs">Docs</a>
            </div>

            <button
  className="nav-theme-toggle icon-btn"
  onClick={() => setIsDarkMode(!isDarkMode)}
  title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
  aria-label="Toggle dark/light mode"
>
  {isDarkMode ? "☀️" : "🌙"}
</button>


<a
  href="https://github.com/harmeshgv/SmartHealth-LLM"
  className="nav-contribute"
  target="_blank"
  rel="noopener noreferrer"
>
  Contribute
</a>
          </div>

          {/* Overlay Grid */}
          <div className="app-grid" />

          <div className="center-content">
            <h1 className="main-title">
              Adaptive healthcare<br />powered by AI
            </h1>

            <div className="cta-btns">
              <button className="btn-primary" onClick={() => setStarted(true)}>
                Get Started
              </button>
              <button className="btn-secondary">Learn More</button>
            </div>

            {/* Removed old demo toggle for cleanliness */}
          </div>

          {/* Top utility bar (optional, can be kept if needed) */}
          {/* <div className="preview-bar">
            <button className="preview-btn">Preview</button>
            <button className="code-btn">Code</button>
          </div> */}
        </>
      ) : (
        // Render chat page and allow going back to landing
        <ChatPage onBack={() => setStarted(false)} />
      )}
    </div>
  );
}
