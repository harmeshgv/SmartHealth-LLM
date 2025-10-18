import React, { useState } from "react";
import "./setup-form.css";


interface SetupFormProps {
  onSetup: (apiKey: string, baseUrl: string, model: string) => void;
}

export default function SetupForm({ onSetup }: SetupFormProps) {
  const [apiKey, setApiKey] = useState("");
  const [provider, setProvider] = useState("groq");
  const [model, setModel] = useState("");

  // Provider → API Base URL mapping (matches backend dict)
  const providerLinks: Record<string, string> = {
    groq: "https://api.groq.com/openai/v1",
    gravix: "https://api.gravixlayer.com/v1/inference",
  };

  // Available models per provider
  const modelOptions: Record<string, string[]> = {
    groq: [
      "openai/gpt-oss-120b",
      "openai/gpt-oss-20b",
      "moonshotai/kimi-k2-instruct-0905",
      "meta-llama/llama-4-scout-17b-16e-instruct",
      "meta-llama/llama-3.3-70b-versatile",
      "meta-llama/llama-3.1-8b-instant",
    ],
    gravix: [
      "mistralai/mistral-nemo-instruct-2407",
      "meta-llama/llama-3.2-3b-instruct",
      "meta-llama/llama-3.1-8b-instruct",
      "deepseek-ai/deepseek-r1-0528-qwen3-8b",
      "meta-llama/llama-3.2-1b-instruct",
      "microsoft/phi-4",
    ],
  };

  const handleSubmit = () => {
    if (!apiKey.trim() || !model) return alert("Please fill all fields!");
    const baseUrl = providerLinks[provider];
    onSetup(apiKey, baseUrl, model);
  };

  const handleProviderChange = (value: string) => {
    setProvider(value);
    setModel(modelOptions[value][0]); // auto-select first model
  };

  return (
    <div className="setup-form">
      <input
        type="text"
        placeholder="Enter API key"
        value={apiKey}
        onChange={(e) => setApiKey(e.target.value)}
      />

      <select value={provider} onChange={(e) => handleProviderChange(e.target.value)}>
        <option value="groq">Groq</option>
        <option value="gravix">Gravix Layer</option>
      </select>

      <select
        value={model}
        onChange={(e) => setModel(e.target.value)}
        disabled={!provider}
      >
        <option value="">Select a model</option>
        {modelOptions[provider].map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </select>

      <button onClick={handleSubmit}>Setup LLM</button>
    </div>

  );
}
