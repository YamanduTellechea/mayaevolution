import React, { useState, useEffect } from "react";
import "./styles.css";

const ChatInterface = () => {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("rag");
  const [messages, setMessages] = useState([]);

  const sendMessage = async () => {
    if (!query.trim()) {
      alert("Por favor, escribe algo antes de enviar.");
      return;
    }

    try {
      const response = await fetch("http://localhost:8000/api/chat/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, mode }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }

      const data = await response.json();
      setMessages([...messages, { user: query, bot: data.answer }]);
      setQuery("");
    } catch (error) {
      console.error("Error:", error);
      alert("Hubo un problema con el servidor.");
    }
  };

  // Scroll automático al fondo de la caja de chat
  useEffect(() => {
    const chatBox = document.getElementById("chat-box");
    if (chatBox) {
      chatBox.scrollTop = chatBox.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="container mt-5">
      <div className="card">
        <h2 className="text-center mb-4">Plataforma de Comparación de Chatbots</h2>
        <div className="row mb-3">
          {/* Selector de Modo */}
          <div className="col-md-4">
            <select
              className="form-select"
              value={mode}
              onChange={(e) => setMode(e.target.value)}
            >
              <option value="rag">RAG</option>
              <option value="finetune">Fine-Tune</option>
              <option value="hybrid">Híbrido</option>
            </select>
          </div>
          {/* Input */}
          <div className="col-md-6">
            <input
              type="text"
              className="form-control"
              placeholder="Escribe tu pregunta..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            />
          </div>
          {/* Botón */}
          <div className="col-md-2">
            <button className="w-100" onClick={sendMessage}>
              Enviar
            </button>
          </div>
        </div>

        {/* Chat Box */}
        <div id="chat-box" className="chat-box">
          {messages.map((m, i) => (
            <div key={i} className="chat-message mb-2">
              <div className="text-primary">
                <strong>Tú:</strong> {m.user}
              </div>
              <div className="text-success">
                <strong>Bot:</strong> {m.bot}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
