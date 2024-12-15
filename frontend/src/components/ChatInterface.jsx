import React, { useState } from 'react';

const ChatInterface = () => {
  // Estados para la pregunta, el modo y los mensajes
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("rag");
  const [messages, setMessages] = useState([]);

  // Función para manejar el envío del mensaje
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

      // Actualiza los mensajes en el estado
      setMessages([...messages, { user: query, bot: data.answer }]);
      setQuery(""); // Limpia el input
    } catch (error) {
      console.error("Error al enviar el mensaje:", error);
      alert("Hubo un error al enviar el mensaje. Por favor, revisa la consola.");
    }
  };

  return (
    <div className="container mt-5">
      <h2 className="text-center mb-4">Chat</h2>
      <div className="row mb-3">
        {/* Selector de modo */}
        <div className="col-md-4">
          <select className="form-select" value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="rag">RAG</option>
            <option value="finetune">Fine-Tune</option>
            <option value="hybrid">Híbrido</option>
          </select>
        </div>
        {/* Input para escribir la pregunta */}
        <div className="col-md-6">
          <input
            type="text"
            className="form-control"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Escribe tu pregunta..."
          />
        </div>
        {/* Botón de enviar */}
        <div className="col-md-2">
          <button className="btn btn-primary w-100" onClick={sendMessage}>
            Enviar
          </button>
        </div>
      </div>
      {/* Caja de mensajes */}
      <div className="chat-box bg-light p-3 rounded">
        {messages.map((m, i) => (
          <div key={i} className="mb-2">
            <div className="text-primary"><strong>Tú:</strong> {m.user}</div>
            <div className="text-success"><strong>Bot:</strong> {m.bot}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ChatInterface;
