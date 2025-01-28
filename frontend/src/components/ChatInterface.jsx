import React, { useState } from "react";
import "./style.css";

const ChatInterface = () => {
  // Estados
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("rag");
  const [messages, setMessages] = useState([]);

  const sendMessage = async () => {
    if (!query.trim()) return;

    try {
      const response = await fetch("http://localhost:8000/api/chat/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, mode }),
      });

      if (!response.ok) throw new Error("Error en la respuesta del servidor");

      const data = await response.json();

      // Validar que "answer" sea un array
      if (data && Array.isArray(data.answer)) {
        setMessages((prevMessages) => [
          ...prevMessages,
          { user: query, bot: data.answer },
        ]);
      } else {
        alert("Error: El servidor no devolvió los datos en el formato esperado.");
      }

      setQuery("");
    } catch (error) {
      console.error("Error al enviar el mensaje:", error);
      alert("Error al enviar el mensaje.");
    }
  };

  return (
    <div className="container">
      <h3>Encuentra tu pelicula</h3>

      <div className="input-group">
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="rag">RAG</option>
          <option value="gpt4">GPT4</option>          
        </select>
        <input
          type="text"
          placeholder="Describe la pelicula..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button onClick={sendMessage}>Enviar</button>
      </div>

      <div className="chat-box">
        {messages.map((m, i) => (
          <div key={i} className="message">
            {/* Mensaje del usuario */}
            <div className="user-message">{m.user}</div>
            {/* Respuesta del bot */}
            <div className="bot-message">
              {m.bot.map((movie, index) => (
                <div key={index} className="movie-card">
                  <h4>{movie.title || "Untitled"}</h4>
                  <p>
                    <strong>Overview:</strong>{" "}
                    {movie.overview || "No overview available."}
                  </p>
                  <p>
                    <strong>Genres:</strong> {movie.genres || "N/A"}
                  </p>
                  <p>
                    <strong>Actors:</strong> {movie.actors || "N/A"}
                  </p>
                  <p>
                    <strong>Rating:</strong> {movie.rating || "N/A"}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ChatInterface;
