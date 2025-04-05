import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import "./style.css";

const ChatHistory = () => {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetch("http://localhost:8000/api/history/")
      .then((res) => res.json())
      .then((data) => setHistory(data))
      .catch((error) => console.error("Error al obtener el historial:", error));
  }, []);

  return (
    <div className="history-container">
      {/* Botón de volver arriba */}
      <div className="back-button-container">
        <Link to="/" className="back-button">← Volver</Link>
      </div>

      <h3>Historial de Interacciones</h3>

      <div className="history-box">
        {history.length === 0 ? (
          <p>No hay interacciones guardadas.</p>
        ) : (
          history.map((entry, index) => (
            <div key={index} className="history-entry">
              <p><strong>Modo:</strong> {entry.mode.toUpperCase()}</p>
              <p><strong>Pregunta:</strong> {entry.query}</p>
              <p><strong>Tiempo:</strong> {entry.response_time} segundos</p>
              {entry.mode === "gpt4" && <p><strong>Coste:</strong> ${entry.cost}</p>}
              <p><strong>Respuesta:</strong> {entry.response}</p>
              <p className="timestamp">{new Date(entry.timestamp).toLocaleString()}</p>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default ChatHistory;
