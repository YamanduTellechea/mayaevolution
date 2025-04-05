// import React, { useState } from "react";
// import "./style.css";

// const ChatInterface = () => {
//   // Estados
//   const [query, setQuery] = useState("");
//   const [mode, setMode] = useState("rag");
//   const [messages, setMessages] = useState([]);

//   const sendMessage = async () => {
//     if (!query.trim()) return;

//     try {
//       const response = await fetch("http://localhost:8000/api/chat/", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ query, mode }),
//       });

//       if (!response.ok) throw new Error("Error en la respuesta del servidor");

//       const data = await response.json();

//       // Validar que "answer" sea un array
//       if (data && Array.isArray(data.answer)) {
//         setMessages((prevMessages) => [
//           ...prevMessages,
//           { user: query, bot: data.answer },
//         ]);
//       } else {
//         alert("Error: El servidor no devolvió los datos en el formato esperado.");
//       }

//       setQuery("");
//     } catch (error) {
//       console.error("Error al enviar el mensaje:", error);
//       alert("Error al enviar el mensaje.");
//     }
//   };

//   return (
//     <div className="container">
//       <h3>Encuentra tu pelicula</h3>

//       <div className="input-group">
//         <select value={mode} onChange={(e) => setMode(e.target.value)}>
//           <option value="rag">RAG</option>
//           <option value="gpt4">GPT4</option>          
//         </select>
//         <input
//           type="text"
//           placeholder="Describe la pelicula..."
//           value={query}
//           onChange={(e) => setQuery(e.target.value)}
//         />
//         <button onClick={sendMessage}>Enviar</button>
//       </div>

//       <div className="chat-box">
//         {messages.map((m, i) => (
//           <div key={i} className="message">
//             {/* Mensaje del usuario */}
//             <div className="user-message">{m.user}</div>
//             {/* Respuesta del bot */}
//             <div className="bot-message">
//               {m.bot.map((movie, index) => (
//                 <div key={index} className="movie-card">
//                   <h4>{movie.title || "Untitled"}</h4>
//                   <p>
//                     <strong>Overview:</strong>{" "}
//                     {movie.overview || "No overview available."}
//                   </p>
//                   <p>
//                     <strong>Genres:</strong> {movie.genres || "N/A"}
//                   </p>
//                   <p>
//                     <strong>Actors:</strong> {movie.actors || "N/A"}
//                   </p>
//                   <p>
//                     <strong>Rating:</strong> {movie.rating || "N/A"}
//                   </p>
//                 </div>
//               ))}
//             </div>
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default ChatInterface;

// import React, { useState } from "react";
// import "./style.css";
// import ModeSelector from "./ModeSelector";

// const ChatInterface = () => {
//   // Estados
//   const [query, setQuery] = useState("");
//   const [mode, setMode] = useState("rag");
//   const [messages, setMessages] = useState([]);
//   const [loading, setLoading] = useState(false);

//   const sendMessage = async () => {
//     if (!query.trim()) return;

//     setLoading(true);
//     const startTime = performance.now(); // Iniciar medición de tiempo

//     try {
//       const response = await fetch("http://localhost:8000/api/chat/", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ query, mode }),
//       });

//       if (!response.ok) throw new Error("Error en la respuesta del servidor");

//       const data = await response.json();
//       const endTime = performance.now(); // Finalizar medición de tiempo
//       const responseTime = ((endTime - startTime) / 1000).toFixed(3); // Convertir a segundos

//       // Estimación del coste para GPT-4 (0.06 $ por 1000 tokens)
//       let estimatedCost = 0;
//       if (mode === "gpt4") {
//         estimatedCost = (data?.cost || 0).toFixed(4); // Coste desde el backend
//       }

//       setMessages((prevMessages) => [
//         ...prevMessages,
//         {
//           user: query,
//           bot: data.answer,
//           time: responseTime,
//           cost: estimatedCost,
//           mode: mode,
//         },
//       ]);

//       setQuery("");
//     } catch (error) {
//       console.error("Error al enviar el mensaje:", error);
//       alert("Error al enviar el mensaje.");
//     }

//     setLoading(false);
//   };

//   return (
//     <div className="container">
//       <h3>Encuentra tu película</h3>

//       <div className="input-group">
//         <ModeSelector mode={mode} setMode={setMode} />
//         <input
//           type="text"
//           placeholder="Describe la película..."
//           value={query}
//           onChange={(e) => setQuery(e.target.value)}
//         />
//         <button onClick={sendMessage} disabled={loading}>
//           {loading ? "Cargando..." : "Enviar"}
//         </button>
//       </div>

//       <div className="chat-box">
//         {messages.map((m, i) => (
//           <div key={i} className="message">
//             {/* Mensaje del usuario */}
//             <div className="user-message">{m.user}</div>

//             {/* Información destacada del tiempo y coste */}
//             <div className="info-bar">
//               <p><strong>Tiempo de respuesta:</strong> {m.time} segundos</p>
//               {m.mode === "gpt4" && <p><strong>Coste:</strong> ${m.cost}</p>}
//             </div>

//             {/* Respuesta del bot con la lista de películas */}
//             <div className="bot-message">
//               {m.bot.length > 0 ? (
//                 m.bot.map((movie, index) => (
//                   <div key={index} className="movie-card">
//                     <h4>{movie.title || "Untitled"}</h4>
//                     <p><strong>Overview:</strong> {movie.overview || "No overview available."}</p>
//                     <p><strong>Genres:</strong> {movie.genres || "N/A"}</p>
//                     <p><strong>Actors:</strong> {movie.actors || "N/A"}</p>
//                     <p><strong>Rating:</strong> {movie.rating || "N/A"}</p>
//                   </div>
//                 ))
//               ) : (
//                 <p>No se encontraron películas.</p>
//               )}
//             </div>
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default ChatInterface;
import React, { useState } from "react";
import "./style.css";
import ModeSelector from "./ModeSelector";

const ChatInterface = () => {
  // Estados
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("rag");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!query.trim()) return;

    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/api/chat/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, mode }),
      });

      if (!response.ok) throw new Error("Error en la respuesta del servidor");

      const data = await response.json();

      // Validar que el backend devuelva el tiempo y el coste
      const responseTime = data.time || "N/A";
      const estimatedCost = data.cost || 0;

      // Validar que "answer" sea un array
      if (data && Array.isArray(data.answer)) {
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            user: query,
            bot: data.answer,
            time: responseTime,
            cost: estimatedCost,
            mode: mode,
          },
        ]);
      } else {
        alert("Error: El servidor no devolvió los datos en el formato esperado.");
      }

      setQuery("");
    } catch (error) {
      console.error("Error al enviar el mensaje:", error);
      alert("Error al enviar el mensaje.");
    }

    setLoading(false);
  };

  return (
    <div className="container">
      <h3>Encuentra tu película</h3>

      <div className="input-group">
        <ModeSelector mode={mode} setMode={setMode} />
        <input
          type="text"
          placeholder="Describe la película..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button onClick={sendMessage} disabled={loading}>
          {loading ? "Cargando..." : "Enviar"}
        </button>
      </div>

      <div className="chat-box">
        {messages.map((m, i) => (
          <div key={i} className="message">
            {/* Mensaje del usuario */}
            <div className="user-message">{m.user}</div>

            {/* Información destacada del tiempo y coste */}
            <div className="info-bar">
              <p><strong>Tiempo de respuesta:</strong> {m.time} segundos</p>
              {m.mode === "gpt4" && <p><strong>Coste:</strong> ${m.cost.toFixed(4)}</p>}
            </div>

            {/* Respuesta del bot con la lista de películas */}
            <div className="bot-message">
              {m.bot.length > 0 ? (
                m.bot.map((movie, index) => (
                  <div key={index} className="movie-card">
                    <h4>{movie.title || "Untitled"}</h4>
                    <p><strong>Overview:</strong> {movie.overview || "No overview available."}</p>
                    <p><strong>Genres:</strong> {movie.genres || "N/A"}</p>
                    <p><strong>Actors:</strong> {movie.actors || "N/A"}</p>
                    <p><strong>Rating:</strong> {movie.rating || "N/A"}</p>
                  </div>
                ))
              ) : (
                <p>No se encontraron películas.</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ChatInterface;
