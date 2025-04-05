
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import './styles.css';
import ChatInterface from './components/ChatInterface';
import ChatHistory from "./components/ChatHistory";


function HomePage() {
  return (
    <div className="homepage">
      <h1>Maya Movie Assistant</h1>
      <p>Elige una opción:</p>
      <div className="homepage-options">
        <Link to="/chat" className="homepage-button">Asistente</Link>
        <Link to="/history" className="homepage-button">Historial</Link>
      </div>
    </div>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/chat" element={<ChatInterface />} />
        <Route path="/history" element={<ChatHistory />} />
      </Routes>
    </Router>
  );
}

export default App;

