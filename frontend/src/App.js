import React from 'react';
import './App.css';
import ChatInterface from './components/ChatInterface';
import ResultsDashboard from './components/ResultsDashboard';

function App() {
  return (
    <div className="App">
      <h1>Plataforma de Comparación de Chatbots</h1>
      <ChatInterface />
      <ResultsDashboard />
    </div>
  );
}

export default App;
