import React from 'react';
import './styles.css';
import ChatInterface from './components/ChatInterface';
import ResultsDashboard from './components/ResultsDashboard';

function App() {
  return (
    <div className="App">
      <h1 className="text-center">Plataforma de Comparación de Chatbots</h1>
      <ChatInterface />
    </div>
  );
}

export default App;