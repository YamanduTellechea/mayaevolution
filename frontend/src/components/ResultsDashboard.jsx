import React, { useEffect, useState } from 'react';

const ResultsDashboard = () => {
  const [results, setResults] = useState(null);

  useEffect(() => {
    fetch("http://localhost:8000/api/results/")
      .then(res => res.json())
      .then(data => setResults(data));
  }, []);

  if (!results) return <div className="text-center mt-5">Cargando resultados...</div>;

  return (
    <div className="container mt-5">
      <h2 className="text-center mb-4">Comparación de Resultados</h2>
      <div className="card p-3 mb-4">
        <h5>Input:</h5>
        <p>{results.comparisons[0].input}</p>
        <h5>Resultados:</h5>
        <ul className="list-group">
          <li className="list-group-item">
            <strong>RAG:</strong> {results.comparisons[0].RAG}
          </li>
          <li className="list-group-item">
            <strong>Fine-Tune:</strong> {results.comparisons[0].FineTune}
          </li>
          <li className="list-group-item">
            <strong>Híbrido:</strong> {results.comparisons[0].Hybrid}
          </li>
        </ul>
      </div>
      <div className="card p-3">
        <h5>Métricas:</h5>
        <ul>
          <li>BLEU: RAG={results.metrics.BLEU.RAG}, FineTune={results.metrics.BLEU.FineTune}, Hybrid={results.metrics.BLEU.Hybrid}</li>
          <li>Cost Estimate: RAG={results.metrics.CostEstimate.RAG}, FineTune={results.metrics.CostEstimate.FineTune}, Hybrid={results.metrics.CostEstimate.Hybrid}</li>
        </ul>
      </div>
    </div>
  );
};

export default ResultsDashboard;
