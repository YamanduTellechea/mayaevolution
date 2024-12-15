import React from 'react';

const ModeSelector = ({ mode, setMode }) => {
  return (
    <select value={mode} onChange={(e) => setMode(e.target.value)}>
      <option value="rag">RAG</option>
      <option value="finetune">Fine-Tune (Llama2)</option>
      <option value="hybrid">Híbrido</option>
    </select>
  );
};

export default ModeSelector;
