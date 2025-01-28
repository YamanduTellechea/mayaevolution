import React from "react";

const ModeSelector = ({ mode, setMode }) => {
  return (
    <select
      className="form-select"
      value={mode}
      onChange={(e) => setMode(e.target.value)}
    >
      <option value="rag">RAG</option>
      <option value="gpt4">GPT4</option>      
    </select>
  );
};

export default ModeSelector;
