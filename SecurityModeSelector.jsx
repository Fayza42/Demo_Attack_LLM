import React from 'react';

const SecurityModeSelector = ({ mode, onModeChange }) => {
  return (
    <select
      value={mode}
      onChange={(e) => onModeChange(e.target.value)}
      className={`selector-base security-selector ${mode.toLowerCase()}`}
    >
      <option value="SAFE">🔒 SAFE</option>
      <option value="UNSAFE">🔓 UNSAFE</option>
    </select>
  );
};

export default SecurityModeSelector;