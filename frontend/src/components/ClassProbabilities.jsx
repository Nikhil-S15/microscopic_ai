import React from 'react';

function ClassProbabilities({ probabilities, compact = false }) {
  if (!probabilities || probabilities.length === 0) {
    return (
      <div className="no-data">
        <p>No probability data available</p>
      </div>
    );
  }

  return (
    <div className={`probabilities ${compact ? 'probabilities-compact' : ''}`}>
      {probabilities.map((prob, index) => {
        const percentage = prob.probability * 100;
        const isTop = index === 0;
        
        return (
          <div 
            key={prob.class_id} 
            className="probability-item"
            style={{
              animation: `fadeInScale 0.3s ease-out ${index * 0.1}s backwards`
            }}
          >
            <div className="probability-header">
              <span className={`probability-name ${isTop ? 'probability-top' : ''}`}>
                {prob.class_name}
              </span>
              <span className={`probability-value ${isTop ? 'probability-top' : ''}`}>
                {percentage.toFixed(1)}%
              </span>
            </div>
            <div className="probability-bar-container">
              <div
                className={`probability-bar ${isTop ? 'probability-bar-top' : ''}`}
                style={{ 
                  width: `${percentage}%`,
                  animation: `expandWidth 0.6s ease-out ${index * 0.1}s backwards`
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default ClassProbabilities;