import React from 'react';

export default function LayerBadge({ layer, fallback }) {
  let badgeClass = 'badge-layer2';
  let label = 'Gemma 4 E2B Instruct';

  if (layer === 'quiz') {
    badgeClass = 'badge-layer1';
    label = 'Quiz Engine';
  } else if (fallback || (layer && layer.includes('fallback'))) {
    badgeClass = 'badge-fallback';
    label = 'Fallback Response';
  } else if (layer === 'gemma' || layer === 'layer2' || layer === 'Gemma 4 E2B Instruct') {
    badgeClass = 'badge-layer2';
    label = 'Gemma 4 E2B Instruct';
  }

  return <span className={`layer-badge ${badgeClass}`}>{label}</span>;
}
