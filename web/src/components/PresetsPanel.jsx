const PRESETS = [
  {
    label: 'AAPL SMA (2y)',
    prompt:
      'Run a robustness suite for AAPL with an SMA strategy, period 2y, fast_period 5-20 and slow_period 21-100.',
  },
  {
    label: 'BTC-USD SMA (2y)',
    prompt:
      'Run a robustness suite for BTC-USD with an SMA strategy, period 2y, fast_period 5-20 and slow_period 21-100.',
  },
  {
    label: 'Blank prompt',
    prompt: '',
  },
];

export default function PresetsPanel({ onSelect }) {
  return (
    <section className="card">
      <h2>Presets</h2>
      <div className="preset-grid">
        {PRESETS.map((preset) => (
          <button
            key={preset.label}
            type="button"
            className="btn btn-secondary"
            onClick={() => onSelect(preset.prompt)}
          >
            {preset.label}
          </button>
        ))}
      </div>
    </section>
  );
}
