import { HealthResponse } from "../lib/types";

interface ModelBadgeProps {
  health: HealthResponse;
}

export default function ModelBadge({ health }: ModelBadgeProps) {
  return (
    <div className="inline-flex items-center gap-3 bg-[var(--color-cream-dark)] border border-[var(--color-border)] rounded px-4 py-2 text-sm">
      <span className="font-[family-name:var(--font-playfair)] font-bold text-[var(--color-accent-maroon)]">
        {health.model}
      </span>
      <span className="text-[var(--color-border)]">|</span>
      <span className="text-[var(--color-text-muted)]">
        MAE: {health.metrics.MAE.toFixed(2)}
      </span>
      <span className="text-[var(--color-border)]">|</span>
      <span className="text-[var(--color-text-muted)]">
        R²: {(health.metrics["R²"] * 100).toFixed(1)}%
      </span>
    </div>
  );
}
