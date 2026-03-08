import { Player } from "../lib/types";
import { POSITION_COLORS } from "../lib/constants";

interface PlayerCardProps {
  player: Player;
  index: number;
}

export default function PlayerCard({ player, index }: PlayerCardProps) {
  return (
    <div className="bg-[var(--color-cream-dark)] border border-[var(--color-border)] rounded p-4 flex items-center gap-4">
      {/* Rank */}
      <div className="text-2xl font-bold text-[var(--color-text-muted)] font-[family-name:var(--font-playfair)] w-8 text-center shrink-0">
        {index + 1}
      </div>

      {/* Player Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-[family-name:var(--font-playfair)] font-bold text-lg truncate">
            {player.name}
          </span>
          <span className={`position-badge ${POSITION_COLORS[player.position]}`}>
            {player.position}
          </span>
        </div>
        <div className="text-sm text-[var(--color-text-muted)] mt-0.5">
          {player.team} &middot; £{player.price.toFixed(1)}m
        </div>
      </div>

      {/* Predicted Points */}
      <div className="text-right shrink-0">
        <div className="text-2xl font-bold text-[var(--color-accent-green)] font-[family-name:var(--font-playfair)]">
          {player.predicted_points.toFixed(1)}
        </div>
        <div className="text-xs text-[var(--color-text-muted)] uppercase tracking-wider">
          pts
        </div>
      </div>
    </div>
  );
}
