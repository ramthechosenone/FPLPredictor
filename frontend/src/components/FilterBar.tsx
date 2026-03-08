import { POSITIONS, TEAMS } from "../lib/constants";

interface FilterBarProps {
  selectedPosition: string;
  setSelectedPosition: (pos: string) => void;
  selectedTeam: string;
  setSelectedTeam: (team: string) => void;
  maxBudget: number;
  setMaxBudget: (budget: number) => void;
  resultCount: number;
}

export default function FilterBar({
  selectedPosition,
  setSelectedPosition,
  selectedTeam,
  setSelectedTeam,
  maxBudget,
  setMaxBudget,
  resultCount,
}: FilterBarProps) {
  return (
    <div className="bg-[var(--color-cream-dark)] border border-[var(--color-border)] rounded p-4 mb-6">
      <div className="flex flex-col md:flex-row md:items-end gap-4">
        {/* Position Filter */}
        <div className="flex-1">
          <label className="table-header block mb-1.5">Position</label>
          <div className="flex flex-wrap gap-1.5">
            {POSITIONS.map((pos) => (
              <button
                key={pos}
                onClick={() => setSelectedPosition(pos)}
                className={`px-3 py-1.5 rounded text-sm font-medium border transition-colors cursor-pointer ${
                  selectedPosition === pos
                    ? "bg-[var(--color-accent-maroon)] text-[var(--color-cream)] border-[var(--color-accent-maroon)]"
                    : "bg-[var(--color-cream)] text-[var(--color-text-primary)] border-[var(--color-border)] hover:border-[var(--color-accent-maroon)]"
                }`}
              >
                {pos === "ALL" ? "All" : pos}
              </button>
            ))}
          </div>
        </div>

        {/* Team Filter */}
        <div className="flex-1">
          <label className="table-header block mb-1.5">Team</label>
          <select
            value={selectedTeam}
            onChange={(e) => setSelectedTeam(e.target.value)}
            className="w-full px-3 py-1.5 rounded border border-[var(--color-border)] bg-[var(--color-cream)] text-[var(--color-text-primary)] text-sm font-[family-name:var(--font-lora)]"
          >
            {TEAMS.map((team) => (
              <option key={team} value={team}>
                {team}
              </option>
            ))}
          </select>
        </div>

        {/* Budget Filter */}
        <div className="flex-1">
          <label className="table-header block mb-1.5">
            Max Price: £{maxBudget.toFixed(1)}m
          </label>
          <input
            type="range"
            min={3.5}
            max={15}
            step={0.1}
            value={maxBudget}
            onChange={(e) => setMaxBudget(parseFloat(e.target.value))}
            className="w-full accent-[var(--color-accent-maroon)]"
          />
          <div className="flex justify-between text-xs text-[var(--color-text-muted)] mt-0.5">
            <span>£3.5m</span>
            <span>£15.0m</span>
          </div>
        </div>
      </div>

      <div className="mt-3 text-sm text-[var(--color-text-muted)] font-[family-name:var(--font-lora)] italic">
        Showing {resultCount} player{resultCount !== 1 ? "s" : ""}
      </div>
    </div>
  );
}
