"use client";

import Header from "../components/Header";
import FilterBar from "../components/FilterBar";
import PlayerTable from "../components/PlayerTable";
import PlayerCard from "../components/PlayerCard";
import Footer from "../components/Footer";
import { usePredictions } from "../hooks/usePredictions";

export default function Home() {
  const {
    filteredPlayers,
    health,
    loading,
    error,
    selectedPosition,
    setSelectedPosition,
    selectedTeam,
    setSelectedTeam,
    maxBudget,
    setMaxBudget,
  } = usePredictions();

  return (
    <div className="max-w-5xl mx-auto px-4">
      <Header />

      {loading && (
        <div className="text-center py-20">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-2 border-[var(--color-accent-maroon)] border-t-transparent" />
          <p className="mt-4 text-[var(--color-text-muted)] font-[family-name:var(--font-lora)] italic">
            Loading predictions...
          </p>
        </div>
      )}

      {error && (
        <div className="text-center py-20">
          <p className="text-[var(--color-accent-maroon)] font-[family-name:var(--font-playfair)] text-xl">
            Failed to load predictions
          </p>
          <p className="mt-2 text-[var(--color-text-muted)] text-sm">{error}</p>
        </div>
      )}

      {!loading && !error && (
        <>
          <FilterBar
            selectedPosition={selectedPosition}
            setSelectedPosition={setSelectedPosition}
            selectedTeam={selectedTeam}
            setSelectedTeam={setSelectedTeam}
            maxBudget={maxBudget}
            setMaxBudget={setMaxBudget}
            resultCount={filteredPlayers.length}
          />

          {/* Desktop: Table */}
          <div className="hidden md:block">
            <PlayerTable players={filteredPlayers} />
          </div>

          {/* Mobile: Cards */}
          <div className="md:hidden space-y-3">
            {filteredPlayers.map((player, i) => (
              <PlayerCard key={player.player_id} player={player} index={i} />
            ))}
          </div>

          {filteredPlayers.length === 0 && (
            <div className="text-center py-12">
              <p className="text-[var(--color-text-muted)] font-[family-name:var(--font-lora)] italic text-lg">
                No players match your filters. Try adjusting the budget or position.
              </p>
            </div>
          )}
        </>
      )}

      <Footer health={health} />
    </div>
  );
}
