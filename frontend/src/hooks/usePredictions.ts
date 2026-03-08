"use client";

import { useState, useEffect, useMemo } from "react";
import { Player, HealthResponse } from "../lib/types";
import { fetchTopPredictions, fetchHealth } from "../lib/api";

interface UsePredictionsReturn {
  players: Player[];
  filteredPlayers: Player[];
  health: HealthResponse | null;
  loading: boolean;
  error: string | null;
  selectedPosition: string;
  setSelectedPosition: (pos: string) => void;
  selectedTeam: string;
  setSelectedTeam: (team: string) => void;
  maxBudget: number;
  setMaxBudget: (budget: number) => void;
}

export function usePredictions(): UsePredictionsReturn {
  const [players, setPlayers] = useState<Player[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedPosition, setSelectedPosition] = useState("ALL");
  const [selectedTeam, setSelectedTeam] = useState("All Teams");
  const [maxBudget, setMaxBudget] = useState(15);

  useEffect(() => {
    async function load() {
      try {
        const [predictions, healthData] = await Promise.all([
          fetchTopPredictions(),
          fetchHealth(),
        ]);
        setPlayers(predictions.players);
        setHealth(healthData);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load data");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const filteredPlayers = useMemo(() => {
    return players
      .filter((p) => selectedPosition === "ALL" || p.position === selectedPosition)
      .filter((p) => selectedTeam === "All Teams" || p.team === selectedTeam)
      .filter((p) => p.price <= maxBudget);
  }, [players, selectedPosition, selectedTeam, maxBudget]);

  return {
    players,
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
  };
}
