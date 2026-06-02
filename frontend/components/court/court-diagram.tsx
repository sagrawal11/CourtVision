"use client"

import { useState } from "react"
import type { AnalysisShot } from "@/lib/types"

type FilterType = "all" | "serve" | "forehand" | "backhand" | "volley"

const SHOT_COLORS: Record<string, string> = {
  serve: "#a855f7",
  forehand: "#3b82f6",
  backhand: "#f97316",
  volley: "#eab308",
  overhead: "#ec4899",
  default: "#6b7280",
}

interface CourtDiagramProps {
  shots?: AnalysisShot[]
}

const W = 760
const H = 400
const PAD = 20

export function CourtDiagram({ shots = [] }: CourtDiagramProps) {
  const [filter, setFilter] = useState<FilterType>("all")

  const filtered = filter === "all"
    ? shots
    : shots.filter((s) => s.shot_type === filter)

  const dotColor = (shot: AnalysisShot) =>
    SHOT_COLORS[shot.shot_type ?? "default"] ?? SHOT_COLORS.default

  const filters: { key: FilterType; label: string; color: string }[] = [
    { key: "all", label: "All", color: "#9ca3af" },
    { key: "serve", label: "Serve", color: SHOT_COLORS.serve },
    { key: "forehand", label: "Forehand", color: SHOT_COLORS.forehand },
    { key: "backhand", label: "Backhand", color: SHOT_COLORS.backhand },
    { key: "volley", label: "Volley", color: SHOT_COLORS.volley },
  ]

  return (
    <div className="w-full bg-[#1a1a1a] rounded-lg border border-[#333333] p-4">
      {/* Filter pills */}
      <div className="flex gap-2 flex-wrap mb-4">
        {filters.map(({ key, label, color }) => (
          <button
            key={key}
            onClick={() => setFilter(key)}
            className={`px-3 py-1 rounded-full text-xs font-medium border transition-all ${
              filter === key
                ? "border-transparent text-white"
                : "border-[#444] text-gray-400 hover:border-gray-500"
            }`}
            style={filter === key ? { backgroundColor: color } : {}}
          >
            {label}
          </button>
        ))}
        <span className="ml-auto text-xs text-gray-500 self-center">
          {filtered.length} shots
        </span>
      </div>

      <div className="overflow-x-auto">
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="mx-auto block">
          {/* Court surface */}
          <rect x={PAD} y={PAD} width={W - PAD * 2} height={H - PAD * 2} fill="#2d5a27" stroke="#4ade80" strokeWidth="2" />

          {/* Net */}
          <line x1={W / 2} y1={PAD} x2={W / 2} y2={H - PAD} stroke="white" strokeWidth="3" />

          {/* Singles sidelines */}
          {[0.12, 0.88].map((frac) => (
            <line key={frac}
              x1={PAD + frac * (W - PAD * 2)} y1={PAD}
              x2={PAD + frac * (W - PAD * 2)} y2={H - PAD}
              stroke="white" strokeWidth="1" opacity="0.5"
            />
          ))}

          {/* Service lines (y = 0.25 and 0.75 of court height from each baseline) */}
          {[PAD + (H - PAD * 2) * 0.25, PAD + (H - PAD * 2) * 0.75].map((y) => (
            <line key={y}
              x1={PAD} y1={y} x2={W - PAD} y2={y}
              stroke="white" strokeWidth="1" opacity="0.5"
            />
          ))}

          {/* Center service line */}
          <line x1={W / 2} y1={PAD + (H - PAD * 2) * 0.25}
                x2={W / 2} y2={PAD + (H - PAD * 2) * 0.75}
                stroke="white" strokeWidth="1" opacity="0.5" />

          {/* Shot dots */}
          {filtered.map((shot, i) => {
            if (shot.x == null || shot.y == null) return null
            const cx = PAD + shot.x * (W - PAD * 2)
            const cy = PAD + shot.y * (H - PAD * 2)
            return (
              <circle
                key={i}
                cx={cx} cy={cy} r={4}
                fill={dotColor(shot)}
                opacity={0.8}
                className="hover:opacity-100 transition-opacity"
              >
                <title>
                  {shot.shot_type ?? "unknown"}{shot.speed_kmh ? ` · ${shot.speed_kmh.toFixed(0)} km/h` : ""}
                  {shot.player ? ` · ${shot.player}` : ""}
                </title>
              </circle>
            )
          })}
        </svg>
      </div>

      {/* Legend */}
      <div className="flex gap-4 flex-wrap justify-center mt-3 text-xs">
        {Object.entries(SHOT_COLORS)
          .filter(([k]) => k !== "default")
          .map(([type, color]) => (
            <div key={type} className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-gray-400 capitalize">{type}</span>
            </div>
          ))}
      </div>
    </div>
  )
}
