"use client"

import type { MatchStats } from "@/lib/types"

interface ServePlacementHeatmapProps {
  stats?: MatchStats | null
}

const W = 760
const H = 240
const PAD = 20

// Six lanes laid out left→right across the two service boxes. The Deuce box
// runs Wide→Body→T toward the center line; the Ad box mirrors it (T→Body→Wide),
// so the two "T" lanes sit either side of the center service line.
const LANES: { key: keyof MatchStats["serve_zones"]; label: string }[] = [
  { key: "deuce_wide", label: "Wide" },
  { key: "deuce_body", label: "Body" },
  { key: "deuce_t", label: "T" },
  { key: "ad_t", label: "T" },
  { key: "ad_body", label: "Body" },
  { key: "ad_wide", label: "Wide" },
]

export function ServePlacementHeatmap({ stats }: ServePlacementHeatmapProps) {
  const zones = stats?.serve_zones
  const total = zones ? Object.values(zones).reduce((a, b) => a + b, 0) : 0

  if (!zones || total === 0) {
    return (
      <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-10 text-center">
        <p className="text-gray-500 text-sm">No serve placement data available.</p>
      </div>
    )
  }

  const max = Math.max(...LANES.map((l) => zones[l.key]))
  const cellW = (W - PAD * 2) / LANES.length
  const boxH = H - PAD * 2

  return (
    <div className="w-full bg-[#1a1a1a] rounded-lg border border-[#333333] p-4">
      <div className="flex items-center mb-3">
        <span className="text-xs text-gray-400">Serve landing placement across the service boxes</span>
        <span className="ml-auto text-xs text-gray-500">{total} serves</span>
      </div>

      <div className="overflow-x-auto">
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="mx-auto block">
          {LANES.map((lane, i) => {
            const count = zones[lane.key]
            const x = PAD + i * cellW
            const intensity = max > 0 ? count / max : 0
            const opacity = count === 0 ? 0.06 : 0.15 + 0.8 * intensity
            return (
              <g key={`${lane.key}`}>
                <rect
                  x={x} y={PAD} width={cellW} height={boxH}
                  fill="#50C878" fillOpacity={opacity}
                  stroke="#333333" strokeWidth="1"
                />
                <text
                  x={x + cellW / 2} y={PAD + boxH / 2 - 4}
                  textAnchor="middle" fill="#ffffff" fontSize="22" fontWeight="700"
                >
                  {count}
                </text>
                <text
                  x={x + cellW / 2} y={PAD + boxH / 2 + 18}
                  textAnchor="middle" fill="#d1d5db" fontSize="11"
                >
                  {lane.label}
                </text>
              </g>
            )
          })}

          {/* Center service line dividing the Deuce and Ad boxes */}
          <line x1={W / 2} y1={PAD} x2={W / 2} y2={H - PAD} stroke="#ffffff" strokeWidth="3" />

          {/* Outer border */}
          <rect x={PAD} y={PAD} width={W - PAD * 2} height={boxH} fill="none" stroke="#4ade80" strokeWidth="2" />

          {/* Box labels */}
          <text x={PAD + (W - PAD * 2) / 4} y={PAD - 6} textAnchor="middle" fill="#9ca3af" fontSize="11">Deuce</text>
          <text x={PAD + (3 * (W - PAD * 2)) / 4} y={PAD - 6} textAnchor="middle" fill="#9ca3af" fontSize="11">Ad</text>
        </svg>
      </div>
    </div>
  )
}
