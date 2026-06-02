import type { Match } from "@/lib/types"

interface MatchStatsProps {
  match: Match
}

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-[#141414] rounded-lg border border-[#2a2a2a] p-4">
      <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">{label}</p>
      <p className="text-2xl font-bold text-white">{value}</p>
      {sub && <p className="text-xs text-gray-500 mt-0.5">{sub}</p>}
    </div>
  )
}

function ServeZoneBar({ label, value, total }: { label: string; value: number; total: number }) {
  const pct = total > 0 ? Math.round((value / total) * 100) : 0
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="w-16 text-gray-400 text-xs">{label}</span>
      <div className="flex-1 bg-[#2a2a2a] rounded-full h-1.5">
        <div className="bg-[#50C878] h-1.5 rounded-full" style={{ width: `${pct}%` }} />
      </div>
      <span className="w-8 text-right text-gray-300 text-xs">{value}</span>
    </div>
  )
}

export function MatchStats({ match }: MatchStatsProps) {
  const s = match.stats

  if (!s) {
    return (
      <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 text-center">
        <p className="text-gray-500 text-sm">Statistics will appear once analysis is complete.</p>
      </div>
    )
  }

  const totalServeZones = Object.values(s.serve_zones ?? {}).reduce((a, b) => a + b, 0)

  return (
    <div className="space-y-4">
      {/* Overview row */}
      <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-5 shadow-xl">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Match Overview</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <StatCard label="Total Points" value={s.total_points} />
          <StatCard
            label="Points Won"
            value={s.poi_points_won}
            sub={`of ${s.total_points} (${s.total_points > 0 ? Math.round(s.poi_points_won / s.total_points * 100) : 0}%)`}
          />
          <StatCard label="Total Shots" value={s.poi_shots} />
          <StatCard
            label="Avg Rally"
            value={s.avg_rally_length ? s.avg_rally_length.toFixed(1) : "—"}
            sub="shots"
          />
        </div>
      </div>

      {/* Serve stats + zones */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-5 shadow-xl">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Serve</h3>
          <div className="grid grid-cols-2 gap-3 mb-4">
            <StatCard
              label="1st Serve In"
              value={s.poi_first_serves_in ? `${s.poi_serve_1_pct}%` : "—"}
              sub={`${s.poi_first_serves_in} / ${s.poi_serves_total}`}
            />
            <StatCard label="Aces" value={s.poi_aces} />
            <StatCard label="Double Faults" value={s.poi_double_faults ?? 0} />
            <StatCard
              label="Avg Serve Speed"
              value={s.poi_serve_speed_avg > 0 ? `${s.poi_serve_speed_avg}` : "—"}
              sub={s.poi_serve_speed_avg > 0 ? "km/h" : ""}
            />
          </div>
        </div>

        <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-5 shadow-xl">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Serve Zones</h3>
          <div className="space-y-2">
            <p className="text-xs text-gray-500 mb-2">Deuce court</p>
            <ServeZoneBar label="T" value={s.serve_zones?.deuce_t ?? 0} total={totalServeZones} />
            <ServeZoneBar label="Body" value={s.serve_zones?.deuce_body ?? 0} total={totalServeZones} />
            <ServeZoneBar label="Wide" value={s.serve_zones?.deuce_wide ?? 0} total={totalServeZones} />
            <p className="text-xs text-gray-500 mt-3 mb-2">Ad court</p>
            <ServeZoneBar label="T" value={s.serve_zones?.ad_t ?? 0} total={totalServeZones} />
            <ServeZoneBar label="Body" value={s.serve_zones?.ad_body ?? 0} total={totalServeZones} />
            <ServeZoneBar label="Wide" value={s.serve_zones?.ad_wide ?? 0} total={totalServeZones} />
          </div>
        </div>
      </div>

      {/* Groundstroke stats */}
      <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-5 shadow-xl">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Groundstrokes</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <StatCard label="Forehands" value={s.poi_forehands} />
          <StatCard
            label="Avg FH Speed"
            value={s.poi_forehand_speed_avg > 0 ? `${s.poi_forehand_speed_avg}` : "—"}
            sub={s.poi_forehand_speed_avg > 0 ? "km/h" : ""}
          />
          <StatCard label="Backhands" value={s.poi_backhands} />
          <StatCard
            label="Avg BH Speed"
            value={s.poi_backhand_speed_avg > 0 ? `${s.poi_backhand_speed_avg}` : "—"}
            sub={s.poi_backhand_speed_avg > 0 ? "km/h" : ""}
          />
        </div>
      </div>
    </div>
  )
}
