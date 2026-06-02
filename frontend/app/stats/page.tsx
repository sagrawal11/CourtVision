"use client"

import { useState, useEffect } from "react"
import { MainLayout } from "@/components/layout/main-layout"
import { ActivationKeyInput } from "@/components/activation/activation-key-input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { useProfile } from "@/hooks/useProfile"
import { useTeams } from "@/hooks/useTeams"
import { useActivation } from "@/hooks/useActivation"
import { createClient } from "@/lib/supabase/client"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from "recharts"

const TOOLTIP_STYLE = { backgroundColor: "#1a1a1a", border: "1px solid #333", borderRadius: "8px" }

type SeasonStats = {
  total_matches: number
  season_totals: {
    total_points: number
    poi_points_won: number
    poi_shots: number
    poi_forehands: number
    poi_backhands: number
    poi_aces: number
  }
  avg_rally_length: number
  avg_serve_speed_kmh: number
  avg_forehand_speed_kmh: number
  avg_backhand_speed_kmh: number
  per_match: Array<{
    match_id: string
    match_date: string | null
    opponent: string | null
    total_points: number
    poi_points_won: number
    poi_winners: number
    poi_errors: number
    avg_rally_length: number
    poi_serve_1_pct: number
  }>
}

function SummaryCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl hover:border-[#50C878]/30 transition-colors">
      <p className="text-sm font-medium text-gray-400 mb-1">{label}</p>
      <p className="text-3xl font-bold text-white">{value}</p>
      {sub && <p className="text-xs text-gray-500 mt-1">{sub}</p>}
    </div>
  )
}

export default function StatsPage() {
  const { profile } = useProfile()
  const { teams } = useTeams()
  const { isActivated } = useActivation()
  const supabase = createClient()
  const [selectedPlayerId, setSelectedPlayerId] = useState<string>("")
  const [teamMembers, setTeamMembers] = useState<any[]>([])
  const [season, setSeason] = useState<SeasonStats | null>(null)
  const [loading, setLoading] = useState(false)

  const isCoach = profile?.role === "coach"
  const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

  // Fetch team members for coach player selector
  useEffect(() => {
    if (!isCoach || teams.length === 0) return
    const load = async () => {
      const { data: { session } } = await supabase.auth.getSession()
      if (!session) return
      const allMembers: any[] = []
      for (const team of teams) {
        try {
          const res = await fetch(`${API}/api/teams/${team.id}/members`, {
            headers: { Authorization: `Bearer ${session.access_token}` },
          })
          if (res.ok) {
            const d = await res.json()
            allMembers.push(...(d.members || [])
              .filter((m: any) => m.users?.role === "player")
              .map((m: any) => ({ id: m.users?.id, name: m.users?.name || m.users?.email || "Unknown" }))
            )
          }
        } catch {}
      }
      const unique = Array.from(new Map(allMembers.map((m: any) => [m.id, m])).values())
      setTeamMembers(unique)
      if (unique.length > 0 && !selectedPlayerId) setSelectedPlayerId(unique[0].id)
    }
    load()
  }, [isCoach, teams]) // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch season stats whenever the target player changes
  useEffect(() => {
    const load = async () => {
      const { data: { session } } = await supabase.auth.getSession()
      if (!session) return
      setLoading(true)
      try {
        const targetId = isCoach && selectedPlayerId ? selectedPlayerId : "me"
        const url = targetId === "me"
          ? `${API}/api/stats/my-stats`
          : `${API}/api/stats/player/${targetId}`
        const res = await fetch(url, { headers: { Authorization: `Bearer ${session.access_token}` } })
        if (res.ok) setSeason(await res.json())
      } catch {}
      setLoading(false)
    }
    if (profile) load()
  }, [profile, selectedPlayerId]) // eslint-disable-line react-hooks/exhaustive-deps

  const t = season?.season_totals
  const pointWinPct = t && t.total_points > 0
    ? Math.round(t.poi_points_won / t.total_points * 100)
    : 0

  const matchPerformance = (season?.per_match ?? []).slice(0, 8).map((m, i) => ({
    name: m.match_date ? new Date(m.match_date + "T00:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric" }) : `M${i + 1}`,
    points_won: m.poi_points_won,
    total: m.total_points,
  }))

  const rallyTrend = (season?.per_match ?? []).slice(0, 8).map((m, i) => ({
    name: m.match_date ? new Date(m.match_date + "T00:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric" }) : `M${i + 1}`,
    rally: m.avg_rally_length,
  }))

  return (
    <MainLayout>
      <div className="mx-auto px-4 py-8 max-w-7xl">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-white">Statistics</h1>
          {isCoach && isActivated && teamMembers.length > 0 && (
            <div className="flex items-center gap-2">
              <Label className="text-gray-400 text-sm">Player:</Label>
              <Select value={selectedPlayerId} onValueChange={setSelectedPlayerId}>
                <SelectTrigger className="w-48 bg-[#1a1a1a] border-[#333333] text-white">
                  <SelectValue placeholder="Select a player" />
                </SelectTrigger>
                <SelectContent className="bg-[#1a1a1a] border-[#333333]">
                  {teamMembers.map((m: any) => (
                    <SelectItem key={m.id} value={m.id} className="text-white hover:bg-[#262626]">{m.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>

        {isCoach && !isActivated && (
          <div className="mb-8"><ActivationKeyInput /></div>
        )}

        {isCoach && !isActivated ? (
          <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-12 text-center shadow-xl opacity-50 pointer-events-none">
            <h2 className="text-xl font-semibold text-white mb-2">Account Activation Required</h2>
            <p className="text-gray-400">Please enter your activation key above to unlock all features.</p>
          </div>
        ) : loading ? (
          <p className="text-gray-400">Loading statistics...</p>
        ) : !season || season.total_matches === 0 ? (
          <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-12 text-center shadow-xl">
            <p className="text-gray-400">No statistics yet. Upload and complete matches to see your analytics.</p>
          </div>
        ) : (
          <>
            {/* Season summary cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <SummaryCard label="Matches Analyzed" value={season.total_matches} />
              <SummaryCard
                label="Point Win %"
                value={`${pointWinPct}%`}
                sub={`${t?.poi_points_won ?? 0} of ${t?.total_points ?? 0} points`}
              />
              <SummaryCard
                label="Avg Rally Length"
                value={season.avg_rally_length ? season.avg_rally_length.toFixed(1) : "—"}
                sub="shots per point"
              />
              <SummaryCard
                label="Avg Serve Speed"
                value={season.avg_serve_speed_kmh > 0 ? `${season.avg_serve_speed_kmh}` : "—"}
                sub={season.avg_serve_speed_kmh > 0 ? "km/h" : ""}
              />
            </div>

            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
              <SummaryCard label="Total Shots" value={t?.poi_shots ?? 0} />
              <SummaryCard label="Forehands" value={t?.poi_forehands ?? 0} sub={season.avg_forehand_speed_kmh > 0 ? `avg ${season.avg_forehand_speed_kmh} km/h` : ""} />
              <SummaryCard label="Backhands" value={t?.poi_backhands ?? 0} sub={season.avg_backhand_speed_kmh > 0 ? `avg ${season.avg_backhand_speed_kmh} km/h` : ""} />
              <SummaryCard label="Aces" value={t?.poi_aces ?? 0} />
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl hover:border-[#50C878]/30 transition-colors">
                <h3 className="text-lg font-semibold text-white mb-4">Points Won per Match</h3>
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={matchPerformance}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="name" stroke="#a0a0a0" fontSize={11} />
                    <YAxis stroke="#a0a0a0" fontSize={11} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Bar dataKey="points_won" fill="#50C878" name="Points Won" />
                    <Bar dataKey="total" fill="#374151" name="Total Points" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl hover:border-[#50C878]/30 transition-colors">
                <h3 className="text-lg font-semibold text-white mb-4">Avg Rally Length Over Time</h3>
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={rallyTrend}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="name" stroke="#a0a0a0" fontSize={11} />
                    <YAxis stroke="#a0a0a0" fontSize={11} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Line type="monotone" dataKey="rally" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Avg Rally" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}
      </div>
    </MainLayout>
  )
}
