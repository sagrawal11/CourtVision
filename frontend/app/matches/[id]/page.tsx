"use client"

import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import Link from "next/link"
import { MainLayout } from "@/components/layout/main-layout"
import { CourtDiagram } from "@/components/court/court-diagram"
import { ProcessingStatus } from "@/components/upload/processing-status"
import { MatchStats } from "@/components/match/match-stats"
import { createClient } from "@/lib/supabase/client"
import type { Match, AnalysisShot } from "@/lib/types"

export default function MatchDetailPage() {
  const params = useParams()
  const matchId = params.id as string
  const supabase = createClient()

  const [match, setMatch] = useState<Match | null>(null)
  const [loading, setLoading] = useState(true)
  const [pointCount, setPointCount] = useState<number | null>(null)
  const [poiSide, setPoiSide] = useState<"near" | "far" | null>(null)
  const [swappingPoi, setSwappingPoi] = useState(false)
  const [poiSwapMsg, setPoiSwapMsg] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true)
      try {
        const { data: { session } } = await supabase.auth.getSession()
        if (!session) { setLoading(false); return }

        const [matchRes, pointsRes] = await Promise.all([
          fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/matches/${matchId}`, {
            headers: { Authorization: `Bearer ${session.access_token}` },
          }),
          fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/matches/${matchId}/points`, {
            headers: { Authorization: `Bearer ${session.access_token}` },
          }),
        ])

        if (!matchRes.ok) throw new Error("Failed to fetch match")
        const matchData = await matchRes.json()
        setMatch(matchData.match)
        setPoiSide(matchData.match.poi_start_side ?? "near")

        if (pointsRes.ok) {
          const pointsData = await pointsRes.json()
          setPointCount((pointsData.points ?? []).length)
        }
      } catch (err) {
        console.error("Error fetching match:", err)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [matchId, supabase])

  if (loading) {
    return (
      <MainLayout>
        <div className="mx-auto px-4 py-8 max-w-7xl">
          <p className="text-gray-400">Loading match...</p>
        </div>
      </MainLayout>
    )
  }

  if (!match) {
    return (
      <MainLayout>
        <div className="mx-auto px-4 py-8 max-w-7xl">
          <p className="text-red-400">Match not found</p>
        </div>
      </MainLayout>
    )
  }

  const swapPoi = async () => {
    if (swappingPoi) return
    setSwappingPoi(true)
    setPoiSwapMsg(null)
    try {
      const { data: { session } } = await supabase.auth.getSession()
      if (!session) return
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/matches/${matchId}/poi`,
        { method: "PATCH", headers: { Authorization: `Bearer ${session.access_token}` } }
      )
      if (res.ok) {
        const data = await res.json()
        setPoiSide(data.poi_start_side)
        setPoiSwapMsg("Player swapped. Re-analyze to update stats.")
        setTimeout(() => setPoiSwapMsg(null), 4000)
      }
    } finally {
      setSwappingPoi(false)
    }
  }

  const analysisShots: AnalysisShot[] = match.analysis_shots ?? []
  const dateLabel = match.match_date
    ? new Date(match.match_date + "T00:00:00").toLocaleDateString("en-US", { weekday: "short", year: "numeric", month: "short", day: "numeric" })
    : new Date(match.created_at).toLocaleDateString("en-US", { weekday: "short", year: "numeric", month: "short", day: "numeric" })

  return (
    <MainLayout>
      <div className="mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-2">
          <div>
            <h1 className="text-2xl font-bold text-white">
              {match.player_name || "Match"}
              {match.opponent ? <span className="text-gray-400 font-normal"> vs {match.opponent}</span> : null}
            </h1>
            <p className="text-gray-500 text-sm mt-0.5">{dateLabel}</p>
          </div>
          {match.status === "completed" && (
            <div className="flex items-center gap-2 self-start sm:self-auto flex-wrap">
              {/* POI indicator */}
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#1a1a1a] border border-[#333] text-xs text-gray-400">
                <span>Tracking:</span>
                <span className="text-white font-medium">{poiSide === "far" ? "Far player" : "Near player"}</span>
                <button
                  onClick={swapPoi}
                  disabled={swappingPoi}
                  className="ml-1 text-[#50C878] hover:text-[#3da860] font-medium disabled:opacity-50"
                  title="Swap tracked player"
                >
                  {swappingPoi ? "..." : "Swap"}
                </button>
              </div>
              {pointCount !== null && pointCount > 0 && (
                <Link
                  href={`/matches/${matchId}/review`}
                  className="px-4 py-2 rounded-lg bg-[#50C878] text-black text-sm font-semibold hover:bg-[#3da860] transition-colors"
                >
                  Review Points ({pointCount})
                </Link>
              )}
            </div>
          )}
        </div>

        {/* POI swap confirmation */}
        {poiSwapMsg && (
          <div className="mt-2 px-3 py-2 rounded bg-[#1a1a1a] border border-[#50C878]/30 text-xs text-[#50C878]">
            {poiSwapMsg}
          </div>
        )}

        {/* Processing banner */}
        {match.status !== "completed" && (
          <div className="mt-4 mb-6">
            <ProcessingStatus matchId={match.id} />
          </div>
        )}

        {match.status === "completed" ? (
          <>
            {/* Shot map */}
            <div className="mt-6">
              <h2 className="text-base font-semibold text-gray-300 mb-3">Shot Map</h2>
              {analysisShots.length > 0 ? (
                <CourtDiagram shots={analysisShots} />
              ) : (
                <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-10 text-center">
                  <p className="text-gray-500 text-sm">No shot data available.</p>
                </div>
              )}
            </div>

            {/* Stats panel */}
            <div className="mt-6">
              <h2 className="text-base font-semibold text-gray-300 mb-3">Match Statistics</h2>
              <MatchStats match={match} />
            </div>
          </>
        ) : (
          <div className="mt-6 bg-[#1a1a1a] rounded-lg border border-[#333333] p-12 text-center">
            <p className="text-gray-400">Analysis in progress. Results will appear when complete.</p>
          </div>
        )}
      </div>
    </MainLayout>
  )
}
