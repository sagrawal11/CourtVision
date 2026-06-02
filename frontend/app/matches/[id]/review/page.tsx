"use client"

import { useEffect, useState, useCallback } from "react"
import { useParams } from "next/navigation"
import Link from "next/link"
import { MainLayout } from "@/components/layout/main-layout"
import { createClient } from "@/lib/supabase/client"
import type { PointRecord } from "@/lib/types"

const OUTCOMES = [
  { key: "winner", label: "Winner", shortcut: "W", color: "#22c55e", description: "Clean shot the opponent couldn't reach" },
  { key: "forced_error", label: "Forced Error", shortcut: "F", color: "#f97316", description: "Opponent's error caused by your good shot" },
  { key: "unforced_error", label: "Unforced Error", shortcut: "U", color: "#ef4444", description: "Opponent's error without pressure" },
  { key: "ace", label: "Ace", shortcut: "A", color: "#a855f7", description: "Unreturned serve" },
  { key: "double_fault", label: "Double Fault", shortcut: "D", color: "#6b7280", description: "Two consecutive faults" },
] as const

type OutcomeKey = typeof OUTCOMES[number]["key"]

function formatTimestamp(seconds: number | null): string {
  if (seconds == null) return "—"
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}

export default function PointReviewPage() {
  const params = useParams()
  const matchId = params.id as string
  const supabase = createClient()

  const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

  const [points, setPoints] = useState<PointRecord[]>([])
  const [currentIdx, setCurrentIdx] = useState(0)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [jumpToUnreviewed, setJumpToUnreviewed] = useState(true)

  useEffect(() => {
    const load = async () => {
      const { data: { session } } = await supabase.auth.getSession()
      if (!session) return
      setLoading(true)
      try {
        const res = await fetch(`${API}/api/matches/${matchId}/points`, {
          headers: { Authorization: `Bearer ${session.access_token}` },
        })
        if (res.ok) {
          const data = await res.json()
          const pts: PointRecord[] = data.points ?? []
          setPoints(pts)
          // Start at first unreviewed point
          if (jumpToUnreviewed) {
            const firstUnreviewed = pts.findIndex((p) => p.manual_outcome == null)
            if (firstUnreviewed >= 0) setCurrentIdx(firstUnreviewed)
            setJumpToUnreviewed(false)
          }
        }
      } catch {}
      setLoading(false)
    }
    load()
  }, [matchId]) // eslint-disable-line react-hooks/exhaustive-deps

  const classify = useCallback(async (outcome: OutcomeKey) => {
    if (saving || points.length === 0) return
    const point = points[currentIdx]
    setSaving(true)

    const { data: { session } } = await supabase.auth.getSession()
    if (!session) { setSaving(false); return }

    try {
      const res = await fetch(`${API}/api/matches/${matchId}/points/${point.point_idx}`, {
        method: "PATCH",
        headers: {
          Authorization: `Bearer ${session.access_token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ manual_outcome: outcome }),
      })
      if (res.ok) {
        const updated = await res.json()
        setPoints((prev) => prev.map((p) => p.point_idx === point.point_idx ? updated.point : p))
        // Auto-advance to next unreviewed
        const nextUnreviewed = points.findIndex((p, i) => i > currentIdx && p.manual_outcome == null)
        if (nextUnreviewed >= 0) {
          setCurrentIdx(nextUnreviewed)
        } else if (currentIdx < points.length - 1) {
          setCurrentIdx((i) => i + 1)
        }
      }
    } catch {}
    setSaving(false)
  }, [saving, points, currentIdx, matchId, supabase, API])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      const key = e.key.toUpperCase()
      const outcome = OUTCOMES.find((o) => o.shortcut === key)
      if (outcome) { classify(outcome.key); return }
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        setCurrentIdx((i) => Math.min(i + 1, points.length - 1))
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        setCurrentIdx((i) => Math.max(i - 1, 0))
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [classify, points.length])

  const reviewed = points.filter((p) => p.manual_outcome != null).length
  const current = points[currentIdx]

  if (loading) {
    return (
      <MainLayout>
        <div className="mx-auto px-4 py-8 max-w-4xl">
          <p className="text-gray-400">Loading points...</p>
        </div>
      </MainLayout>
    )
  }

  if (points.length === 0) {
    return (
      <MainLayout>
        <div className="mx-auto px-4 py-8 max-w-4xl">
          <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-12 text-center">
            <p className="text-gray-400 mb-4">No points detected for this match yet.</p>
            <Link href={`/matches/${matchId}`} className="text-[#50C878] hover:underline text-sm">
              ← Back to match
            </Link>
          </div>
        </div>
      </MainLayout>
    )
  }

  return (
    <MainLayout>
      <div className="mx-auto px-4 py-8 max-w-4xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <Link href={`/matches/${matchId}`} className="text-gray-500 hover:text-gray-300 text-sm">
              ← Back to match
            </Link>
            <h1 className="text-2xl font-bold text-white mt-1">Review Points</h1>
          </div>
          <div className="text-right">
            <p className="text-2xl font-bold text-white">{reviewed} <span className="text-gray-500 text-lg font-normal">/ {points.length}</span></p>
            <p className="text-xs text-gray-500">points reviewed</p>
          </div>
        </div>

        {/* Progress bar */}
        <div className="w-full bg-[#2a2a2a] rounded-full h-1.5 mb-6">
          <div
            className="bg-[#50C878] h-1.5 rounded-full transition-all"
            style={{ width: `${(reviewed / points.length) * 100}%` }}
          />
        </div>

        {/* Point card */}
        {current && (
          <div className="bg-[#1a1a1a] rounded-xl border border-[#333333] p-6 mb-4">
            {/* Point header */}
            <div className="flex items-start justify-between mb-5">
              <div>
                <p className="text-xs text-gray-500 uppercase tracking-wider">Point {current.point_idx + 1}</p>
                <div className="flex items-center gap-3 mt-1">
                  <span className="text-white font-medium">
                    {formatTimestamp(current.start_timestamp_s)} – {formatTimestamp(current.end_timestamp_s)}
                  </span>
                  {current.serve_player && (
                    <span className="px-2 py-0.5 rounded-full bg-[#2a2a2a] text-xs text-gray-400">
                      {current.serve_player} serves
                    </span>
                  )}
                  {current.rally_length > 0 && (
                    <span className="px-2 py-0.5 rounded-full bg-[#2a2a2a] text-xs text-gray-400">
                      {current.rally_length} shot{current.rally_length !== 1 ? "s" : ""}
                    </span>
                  )}
                </div>
              </div>
              {current.manual_outcome && (
                <span
                  className="px-3 py-1 rounded-full text-xs font-medium text-white"
                  style={{ backgroundColor: OUTCOMES.find((o) => o.key === current.manual_outcome)?.color ?? "#333" }}
                >
                  {OUTCOMES.find((o) => o.key === current.manual_outcome)?.label ?? current.manual_outcome}
                </span>
              )}
            </div>

            {/* Classification buttons */}
            <div className="grid grid-cols-1 sm:grid-cols-5 gap-2">
              {OUTCOMES.map((outcome) => {
                const isSelected = current.manual_outcome === outcome.key
                return (
                  <button
                    key={outcome.key}
                    onClick={() => classify(outcome.key)}
                    disabled={saving}
                    className={`relative flex flex-col items-center gap-1 px-3 py-4 rounded-lg border text-sm font-medium transition-all ${
                      isSelected
                        ? "text-white border-transparent"
                        : "border-[#333] text-gray-400 hover:border-gray-500 hover:text-white"
                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                    style={isSelected ? { backgroundColor: outcome.color, borderColor: outcome.color } : {}}
                    title={outcome.description}
                  >
                    <span className="absolute top-1.5 right-2 text-[10px] opacity-50 font-mono">{outcome.shortcut}</span>
                    <span>{outcome.label}</span>
                  </button>
                )
              })}
            </div>

            <p className="text-xs text-gray-600 mt-3 text-center">
              Keyboard: W / F / U / A / D to classify · ← → to navigate
            </p>
          </div>
        )}

        {/* Point navigator */}
        <div className="flex items-center justify-between gap-3">
          <button
            onClick={() => setCurrentIdx((i) => Math.max(0, i - 1))}
            disabled={currentIdx === 0}
            className="px-4 py-2 rounded-lg bg-[#1a1a1a] border border-[#333] text-gray-400 text-sm hover:border-gray-500 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            ← Prev
          </button>

          {/* Mini point strip */}
          <div className="flex gap-1 flex-wrap justify-center flex-1 max-h-20 overflow-y-auto">
            {points.map((p, i) => {
              const outcome = OUTCOMES.find((o) => o.key === p.manual_outcome)
              return (
                <button
                  key={p.point_idx}
                  onClick={() => setCurrentIdx(i)}
                  className={`w-6 h-6 rounded text-xs font-bold transition-all ${
                    i === currentIdx ? "ring-2 ring-white ring-offset-1 ring-offset-[#111]" : ""
                  }`}
                  style={{
                    backgroundColor: outcome ? outcome.color : "#2a2a2a",
                    color: outcome ? "white" : "#666",
                  }}
                  title={`Point ${p.point_idx + 1}${outcome ? `: ${outcome.label}` : " (unreviewed)"}`}
                >
                  {p.point_idx + 1}
                </button>
              )
            })}
          </div>

          <button
            onClick={() => setCurrentIdx((i) => Math.min(points.length - 1, i + 1))}
            disabled={currentIdx === points.length - 1}
            className="px-4 py-2 rounded-lg bg-[#1a1a1a] border border-[#333] text-gray-400 text-sm hover:border-gray-500 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Next →
          </button>
        </div>

        {reviewed === points.length && (
          <div className="mt-6 bg-[#0f2a1a] border border-[#50C878]/30 rounded-lg p-4 text-center">
            <p className="text-[#50C878] font-medium">All points reviewed!</p>
            <Link href={`/matches/${matchId}`} className="text-sm text-gray-400 hover:text-white mt-1 inline-block">
              View match results →
            </Link>
          </div>
        )}
      </div>
    </MainLayout>
  )
}
