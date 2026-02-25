"use client"

import { useEffect, useState } from "react"
import { createClient } from "@/lib/supabase/client"
import type { Match } from "@/lib/types"

interface ProcessingStatusProps {
  matchId: string
}

const statusConfig = {
  pending: {
    bg: "bg-amber-900/30",
    text: "text-amber-400",
    border: "border-amber-800",
    label: "Pending",
    message: "Waiting to start processing...",
  },
  generating_frames: {
    bg: "bg-amber-900/30",
    text: "text-amber-400",
    border: "border-amber-800",
    label: "Generating Frames",
    message: "Extracting video frames for player selection...",
  },
  player_selection: {
    bg: "bg-purple-900/30",
    text: "text-purple-400",
    border: "border-purple-800",
    label: "Needs Player Selection",
    message: "Waiting for you to select the target player.",
  },
  court_setup: {
    bg: "bg-purple-900/30",
    text: "text-purple-400",
    border: "border-purple-800",
    label: "Needs Court Setup",
    message: "Waiting for you to confirm the court dimensions.",
  },
  processing: {
    bg: "bg-blue-900/30",
    text: "text-blue-400",
    border: "border-blue-800",
    label: "Processing",
    message: "Analyzing video... This may take up to an hour.",
  },
  completed: {
    bg: "bg-emerald-900/30",
    text: "text-emerald-400",
    border: "border-emerald-800",
    label: "Completed",
    message: "Processing complete! Your match is ready.",
  },
  failed: {
    bg: "bg-red-900/30",
    text: "text-red-400",
    border: "border-red-800",
    label: "Failed",
    message: "Processing failed. Please try again.",
  },
}

export function ProcessingStatus({ matchId }: ProcessingStatusProps) {
  const supabase = createClient()
  const [match, setMatch] = useState<Match | null>(null)

  useEffect(() => {
    const fetchMatch = async () => {
      const { data: { session } } = await supabase.auth.getSession()
      if (!session) return

      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/matches/${matchId}`, {
          headers: {
            'Authorization': `Bearer ${session.access_token}`,
          },
        })

        if (response.ok) {
          const data = await response.json()
          setMatch(data.match)
        }
      } catch (err) {
        console.error('Error fetching match:', err)
      }
    }

    fetchMatch()

    // Poll every 5 seconds if processing
    const interval = setInterval(() => {
      if (match?.status === "processing" || match?.status === "pending") {
        fetchMatch()
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [matchId, match?.status, supabase])

  if (!match) return null

  const config = statusConfig[match.status]

  return (
    <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl flex items-center gap-4">
      <div className={`px-4 py-2 rounded-lg ${config.bg} ${config.text} ${config.border} border font-semibold`}>
        {config.label}
      </div>
      <div className="flex-1">
        <p className="text-sm text-gray-400">{config.message}</p>
        {match.processed_at && (
          <p className="text-xs text-gray-500 mt-1">Completed at {new Date(match.processed_at).toLocaleString()}</p>
        )}
      </div>
      {match.status === "processing" && (
        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600" />
      )}
    </div>
  )
}
