import Link from "next/link"
import type { Match } from "@/lib/types"

interface MatchCardProps {
  match: Match
}

const statusColors: Record<string, string> = {
  pending: "bg-amber-900/30 text-amber-400 border-amber-800",
  court_setup: "bg-purple-900/30 text-purple-400 border-purple-800",
  processing: "bg-blue-900/30 text-blue-400 border-blue-800",
  completed: "bg-emerald-900/30 text-emerald-400 border-emerald-800",
  failed: "bg-red-900/30 text-red-400 border-red-800",
}

const statusLabels: Record<string, string> = {
  pending: "Pending",
  court_setup: "Court Setup",
  processing: "Processing",
  completed: "Completed",
  failed: "Failed",
}

export function MatchCard({ match }: MatchCardProps) {
  const statusColor = statusColors[match.status] ?? "bg-gray-900/30 text-gray-400 border-gray-700"
  const statusLabel = statusLabels[match.status] ?? match.status

  return (
    <Link href={`/matches/${match.id}`}>
      <div className="bg-[#111111] rounded-xl border border-[#2a2a2a] p-5 hover:border-[#50C878]/50 hover:shadow-lg hover:shadow-[#50C878]/10 transition-all cursor-pointer group">
        <div className="flex justify-between items-start mb-3">
          {/* Player name as title */}
          <h3 className="text-base font-bold text-white group-hover:text-[#50C878] transition-colors truncate">
            {match.player_name || "Unknown Player"}
          </h3>
          <span className={`ml-2 shrink-0 px-2 py-0.5 rounded-full text-xs font-medium border ${statusColor}`}>
            {statusLabel}
          </span>
        </div>

        {match.notes && (
          <p className="text-xs text-gray-500 mt-2 line-clamp-2">{match.notes}</p>
        )}
      </div>
    </Link>
  )
}
