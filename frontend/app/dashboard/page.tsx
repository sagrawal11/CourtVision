"use client"

import { useState, useEffect } from "react"
import { MainLayout } from "@/components/layout/main-layout"
import { MatchCard } from "@/components/dashboard/match-card"
import { ActivationKeyInput } from "@/components/activation/activation-key-input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { useMatches } from "@/hooks/useMatches"
import { useProfile } from "@/hooks/useProfile"
import { useTeams } from "@/hooks/useTeams"
import { useActivation } from "@/hooks/useActivation"
import { createClient } from "@/lib/supabase/client"
import type { Match } from "@/lib/types"

export default function DashboardPage() {
  const { data: matches, isLoading, error } = useMatches()
  const { profile } = useProfile()
  const { teams } = useTeams()
  const { isActivated } = useActivation()
  const supabase = createClient()
  const [expandedDate, setExpandedDate] = useState<string | null>(null)
  const [selectedPlayerId, setSelectedPlayerId] = useState<string>("all")
  const [teamMembers, setTeamMembers] = useState<any[]>([])

  const isCoach = profile?.role === "coach"

  // Fetch team members for coach
  useEffect(() => {
    const fetchTeamMembers = async () => {
      if (!isCoach || teams.length === 0) {
        setTeamMembers([])
        return
      }

      const allMembers: any[] = []
      for (const team of teams) {
        const { data: { session } } = await supabase.auth.getSession()
        if (!session) continue

        try {
          const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/teams/${team.id}/members`, {
            headers: {
              'Authorization': `Bearer ${session.access_token}`,
            },
          })

          if (response.ok) {
            const data = await response.json()
            const members = (data.members || []).filter((m: any) => m.users?.role === 'player')
            allMembers.push(...members.map((m: any) => ({
              id: m.users?.id,
              name: m.users?.name || m.users?.email || 'Unknown',
              email: m.users?.email,
            })))
          }
        } catch (err) {
          console.error('Error fetching team members:', err)
        }
      }

      // Remove duplicates
      const uniqueMembers = Array.from(
        new Map(allMembers.map((m: any) => [m.id, m])).values()
      )
      setTeamMembers(uniqueMembers)
    }

    fetchTeamMembers()
  }, [isCoach, teams, supabase])

  // Filter matches by selected player
  const filteredMatches = selectedPlayerId === "all" 
    ? matches 
    : matches?.filter((m: Match) => m.user_id === selectedPlayerId) || []

  // Group matches by date
  const groupedMatches = filteredMatches?.reduce(
    (groups: Record<string, Match[]>, match: Match) => {
      const date = new Date(match.created_at).toLocaleDateString()
      if (!groups[date]) {
        groups[date] = []
      }
      groups[date].push(match)
      return groups
    },
    {} as Record<string, Match[]>,
  )

  const sortedDates = groupedMatches
    ? Object.keys(groupedMatches).sort((a, b) => new Date(b).getTime() - new Date(a).getTime())
    : []

  return (
    <MainLayout>
      <div className="mx-auto px-4 py-8 max-w-7xl">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-white">Dashboard</h1>
          {isCoach && isActivated && teamMembers.length > 0 && (
            <div className="flex items-center gap-2">
              <Label htmlFor="playerFilter" className="text-gray-400 text-sm">
                Filter by Player:
              </Label>
              <Select value={selectedPlayerId} onValueChange={setSelectedPlayerId}>
                <SelectTrigger className="w-48 bg-[#1a1a1a] border-[#333333] text-white">
                  <SelectValue placeholder="All Players" />
                </SelectTrigger>
                <SelectContent className="bg-[#1a1a1a] border-[#333333]">
                  <SelectItem value="all" className="text-white hover:bg-[#262626]">
                    All Players
                  </SelectItem>
                  {teamMembers.map((member: any) => (
                    <SelectItem key={member.id} value={member.id} className="text-white hover:bg-[#262626]">
                      {member.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>

        {/* Activation Key Input for Coaches */}
        {isCoach && !isActivated && (
          <div className="mb-8">
            <ActivationKeyInput />
          </div>
        )}

        {/* Locked State for Coaches */}
        {isCoach && !isActivated && (
          <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-12 text-center shadow-xl opacity-50 pointer-events-none">
            <h2 className="text-xl font-semibold text-white mb-2">Account Activation Required</h2>
            <p className="text-gray-400 mb-4">Please enter your activation key above to unlock all features.</p>
            <p className="text-sm text-gray-500">Contact us if you need an activation key.</p>
          </div>
        )}

        {/* Regular Dashboard Content (only shown if coach is activated or player) */}
        {(!isCoach || isActivated) && (
          <>
            {isLoading && <div className="text-center text-gray-400 py-12">Loading matches...</div>}

        {error && (
          <div className="bg-red-900/20 border-2 border-red-800 rounded-lg p-4">
            <p className="text-red-400">Error loading matches. Please try again.</p>
          </div>
        )}

        {!isLoading && !error && (!filteredMatches || filteredMatches.length === 0) && (
          <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-12 text-center shadow-xl">
            <h2 className="text-xl font-semibold text-white mb-2">Welcome to Courtvision!</h2>
            <p className="text-gray-400 mb-4">Upload your first match to see detailed analytics and insights.</p>
            <p className="text-sm text-gray-500">Click the + button in the bottom right to get started.</p>
          </div>
        )}

        {!isLoading && !error && sortedDates.length > 0 && (
          <div className="space-y-4">
            {sortedDates.map((date: string) => {
              const dateMatches = groupedMatches![date]
              const isExpanded = expandedDate === date

              return (
                <div key={date} className="bg-[#1a1a1a] rounded-lg border border-[#333333] shadow-xl overflow-hidden transition-all duration-200 ease-in-out hover:border-[#50C878]/30">
                  <button
                    onClick={() => setExpandedDate(isExpanded ? null : date)}
                    className="w-full flex justify-between items-center px-6 py-4 hover:bg-[#262626]/50 transition-all duration-200 ease-in-out"
                  >
                    <span className="text-lg font-semibold text-white">{date}</span>
                    <span className="text-sm text-gray-400">
                      {dateMatches.length} {dateMatches.length === 1 ? "match" : "matches"}
                    </span>
                  </button>

                  {isExpanded && (
                    <div className="px-6 pb-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {dateMatches.map((match: Match) => (
                          <MatchCard key={match.id} match={match} />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
          </>
        )}
      </div>
    </MainLayout>
  )
}
