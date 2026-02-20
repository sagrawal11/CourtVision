"use client"

import { useState, useEffect } from "react"
import { MainLayout } from "@/components/layout/main-layout"
import { ActivationKeyInput } from "@/components/activation/activation-key-input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { useMatches } from "@/hooks/useMatches"
import { useProfile } from "@/hooks/useProfile"
import { useTeams } from "@/hooks/useTeams"
import { useActivation } from "@/hooks/useActivation"
import { createClient } from "@/lib/supabase/client"
import type { Match } from "@/lib/types"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts"

const COLORS = ["#50C878", "#ef4444", "#3b82f6"]

export default function StatsPage() {
  const { data: matches, isLoading } = useMatches()
  const { profile } = useProfile()
  const { teams } = useTeams()
  const { isActivated } = useActivation()
  const supabase = createClient()
  const [selectedPlayerId, setSelectedPlayerId] = useState<string>("")
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
      if (uniqueMembers.length > 0 && !selectedPlayerId) {
        setSelectedPlayerId(uniqueMembers[0].id)
      }
    }

    fetchTeamMembers()
  }, [isCoach, teams, supabase, selectedPlayerId])

  // Filter matches by selected player
  const filteredMatches = isCoach && selectedPlayerId
    ? matches?.filter((m: Match) => m.user_id === selectedPlayerId) || []
    : matches || []

  const completedMatches = filteredMatches.filter((m: Match) => m.status === "completed") || []
  const totalMatches = completedMatches.length
  const totalShots = 0 // Placeholder - would come from shots data
  const avgShotsPerMatch = totalMatches > 0 ? Math.round(totalShots / totalMatches) : 0

  // Placeholder chart data
  const shotDistribution = [
    { name: "Winners", value: 35 },
    { name: "Errors", value: 25 },
    { name: "In Play", value: 40 },
  ]

  const matchPerformance = completedMatches.slice(0, 5).map((match: Match, i: number) => ({
    name: `Match ${i + 1}`,
    winners: Math.floor(Math.random() * 30) + 10,
    errors: Math.floor(Math.random() * 20) + 5,
  }))

  return (
    <MainLayout>
      <div className="mx-auto px-4 py-8 max-w-7xl">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-white">Statistics</h1>
          {isCoach && isActivated && teamMembers.length > 0 && (
            <div className="flex items-center gap-2">
              <Label htmlFor="playerSelect" className="text-gray-400 text-sm">
                Select Player:
              </Label>
              <Select value={selectedPlayerId} onValueChange={setSelectedPlayerId}>
                <SelectTrigger className="w-48 bg-[#1a1a1a] border-[#333333] text-white">
                  <SelectValue placeholder="Select a player" />
                </SelectTrigger>
                <SelectContent className="bg-[#1a1a1a] border-[#333333]">
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
        {isCoach && !isActivated ? (
          <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-12 text-center shadow-xl opacity-50 pointer-events-none">
            <h2 className="text-xl font-semibold text-white mb-2">Account Activation Required</h2>
            <p className="text-gray-400 mb-4">Please enter your activation key above to unlock all features.</p>
            <p className="text-sm text-gray-500">Contact us if you need an activation key.</p>
          </div>
        ) : (
          <>
            {isLoading ? (
          <p className="text-gray-400">Loading statistics...</p>
        ) : totalMatches === 0 ? (
          <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-12 text-center shadow-xl">
            <p className="text-gray-400">No statistics yet. Upload matches to see your performance analytics.</p>
          </div>
        ) : (
          <>
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
              <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl transition-all duration-200 ease-in-out hover:border-[#50C878]/30">
                <p className="text-sm font-medium text-gray-400 mb-2">Total Matches</p>
                <p className="text-3xl font-bold text-white">{totalMatches}</p>
              </div>
              <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl transition-all duration-200 ease-in-out hover:border-[#50C878]/30">
                <p className="text-sm font-medium text-gray-400 mb-2">Total Shots</p>
                <p className="text-3xl font-bold text-white">{totalShots}</p>
              </div>
              <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl transition-all duration-200 ease-in-out hover:border-[#50C878]/30">
                <p className="text-sm font-medium text-gray-400 mb-2">Win Rate</p>
                <p className="text-3xl font-bold text-[#50C878]">58%</p>
              </div>
              <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl transition-all duration-200 ease-in-out hover:border-[#50C878]/30">
                <p className="text-sm font-medium text-gray-400 mb-2">Avg Shots/Match</p>
                <p className="text-3xl font-bold text-white">{avgShotsPerMatch}</p>
              </div>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Shot Distribution */}
              <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl transition-all duration-200 ease-in-out hover:border-[#50C878]/30">
                <h3 className="text-lg font-semibold text-white mb-4">Shot Distribution</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={shotDistribution}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="value"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {shotDistribution.map((_: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#1a1a1a",
                        border: "1px solid #333",
                        borderRadius: "8px",
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Performance by Match */}
              <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl transition-all duration-200 ease-in-out hover:border-[#50C878]/30">
                <h3 className="text-lg font-semibold text-white mb-4">Performance by Match</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={matchPerformance}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="name" stroke="#a0a0a0" />
                    <YAxis stroke="#a0a0a0" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#1a1a1a",
                        border: "1px solid #333",
                        borderRadius: "8px",
                      }}
                    />
                    <Bar dataKey="winners" fill="#50C878" name="Winners" />
                    <Bar dataKey="errors" fill="#ef4444" name="Errors" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
            )}
          </>
        )}
      </div>
    </MainLayout>
  )
}
