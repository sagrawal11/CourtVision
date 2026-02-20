'use client'

import { useQuery } from '@tanstack/react-query'
import { createClient } from '@/lib/supabase/client'
import { useAuth } from './useAuth'
import { useTeams } from './useTeams'
import type { Team } from '@/lib/types'

export function useTeamActivation() {
  const { getUser } = useAuth()
  const { teams } = useTeams()
  const supabase = createClient()

  const { data: hasActivatedTeam, isLoading } = useQuery({
    queryKey: ['team-activation-status', teams.map((t: Team) => t.id).join(',')],
    queryFn: async () => {
      const user = await getUser()
      if (!user || teams.length === 0) return false

      const { data: { session } } = await supabase.auth.getSession()
      if (!session) return false

      // For each team, check if it has an activated coach
      for (const team of teams) {
        try {
          // Get all coaches on this team
          const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/teams/${team.id}/members`, {
            headers: {
              'Authorization': `Bearer ${session.access_token}`,
            },
          })

          if (response.ok) {
            const data = await response.json()
            // Backend returns { members: [...] } - handle both formats
            const membersList = data.members || data || []
            const coaches = membersList.filter((m: any) => m.users?.role === 'coach')
            
            // Check if any coach is activated
            const hasActivatedCoach = coaches.some((coach: any) => {
              // Check if coach has activated_at set (not null and not undefined)
              const activatedAt = coach.users?.activated_at
              return activatedAt !== null && activatedAt !== undefined && activatedAt !== ''
            })

            if (hasActivatedCoach) {
              return true
            }
          }
        } catch (err) {
          console.error('Error checking team activation:', err)
        }
      }

      return false
    },
    enabled: teams.length > 0,
  })

  return {
    hasActivatedTeam: hasActivatedTeam || false,
    isLoading,
  }
}
