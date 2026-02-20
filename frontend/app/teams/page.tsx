"use client"

import { useState } from "react"
import { MainLayout } from "@/components/layout/main-layout"
import { CreateTeam } from "@/components/team/create-team"
import { TeamCode } from "@/components/team/team-code"
import { TeamMembers } from "@/components/team/team-members"
import { ActivationKeyInput } from "@/components/activation/activation-key-input"
import { CreateTeamLockedModal } from "@/components/team/create-team-locked-modal"
import { useTeams } from "@/hooks/useTeams"
import { useProfile } from "@/hooks/useProfile"
import { useActivation } from "@/hooks/useActivation"
import { Button } from "@/components/ui/button"
import type { Team } from "@/lib/types"

export default function TeamsPage() {
  const { teams, isLoading } = useTeams()
  const { profile } = useProfile()
  const { isActivated } = useActivation()
  const [showCreate, setShowCreate] = useState(false)
  const [showLockedModal, setShowLockedModal] = useState(false)

  const isCoach = profile?.role === "coach"

  return (
    <MainLayout>
      <div className="mx-auto px-4 py-8 max-w-7xl">
        <h1 className="text-3xl font-bold text-white mb-6">Teams</h1>

        {isCoach ? (
          <div className="space-y-6">
            {/* Activation Key Input for Coaches */}
            {!isActivated && (
              <div className="mb-6">
                <ActivationKeyInput />
              </div>
            )}

            {/* Create Team Section - Locked if not activated */}
            <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl transition-all duration-200 ease-in-out hover:border-[#50C878]/30">
              <h2 className="text-xl font-semibold text-white mb-4">Create Team</h2>

              {teams.length === 0 && !showCreate && (
                <div className="mb-4">
                  <p className="text-gray-400 mb-2">Welcome! Get started by creating your first team.</p>
                  <p className="text-sm text-gray-500">Teams help you organize and track your players&apos; matches.</p>
                </div>
              )}

              {!showCreate ? (
                <Button
                  onClick={() => {
                    if (!isActivated) {
                      setShowLockedModal(true)
                    } else {
                      setShowCreate(true)
                    }
                  }}
                  className="bg-[#50C878] hover:bg-[#45b069] text-black font-semibold"
                >
                  Create New Team
                </Button>
              ) : (
                <CreateTeam onCreated={() => setShowCreate(false)} />
              )}
            </div>

            {/* Join Team Section - Always available */}
            <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl transition-all duration-200 ease-in-out hover:border-[#50C878]/30">
              <h2 className="text-xl font-semibold text-white mb-4">Join Team</h2>
              <p className="text-sm text-gray-400 mb-4">
                Join an existing team if another coach created it. If the team has an activated coach, your account will be activated automatically.
              </p>
              <TeamCode />
            </div>

            {/* Your Teams Section */}
            {!isLoading && teams.length > 0 && (
              <div className="space-y-4">
                <h2 className="text-xl font-semibold text-white">Your Teams</h2>
                {teams.map((team: Team) => (
                  <div key={team.id} className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl transition-all duration-200 ease-in-out hover:border-[#50C878]/30 animate-fade-in">
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="text-lg font-semibold text-white">{team.name}</h3>
                      <span className="bg-black/50 border border-[#50C878]/50 rounded px-3 py-1 text-sm font-mono font-semibold text-[#50C878]">
                        {team.code}
                      </span>
                    </div>
                    <div className="mt-4">
                      <h4 className="text-sm font-medium text-gray-400 mb-3">Team Members</h4>
                      <TeamMembers teamId={team.id} teamName={team.name} />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-6">
            {/* Join Team Section */}
            <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl transition-all duration-200 ease-in-out hover:border-[#50C878]/30">
              <h2 className="text-xl font-semibold text-white mb-4">Join Team</h2>

              {teams.length === 0 && (
                <div className="mb-4">
                  <p className="text-gray-400 mb-2">Join a team to get started!</p>
                  <p className="text-sm text-gray-500">Ask your coach for a team code to join.</p>
                </div>
              )}

              <TeamCode />
            </div>

            {/* Your Teams Section */}
            {!isLoading && teams.length > 0 && (
              <div className="space-y-4">
                <h2 className="text-xl font-semibold text-white">Your Teams</h2>
                {teams.map((team: Team) => (
                  <div key={team.id} className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl transition-all duration-200 ease-in-out hover:border-[#50C878]/30 animate-fade-in">
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="text-lg font-semibold text-white">{team.name}</h3>
                      <span className="bg-black/50 border border-[#50C878]/50 rounded px-3 py-1 text-sm font-mono font-semibold text-[#50C878]">
                        {team.code}
                      </span>
                    </div>
                    <div className="mt-4">
                      <h4 className="text-sm font-medium text-gray-400 mb-3">Team Members</h4>
                      <TeamMembers teamId={team.id} teamName={team.name} />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
      <CreateTeamLockedModal isOpen={showLockedModal} onClose={() => setShowLockedModal(false)} />
    </MainLayout>
  )
}
