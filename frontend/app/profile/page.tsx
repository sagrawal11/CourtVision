"use client"

import { MainLayout } from "@/components/layout/main-layout"
import { useProfile } from "@/hooks/useProfile"
import { useTeams } from "@/hooks/useTeams"
import { useAuth } from "@/hooks/useAuth"
import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Trash2, Archive, Edit, LogOut } from "lucide-react"
import { RenameTeamModal } from "@/components/team/rename-team-modal"
import { ArchiveTeamModal } from "@/components/team/archive-team-modal"
import { DeleteTeamModal } from "@/components/team/delete-team-modal"
import { LeaveTeamModal } from "@/components/team/leave-team-modal"
import type { Team } from "@/lib/types"

export default function ProfilePage() {
  const { profile, isLoading } = useProfile()
  const { teams, archivedTeams, archiveTeam, unarchiveTeam, deleteTeam, leaveTeam, isArchiving, isUnarchiving, isDeleting, isLeaving } = useTeams()
  const { getUser } = useAuth()
  const [email, setEmail] = useState<string | null>(null)
  const [renameModalOpen, setRenameModalOpen] = useState(false)
  const [archiveModalOpen, setArchiveModalOpen] = useState(false)
  const [deleteModalOpen, setDeleteModalOpen] = useState(false)
  const [leaveModalOpen, setLeaveModalOpen] = useState(false)
  const [selectedTeamId, setSelectedTeamId] = useState<string | null>(null)
  const [selectedTeamName, setSelectedTeamName] = useState<string | null>(null)

  useEffect(() => {
    const fetchEmail = async () => {
      const user = await getUser()
      setEmail(user?.email || null)
    }
    fetchEmail()
  }, [getUser])

  return (
    <MainLayout>
      <div className="mx-auto px-4 py-8 max-w-7xl">
        <h1 className="text-3xl font-bold text-white mb-6">Profile</h1>

        <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl transition-all duration-200 ease-in-out">
          {isLoading ? (
            <p className="text-gray-400">Loading profile...</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <p className="text-sm font-medium text-gray-400 mb-1">Name</p>
                <p className="text-base text-white">{profile?.name || "Not set"}</p>
              </div>

              <div>
                <p className="text-sm font-medium text-gray-400 mb-1">Email</p>
                <p className="text-base text-white">{email || "Loading..."}</p>
              </div>

              <div>
                <p className="text-sm font-medium text-gray-400 mb-1">Role</p>
                <span
                  className={`inline-flex px-2 py-1 rounded text-sm font-medium border ${
                    profile?.role === "coach"
                      ? "bg-blue-900/30 text-blue-400 border-blue-800"
                      : "bg-emerald-900/30 text-emerald-400 border-emerald-800"
                  }`}
                >
                  {profile?.role ? profile.role.charAt(0).toUpperCase() + profile.role.slice(1) : "Unknown"}
                </span>
              </div>

              <div>
                <p className="text-sm font-medium text-gray-400 mb-2">Teams</p>
                {teams.length === 0 ? (
                  <p className="text-sm text-gray-500 italic">Not part of any teams yet.</p>
                ) : (
                  <div className="space-y-3">
                    {teams.map((team: Team) => {
                      const isCoach = profile?.role === "coach"
                      return (
                        <div key={team.id} className="bg-black/50 rounded border border-[#333333] p-4 transition-all duration-200 ease-in-out hover:border-[#50C878]/50">
                          <div className="flex justify-between items-start mb-3">
                            <div>
                              <p className="font-medium text-white">{team.name}</p>
                              <p className="text-xs text-gray-400 font-mono mt-1">{team.code}</p>
                            </div>
                          </div>
                          
                          {isCoach ? (
                            <div className="flex flex-wrap gap-2 mt-3 pt-3 border-t border-[#333333]">
                              <Button
                                variant="outline"
                                size="sm"
                                className="border-[#333333] text-gray-300 hover:border-[#50C878] hover:text-[#50C878] hover:bg-transparent bg-transparent"
                                onClick={() => {
                                  setSelectedTeamId(team.id)
                                  setSelectedTeamName(team.name)
                                  setRenameModalOpen(true)
                                }}
                              >
                                <Edit className="h-4 w-4 mr-2" />
                                Rename
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                className="border-[#333333] text-gray-300 hover:border-yellow-600 hover:text-yellow-500 hover:bg-transparent bg-transparent"
                                onClick={() => {
                                  setSelectedTeamId(team.id)
                                  setSelectedTeamName(team.name)
                                  setArchiveModalOpen(true)
                                }}
                              >
                                <Archive className="h-4 w-4 mr-2" />
                                Archive
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                className="border-red-800 text-red-400 hover:bg-red-600 hover:text-black hover:border-red-600 bg-transparent"
                                onClick={() => {
                                  setSelectedTeamId(team.id)
                                  setSelectedTeamName(team.name)
                                  setDeleteModalOpen(true)
                                }}
                              >
                                <Trash2 className="h-4 w-4 mr-2" />
                                Delete
                              </Button>
                            </div>
                          ) : (
                            <div className="flex flex-wrap gap-2 mt-3 pt-3 border-t border-[#333333]">
                              <Button
                                variant="outline"
                                size="sm"
                                className="border-red-800 text-red-400 hover:bg-red-600 hover:text-black hover:border-red-600 bg-transparent"
                                onClick={() => {
                                  setSelectedTeamId(team.id)
                                  setSelectedTeamName(team.name)
                                  setLeaveModalOpen(true)
                                }}
                              >
                                <LogOut className="h-4 w-4 mr-2" />
                                Leave Team
                              </Button>
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>

              {/* Archived Teams Section (Coaches Only) */}
              {profile?.role === "coach" && archivedTeams && archivedTeams.length > 0 && (
                <div className="mt-6">
                  <p className="text-sm font-medium text-gray-400 mb-2">Archived Teams</p>
                  <div className="space-y-3">
                    {archivedTeams.map((team: any) => {
                      const archivedByName = team.archived_by_user?.name || "Unknown"
                      const archivedDate = team.archived_at 
                        ? new Date(team.archived_at).toLocaleDateString('en-US', { 
                            year: 'numeric', 
                            month: 'short', 
                            day: 'numeric' 
                          })
                        : "Unknown date"
                      
                      return (
                        <div key={team.id} className="bg-black/50 rounded border border-yellow-600/50 p-4 opacity-75">
                          <div className="flex justify-between items-start mb-3">
                            <div>
                              <p className="font-medium text-white">{team.name}</p>
                              <p className="text-xs text-gray-400 font-mono mt-1">{team.code}</p>
                              <p className="text-xs text-yellow-500 mt-1 italic">
                                Archived by {archivedByName} on {archivedDate}
                              </p>
                            </div>
                          </div>
                        <div className="flex flex-wrap gap-2 mt-3 pt-3 border-t border-[#333333]">
                          <Button
                            variant="outline"
                            size="sm"
                            className="border-[#333333] text-gray-300 hover:border-[#50C878] hover:text-[#50C878] hover:bg-transparent bg-transparent"
                            onClick={async () => {
                              try {
                                await unarchiveTeam(team.id)
                              } catch (error) {
                                console.error('Failed to unarchive team:', error)
                              }
                            }}
                            disabled={isUnarchiving}
                          >
                            <Archive className="h-4 w-4 mr-2" />
                            {isUnarchiving ? "Unarchiving..." : "Unarchive"}
                          </Button>
                        </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {selectedTeamId && selectedTeamName && (
        <>
          <RenameTeamModal
            isOpen={renameModalOpen}
            onClose={() => {
              setRenameModalOpen(false)
              setSelectedTeamId(null)
              setSelectedTeamName(null)
            }}
            teamId={selectedTeamId}
            currentName={selectedTeamName}
          />
          <ArchiveTeamModal
            isOpen={archiveModalOpen}
            onClose={() => {
              setArchiveModalOpen(false)
              setSelectedTeamId(null)
              setSelectedTeamName(null)
            }}
            onConfirm={async () => {
              if (selectedTeamId) {
                try {
                  await archiveTeam(selectedTeamId)
                  setArchiveModalOpen(false)
                  setSelectedTeamId(null)
                  setSelectedTeamName(null)
                } catch (error) {
                  console.error('Failed to archive team:', error)
                }
              }
            }}
            teamName={selectedTeamName}
            loading={isArchiving}
          />
          <DeleteTeamModal
            isOpen={deleteModalOpen}
            onClose={() => {
              setDeleteModalOpen(false)
              setSelectedTeamId(null)
              setSelectedTeamName(null)
            }}
            onConfirm={async () => {
              if (selectedTeamId) {
                try {
                  await deleteTeam(selectedTeamId)
                  setDeleteModalOpen(false)
                  setSelectedTeamId(null)
                  setSelectedTeamName(null)
                } catch (error) {
                  console.error('Failed to delete team:', error)
                }
              }
            }}
            teamName={selectedTeamName}
            loading={isDeleting}
          />
          <LeaveTeamModal
            isOpen={leaveModalOpen}
            onClose={() => {
              setLeaveModalOpen(false)
              setSelectedTeamId(null)
              setSelectedTeamName(null)
            }}
            onConfirm={async () => {
              if (selectedTeamId) {
                try {
                  await leaveTeam(selectedTeamId)
                  setLeaveModalOpen(false)
                  setSelectedTeamId(null)
                  setSelectedTeamName(null)
                } catch (error) {
                  console.error('Failed to leave team:', error)
                }
              }
            }}
            teamName={selectedTeamName}
            loading={isLeaving}
          />
        </>
      )}
    </MainLayout>
  )
}
