"use client"

import { useState } from "react"
import { useRouter, usePathname } from "next/navigation"
import Image from "next/image"
import Link from "next/link"
import { Home, BarChart3, Users, User, LogOut } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { useAuth } from "@/hooks/useAuth"

const navItems = [
  { name: "Dashboard", href: "/dashboard", icon: Home },
  { name: "Stats", href: "/stats", icon: BarChart3 },
  { name: "Teams", href: "/teams", icon: Users },
]

export function Sidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const { signOut } = useAuth()
  const [isSigningOut, setIsSigningOut] = useState(false)

  const handleSignOut = async () => {
    setIsSigningOut(true)
    try {
      await signOut()
      router.push("/")
    } finally {
      setIsSigningOut(false)
    }
  }

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 bg-[#0a0a0a] border-r border-[#333333] flex flex-col">
      {/* Header with Logo */}
      <div className="h-16 flex items-center px-4 border-b border-[#333333]">
        <Link href="/dashboard" className="flex items-center gap-3 hover:opacity-90 transition-opacity">
          <Image src="/CourtVisionLogo.png" alt="Courtvision Logo" width={40} height={40} className="w-10 h-10" />
          <span className="text-lg font-bold text-white">Courtvision</span>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-4 space-y-1">
        {navItems.map((item) => {
          const isActive = pathname === item.href
          return (
            <Link
              key={item.name}
              href={item.href}
              className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200 ease-in-out ${
                isActive
                  ? "bg-[#50C878]/20 text-[#50C878] border-l-2 border-[#50C878]"
                  : "text-gray-300 hover:bg-[#262626] hover:text-white transform hover:translate-x-1"
              }`}
            >
              <item.icon className="h-5 w-5" />
              {item.name}
            </Link>
          )
        })}
      </nav>

      {/* Footer with Profile */}
      <div className="p-4 border-t border-[#333333]">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              className="w-full justify-start gap-3 text-gray-300 hover:text-white hover:bg-[#262626]"
            >
              <User className="h-5 w-5" />
              <span>Account</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-48 bg-[#1a1a1a] border-[#333333]">
            <DropdownMenuItem
              onClick={() => router.push("/profile")}
              className="text-white hover:bg-[#262626] cursor-pointer transition-all duration-200 ease-in-out"
            >
              <User className="mr-2 h-4 w-4" />
              Profile
            </DropdownMenuItem>
            <DropdownMenuSeparator className="bg-[#333333]" />
            <DropdownMenuItem
              onClick={handleSignOut}
              disabled={isSigningOut}
              className="text-white hover:bg-[#262626] cursor-pointer transition-all duration-200 ease-in-out"
            >
              <LogOut className="mr-2 h-4 w-4" />
              {isSigningOut ? "Signing out..." : "Sign Out"}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </aside>
  )
}
