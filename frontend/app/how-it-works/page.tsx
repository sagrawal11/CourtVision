"use client"

import { useState } from "react"
import { LandingNav } from "@/components/landing/landing-nav"
import { Footer } from "@/components/landing/footer"
import { Button } from "@/components/ui/button"
import { AuthModal } from "@/components/auth/auth-modal"
import { AnimatedStepCard } from "@/components/how-it-works/animated-step-card"

const steps = [
  {
    step: 1,
    title: "Upload Your Match",
    description: "Simply paste your Playsight video link into our upload modal. Our system will extract the video and prepare it for analysis. No need to download or process files manually. Just paste, click, and you're done.",
    imageUrl: "https://images.unsplash.com/photo-1622163642992-9db3c26b4e4a?w=800&h=600&fit=crop&q=80",
    imageAlt: "Tennis player uploading match video"
  },
  {
    step: 2,
    title: "Identify the Player",
    description: "We'll show you several frames from the video. Click on yourself (or the player you want to track) in each frame. Our computer vision system uses this information to track the player throughout the entire match using advanced color recognition and player tracking algorithms.",
    imageUrl: "https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=800&h=600&fit=crop&q=80",
    imageAlt: "Player identification interface"
  },
  {
    step: 3,
    title: "Automatic Processing",
    description: "Our computer vision backend processes the video automatically. Using state-of-the-art models for player detection, ball tracking, and shot classification, we analyze every moment of the match. This typically takes about an hour, and you'll receive real-time updates on the progress.",
    imageUrl: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=600&fit=crop&q=80",
    imageAlt: "Computer vision processing visualization"
  },
  {
    step: 4,
    title: "View Your Analytics",
    description: "Once processing is complete, explore your match on an interactive court diagram. Every shot is visualized with color-coded lines showing winners (green), errors (red), and shots in play (blue). Click on any shot to jump directly to that moment in the video. View comprehensive statistics broken down by game, set, and match.",
    imageUrl: "https://images.unsplash.com/photo-1551958219-acbc608c6377?w=800&h=600&fit=crop&q=80",
    imageAlt: "Interactive tennis court analytics dashboard"
  },
  {
    step: 5,
    title: "Track Your Progress",
    description: "Coaches can view all their players' matches in one dashboard, with powerful filtering options. Players can track their own performance over time. Our statistics page provides detailed insights into your game patterns, helping you identify areas for improvement and celebrate your victories.",
    imageUrl: "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=800&h=600&fit=crop&q=80",
    imageAlt: "Progress tracking and statistics dashboard"
  }
]

export default function HowItWorksPage() {
  const [authOpen, setAuthOpen] = useState(false)
  const [authMode, setAuthMode] = useState<"signin" | "signup">("signup")

  const openSignUp = () => {
    setAuthMode("signup")
    setAuthOpen(true)
  }

  const openSignIn = () => {
    setAuthMode("signin")
    setAuthOpen(true)
  }

  return (
    <main className="min-h-screen bg-black overflow-x-hidden">
      <LandingNav />
      
      {/* Hero Section */}
      <section className="pt-32 pb-8 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold text-white mb-6 animate-fade-in">
            How It <span className="text-[#50C878]">Works</span>
          </h1>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto animate-fade-in-delay">
            Transform your tennis matches into actionable insights with our computer vision powered analytics platform
          </p>
        </div>
      </section>

      {/* Animated Steps */}
      <div className="relative max-w-7xl mx-auto">
        {steps.map((step: typeof steps[number], index: number) => (
          <AnimatedStepCard key={step.step} step={step} index={index} />
        ))}
      </div>

      {/* CTA Section */}
      <section className="py-20 px-4">
        <div className="max-w-4xl mx-auto">
          <div className="bg-gradient-to-br from-[#1a1a1a] to-[#0f0f0f] rounded-2xl p-8 lg:p-12 border border-[#333333] text-center relative overflow-hidden group">
            {/* Animated background */}
            <div className="absolute inset-0 bg-gradient-to-br from-[#50C878]/10 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700" />
            <div className="absolute top-0 right-0 w-64 h-64 bg-[#50C878]/10 rounded-bl-full blur-3xl" />
            
            <div className="relative z-10">
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
                Ready to get started?
              </h2>
              <p className="text-gray-400 mb-8 text-lg">
                Sign in to your existing account or create a new one to start analyzing your matches today.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button
                  onClick={openSignIn}
                  variant="outline"
                  className="!border-white !text-white hover:!text-[#50C878] hover:!border-[#50C878] !bg-transparent hover:!bg-transparent transition-all duration-200 ease-in-out transform hover:scale-105"
                >
                  Sign In
                </Button>
                <Button
                  onClick={openSignUp}
                  className="bg-[#50C878] hover:bg-[#45b069] text-black font-semibold transition-all duration-200 ease-in-out transform hover:scale-105"
                >
                  Create Account
                </Button>
              </div>
            </div>
          </div>
        </div>
      </section>

      <Footer />
      <AuthModal isOpen={authOpen} onClose={() => setAuthOpen(false)} initialMode={authMode} />
    </main>
  )
}
