import { Button } from "@/components/ui/button"
import { ArrowRight, Sparkles } from "lucide-react"

export function HeroSection() {
  return (
    <section className="gradient-bg py-24 px-6">
      <div className="container mx-auto text-center max-w-4xl">
        <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-2 rounded-full text-sm font-medium mb-8">
          <Sparkles className="h-4 w-4" />
          AI-Powered Compliance Framework
        </div>

        <h1 className="text-5xl md:text-6xl font-bold text-balance mb-6">
          TERRACOMPLY
          <span className="text-primary block">AI Compliance Platform</span>
        </h1>

        <p className="text-xl text-muted-foreground text-balance mb-8 leading-relaxed">
          {
            "Streamline real estate compliance with specialized AI agents for MLS rules, zoning codes, transaction reviews, and regulatory requirements. Built with CrewAI for intelligent routing and multimodal analysis."
          }
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button size="lg" className="text-lg px-8">
            Start Building
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
          <Button variant="outline" size="lg" className="text-lg px-8 bg-transparent">
            View Architecture
          </Button>
        </div>
      </div>
    </section>
  )
}
