import { Header } from "@/components/header"
import { HeroSection } from "@/components/hero-section"
import { ComplianceModels } from "@/components/compliance-models"
import { QueryInterface } from "@/components/query-interface"
import { TechStack } from "@/components/tech-stack"
import { ImplementationFlow } from "@/components/implementation-flow"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main>
        <HeroSection />
        <ComplianceModels />
        <QueryInterface />
        <TechStack />
        <ImplementationFlow />
      </main>
    </div>
  )
}
