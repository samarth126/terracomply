import { Button } from "@/components/ui/button"
import { Building2, Settings } from "lucide-react"

export function Header() {
  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Building2 className="h-8 w-8 text-primary" />
              <span className="text-xl font-bold text-foreground">TERRACOMPLY</span>
            </div>
          </div>

          <nav className="hidden md:flex items-center gap-8">
            <a href="#models" className="text-muted-foreground hover:text-foreground transition-colors">
              Models
            </a>
            <a href="#query" className="text-muted-foreground hover:text-foreground transition-colors">
              Query Interface
            </a>
            <a href="#architecture" className="text-muted-foreground hover:text-foreground transition-colors">
              Architecture
            </a>
            <a href="#implementation" className="text-muted-foreground hover:text-foreground transition-colors">
              Implementation
            </a>
          </nav>

          <div className="flex items-center gap-3">
            <Button variant="ghost" size="sm">
              <Settings className="h-4 w-4 mr-2" />
              Settings
            </Button>
            <Button size="sm">Get Started</Button>
          </div>
        </div>
      </div>
    </header>
  )
}
