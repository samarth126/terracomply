import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Brain, Database, Cpu, Cloud, GitBranch, Zap, Shield, Globe } from "lucide-react"

const techComponents = [
  {
    category: "AI Framework",
    icon: Brain,
    items: [
      { name: "CrewAI", description: "Multi-agent orchestration", status: "Core" },
      { name: "Central Router Agent", description: "Query routing & coordination", status: "Core" },
      { name: "Specialized Agents", description: "5 domain-specific agents", status: "Core" },
    ],
  },
  {
    category: "AI Models",
    icon: Cpu,
    items: [
      { name: "Fine-tuned Models", description: "Transaction & tenant law models", status: "Training" },
      { name: "RAG Systems", description: "MLS, zoning, AML knowledge bases", status: "Active" },
      { name: "Gemini API", description: "Multimodal image analysis", status: "Integration" },
    ],
  },
  {
    category: "Data & Storage",
    icon: Database,
    items: [
      { name: "Vector Database", description: "Embeddings for RAG systems", status: "Required" },
      { name: "Document Store", description: "Regulatory documents & codes", status: "Required" },
      { name: "Model Registry", description: "Version control for AI models", status: "Planned" },
    ],
  },
  {
    category: "Infrastructure",
    icon: Cloud,
    items: [
      { name: "API Gateway", description: "Request routing & rate limiting", status: "Required" },
      { name: "Model Serving", description: "Scalable inference endpoints", status: "Required" },
      { name: "Monitoring", description: "Model performance tracking", status: "Planned" },
    ],
  },
]

export function TechStack() {
  return (
    <section id="architecture" className="py-24 px-6">
      <div className="container mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold mb-4">Technical Architecture</h2>
          <p className="text-xl text-muted-foreground text-balance max-w-3xl mx-auto">
            {
              "Built on CrewAI framework with specialized agents, RAG systems, and fine-tuned models for comprehensive real estate compliance."
            }
          </p>
        </div>

        {/* Architecture Diagram */}
        <Card className="mb-12 gradient-border">
          <div className="gradient-border-content">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <GitBranch className="h-5 w-5 text-primary" />
                System Architecture Flow
              </CardTitle>
              <CardDescription>
                How queries flow through the CrewAI framework to specialized compliance models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col lg:flex-row items-center justify-between gap-8 p-6">
                {/* User Input */}
                <div className="text-center">
                  <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mb-3">
                    <Globe className="h-8 w-8 text-primary" />
                  </div>
                  <h4 className="font-semibold mb-2">User Query</h4>
                  <p className="text-sm text-muted-foreground">Text, Image, or Document</p>
                </div>

                <div className="flex items-center">
                  <div className="w-8 h-0.5 bg-primary/30 hidden lg:block" />
                  <Zap className="h-5 w-5 text-primary mx-2" />
                  <div className="w-8 h-0.5 bg-primary/30 hidden lg:block" />
                </div>

                {/* Central Agent */}
                <div className="text-center">
                  <div className="w-16 h-16 bg-accent/10 rounded-full flex items-center justify-center mb-3">
                    <Brain className="h-8 w-8 text-accent" />
                  </div>
                  <h4 className="font-semibold mb-2">Central Router</h4>
                  <p className="text-sm text-muted-foreground">CrewAI Orchestrator</p>
                </div>

                <div className="flex items-center">
                  <div className="w-8 h-0.5 bg-primary/30 hidden lg:block" />
                  <Zap className="h-5 w-5 text-primary mx-2" />
                  <div className="w-8 h-0.5 bg-primary/30 hidden lg:block" />
                </div>

                {/* Specialized Models */}
                <div className="text-center">
                  <div className="w-16 h-16 bg-chart-2/10 rounded-full flex items-center justify-center mb-3">
                    <Shield className="h-8 w-8 text-chart-2" />
                  </div>
                  <h4 className="font-semibold mb-2">Specialized Agent</h4>
                  <p className="text-sm text-muted-foreground">Domain Expert Model</p>
                </div>
              </div>
            </CardContent>
          </div>
        </Card>

        {/* Tech Stack Components */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {techComponents.map((component, index) => {
            const Icon = component.icon
            return (
              <Card key={index}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Icon className="h-5 w-5 text-primary" />
                    {component.category}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {component.items.map((item, itemIndex) => (
                      <div key={itemIndex} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
                        <div>
                          <h5 className="font-medium text-sm">{item.name}</h5>
                          <p className="text-xs text-muted-foreground">{item.description}</p>
                        </div>
                        <Badge
                          variant={
                            item.status === "Core" || item.status === "Active"
                              ? "default"
                              : item.status === "Required"
                                ? "secondary"
                                : "outline"
                          }
                          className="text-xs"
                        >
                          {item.status}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>
    </section>
  )
}
