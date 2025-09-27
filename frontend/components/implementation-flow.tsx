import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { CheckCircle, Circle, ArrowRight, Database, Brain, Settings, TestTube, Rocket, MapPin } from "lucide-react"

const implementationSteps = [
  {
    phase: "Phase 1: Foundation",
    duration: "2-3 weeks",
    status: "ready",
    icon: Database,
    steps: [
      "Set up CrewAI framework and environment",
      "Design central router agent architecture",
      "Create vector database for RAG systems",
      "Collect and organize regulatory documents",
    ],
  },
  {
    phase: "Phase 2: Model Development",
    duration: "4-6 weeks",
    status: "planning",
    icon: Brain,
    steps: [
      "Build RAG systems for MLS, zoning, and AML compliance",
      "Fine-tune models for transaction and tenant law",
      "Integrate Gemini API for multimodal analysis",
      "Develop specialized compliance agents",
    ],
  },
  {
    phase: "Phase 3: Integration",
    duration: "2-3 weeks",
    status: "pending",
    icon: Settings,
    steps: [
      "Connect agents through CrewAI orchestration",
      "Implement query routing logic",
      "Build user interface and API endpoints",
      "Set up model serving infrastructure",
    ],
  },
  {
    phase: "Phase 4: Testing & Validation",
    duration: "3-4 weeks",
    status: "pending",
    icon: TestTube,
    steps: [
      "Test accuracy across all compliance domains",
      "Validate multimodal image analysis",
      "Performance testing and optimization",
      "User acceptance testing with real cases",
    ],
  },
  {
    phase: "Phase 5: Deployment",
    duration: "1-2 weeks",
    status: "pending",
    icon: Rocket,
    steps: [
      "Production deployment and monitoring",
      "User training and documentation",
      "Feedback collection and iteration",
      "Scale to additional jurisdictions",
    ],
  },
]

const scopeRecommendations = [
  {
    scope: "Single State Focus",
    icon: MapPin,
    pros: ["Faster development", "Focused compliance rules", "Easier validation"],
    cons: ["Limited market reach", "State-specific only"],
    recommendation: "Start with California or Texas for maximum impact",
  },
  {
    scope: "Multi-State Approach",
    icon: MapPin,
    pros: ["Broader market appeal", "Scalable architecture", "Higher ROI potential"],
    cons: ["Complex rule variations", "Longer development time"],
    recommendation: "Begin with 3-5 major real estate markets",
  },
]

export function ImplementationFlow() {
  return (
    <section id="implementation" className="py-24 px-6 bg-card/30">
      <div className="container mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold mb-4">Implementation Roadmap</h2>
          <p className="text-xl text-muted-foreground text-balance max-w-3xl mx-auto">
            {
              "Step-by-step guide to building your real estate compliance platform with CrewAI framework and specialized AI models."
            }
          </p>
        </div>

        {/* Scope Recommendations */}
        <div className="mb-16">
          <h3 className="text-2xl font-semibold mb-8 text-center">Scope Recommendations</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
            {scopeRecommendations.map((scope, index) => {
              const Icon = scope.icon
              return (
                <Card key={index} className="gradient-border">
                  <div className="gradient-border-content">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Icon className="h-5 w-5 text-primary" />
                        {scope.scope}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div>
                          <h5 className="font-medium text-sm mb-2 text-green-400">Pros</h5>
                          <ul className="space-y-1">
                            {scope.pros.map((pro, i) => (
                              <li key={i} className="text-sm text-muted-foreground flex items-center gap-2">
                                <CheckCircle className="h-3 w-3 text-green-400 flex-shrink-0" />
                                {pro}
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <h5 className="font-medium text-sm mb-2 text-orange-400">Cons</h5>
                          <ul className="space-y-1">
                            {scope.cons.map((con, i) => (
                              <li key={i} className="text-sm text-muted-foreground flex items-center gap-2">
                                <Circle className="h-3 w-3 text-orange-400 flex-shrink-0" />
                                {con}
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div className="pt-2 border-t border-border">
                          <p className="text-sm font-medium text-primary">{scope.recommendation}</p>
                        </div>
                      </div>
                    </CardContent>
                  </div>
                </Card>
              )
            })}
          </div>
        </div>

        {/* Implementation Timeline */}
        <div className="space-y-6">
          {implementationSteps.map((phase, index) => {
            const Icon = phase.icon
            const isCompleted = phase.status === "completed"
            const isActive = phase.status === "ready"

            return (
              <Card key={index} className={`${isActive ? "gradient-border" : ""}`}>
                <div className={isActive ? "gradient-border-content" : ""}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div
                          className={`p-2 rounded-lg ${
                            isCompleted ? "bg-green-500/10" : isActive ? "bg-primary/10" : "bg-muted"
                          }`}
                        >
                          <Icon
                            className={`h-5 w-5 ${
                              isCompleted ? "text-green-400" : isActive ? "text-primary" : "text-muted-foreground"
                            }`}
                          />
                        </div>
                        <div>
                          <CardTitle className="text-lg">{phase.phase}</CardTitle>
                          <CardDescription>{phase.duration}</CardDescription>
                        </div>
                      </div>
                      <Badge variant={isCompleted ? "default" : isActive ? "default" : "secondary"}>
                        {phase.status === "ready"
                          ? "Ready to Start"
                          : phase.status === "planning"
                            ? "In Planning"
                            : phase.status === "completed"
                              ? "Completed"
                              : "Pending"}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {phase.steps.map((step, stepIndex) => (
                        <div key={stepIndex} className="flex items-center gap-2 p-3 bg-muted/30 rounded-lg">
                          {isCompleted ? (
                            <CheckCircle className="h-4 w-4 text-green-400 flex-shrink-0" />
                          ) : (
                            <Circle className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                          )}
                          <span className="text-sm">{step}</span>
                        </div>
                      ))}
                    </div>
                    {isActive && (
                      <div className="mt-4 pt-4 border-t border-border">
                        <Button>
                          Start Phase 1
                          <ArrowRight className="ml-2 h-4 w-4" />
                        </Button>
                      </div>
                    )}
                  </CardContent>
                </div>
              </Card>
            )
          })}
        </div>

        {/* Next Steps */}
        <Card className="mt-12 gradient-border">
          <div className="gradient-border-content">
            <CardHeader>
              <CardTitle>Ready to Get Started?</CardTitle>
              <CardDescription>
                {
                  "Begin with Phase 1 to establish your CrewAI framework and start building your first compliance agent."
                }
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col sm:flex-row gap-4">
                <Button size="lg">
                  Start Building Now
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
                <Button variant="outline" size="lg">
                  Download Implementation Guide
                </Button>
                <Button variant="ghost" size="lg">
                  Schedule Consultation
                </Button>
              </div>
            </CardContent>
          </div>
        </Card>
      </div>
    </section>
  )
}
