import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { FileCheck, Building, Users, Shield, MapPin, CheckCircle, Settings, Eye } from "lucide-react"

const complianceModels = [
  {
    id: "deal-listing",
    title: "Deal & Listing Compliance",
    description:
      "Check listings against MLS rules, Fair Housing Act wording, RESPA/TILA/CFPB disclosures, and local advertising rules.",
    icon: FileCheck,
    type: "RAG System",
    status: "Active",
    accuracy: "94.2%",
    features: ["MLS Rule Validation", "Fair Housing Compliance", "RESPA/TILA Checks", "Local Ad Rules"],
  },
  {
    id: "transaction-review",
    title: "Transaction Review",
    description: "Validate offer/contract packets for required addenda, contingencies, and mandatory disclosures.",
    icon: Building,
    type: "RAG System",
    status: "Active",
    accuracy: "96.8%",
    features: ["Contract Validation", "Addenda Checks", "Lead Paint Disclosure", "Flood Zone Analysis"],
  },
  {
    id: "property-zoning",
    title: "Property & Zoning Checks",
    description:
      "Summarize zoning codes, permitted uses, short-term rental rules, setback requirements, and parking regulations.",
    icon: MapPin,
    type: "RAG System",
    status: "Active",
    accuracy: "91.5%",
    features: ["Zoning Code Analysis", "Permitted Use Validation", "STR Compliance", "Setback Requirements"],
  },
  {
    id: "tenant-landlord",
    title: "Tenant/Landlord Compliance",
    description:
      "State landlord-tenant statutes including notice periods, habitability standards, security deposits, and eviction timelines.",
    icon: Users,
    type: "RAG System",
    status: "Training",
    accuracy: "89.3%",
    features: ["Notice Period Rules", "Habitability Standards", "Security Deposit Laws", "Eviction Procedures"],
  },
  {
    id: "aml-fraud",
    title: "AML/Fraud Risk Assessment",
    description:
      "FinCEN Geographic Targeting Orders, beneficial ownership checks, and KYC/KYB workflows for luxury/foreign buyers.",
    icon: Shield,
    type: "Fine-tuned Model",
    status: "Development",
    accuracy: "97.1%",
    features: ["FinCEN GTO Compliance", "Beneficial Ownership", "KYC/KYB Workflows", "Risk Scoring"],
  },
]

export function ComplianceModels() {
  return (
    <section id="models" className="py-24 px-6">
      <div className="container mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold mb-4">Specialized Compliance Models</h2>
          <p className="text-xl text-muted-foreground text-balance max-w-3xl mx-auto">
            {
              "Each model is either fine-tuned or uses RAG systems for specific regulatory domains, ensuring accurate and up-to-date compliance checking."
            }
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {complianceModels.map((model) => {
            const Icon = model.icon
            return (
              <div key={model.id} className="gradient-border">
                <Card className="gradient-border-content h-full">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-primary/10 rounded-lg">
                          <Icon className="h-6 w-6 text-primary" />
                        </div>
                        <div>
                          <Badge variant={model.status === "Active" ? "default" : "secondary"} className="mb-2">
                            {model.status}
                          </Badge>
                        </div>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        {model.type}
                      </Badge>
                    </div>
                    <CardTitle className="text-lg">{model.title}</CardTitle>
                    <CardDescription className="text-sm leading-relaxed">{model.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="mb-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-muted-foreground">Accuracy</span>
                        <span className="text-sm font-medium text-primary">{model.accuracy}</span>
                      </div>
                      <div className="w-full bg-secondary rounded-full h-2">
                        <div
                          className="bg-primary h-2 rounded-full transition-all duration-300"
                          style={{ width: model.accuracy }}
                        />
                      </div>
                    </div>

                    <div className="space-y-2 mb-4">
                      {model.features.map((feature, index) => (
                        <div key={index} className="flex items-center gap-2 text-sm">
                          <CheckCircle className="h-3 w-3 text-primary flex-shrink-0" />
                          <span className="text-muted-foreground">{feature}</span>
                        </div>
                      ))}
                    </div>

                    <div className="flex gap-2">
                      <Button variant="outline" size="sm" className="flex-1 bg-transparent">
                        <Settings className="h-4 w-4 mr-2" />
                        Configure
                      </Button>
                      <Button variant="ghost" size="sm">
                        <Eye className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
