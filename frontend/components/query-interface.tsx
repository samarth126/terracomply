import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Upload, Send, ImageIcon, FileText, Mic, Sparkles, ArrowRight } from "lucide-react"

const exampleQueries = [
  {
    type: "Text Query",
    icon: FileText,
    query: "Can I build a balcony on this property according to local zoning rules?",
    model: "Property & Zoning",
  },
  {
    type: "Image Analysis",
    icon: ImageIcon,
    query: "Upload property photo to check setback compliance",
    model: "Multimodal Analysis",
  },
  {
    type: "Document Review",
    icon: Upload,
    query: "Review this purchase contract for missing disclosures",
    model: "Transaction Review",
  },
]

export function QueryInterface() {
  return (
    <section id="query" className="py-24 px-6 bg-card/30">
      <div className="container mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold mb-4">Intelligent Query Interface</h2>
          <p className="text-xl text-muted-foreground text-balance max-w-3xl mx-auto">
            {
              "Submit text, images, or documents. Our central AI agent routes your query to the most appropriate specialized compliance model."
            }
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {/* Query Input */}
          <Card className="gradient-border">
            <div className="gradient-border-content">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 mt-4">
                  <Sparkles className="h-5 w-5 text-primary" />
                  Submit Your Query
                </CardTitle>
                <CardDescription className="mb-0.5">
                  Ask questions about compliance, upload property images, or submit documents for review.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  placeholder="Example: Can I build a balcony on this property? What zoning restrictions apply?"
                  className="min-h-[120px] resize-none"
                />

                <div className="flex flex-wrap gap-2">
                  <Button variant="outline" size="sm">
                    <Upload className="h-4 w-4 mr-2" />
                    Upload Image
                  </Button>
                  <Button variant="outline" size="sm">
                    <FileText className="h-4 w-4 mr-2" />
                    Upload Document
                  </Button>
                  <Button variant="outline" size="sm">
                    <Mic className="h-4 w-4 mr-2" />
                    Voice Input
                  </Button>
                </div>

                <div className="pt-4 border-t border-border">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 mb-4">
                      <Badge variant="secondary" className="text-xs">
                        Auto-routing enabled
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        Multimodal ready
                      </Badge>
                    </div>
                    <Button className="mb-4">
                      <Send className="h-4 w-4 mr-2" />
                      Submit Query
                    </Button>
                  </div>
                </div>
              </CardContent>
            </div>
          </Card>

          {/* Example Queries */}
          <div className="space-y-4">
            <h3 className="text-xl font-semibold mb-4">Example Queries</h3>
            {exampleQueries.map((example, index) => {
              const Icon = example.icon
              return (
                <Card key={index} className="hover:bg-card/80 transition-colors cursor-pointer">
                  <CardContent className="p-4">
                    <div className="flex items-start gap-3">
                      <div className="p-2 bg-primary/10 rounded-lg">
                        <Icon className="h-4 w-4 text-primary" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <Badge variant="outline" className="text-xs">
                            {example.type}
                          </Badge>
                          <ArrowRight className="h-3 w-3 text-muted-foreground" />
                          <Badge variant="secondary" className="text-xs">
                            {example.model}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">{example.query}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </div>
      </div>
    </section>
  )
}
