"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { MessageCircle, Users, HelpCircle, Building2, Send, ArrowLeft, ImageIcon, Loader2 } from "lucide-react"

// Add API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

// Add types for API response
interface APIResponse {
  result: {
    domain: string
    query: string
    meta: any
    result: string
  }
  routing: {
    domain: string
    geo_hints: {
      state: string | null
      city: string | null
      zip: string | null
    }
    rationale: string
  }
  refined_query: string
}

export default function HomePage() {
  const [currentView, setCurrentView] = useState<string>("home")
  const [selectedOption, setSelectedOption] = useState<string | null>(null)
  const [messages, setMessages] = useState<Array<{ id: number; text: string; sender: "user" | "ai"; image?: string; isLoading?: boolean }>>(
    [],
  )
  const [inputText, setInputText] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const options = [
    {
      id: "speak",
      title: "Speak",
      description: "Ask your questions...",
      icon: MessageCircle,
      color: "bg-card hover:bg-accent",
    },
    {
      id: "peek",
      title: "Peek",
      description: "Know our story",
      icon: Users,
      color: "bg-card hover:bg-accent",
    },
    {
      id: "seek",
      title: "Seek",
      description: "Still need help? Reach us",
      icon: HelpCircle,
      color: "bg-card hover:bg-accent",
    },
  ]

  const handleOptionSelect = (optionId: string) => {
    setSelectedOption(optionId)
    setCurrentView(optionId)
  }

  // Updated handleSendMessage function to call your API
  const handleSendMessage = async () => {
    if (inputText.trim() && !isLoading) {
      const userMessage = {
        id: Date.now(),
        text: inputText,
        sender: "user" as const,
      }
      
      // Add user message immediately
      setMessages((prev) => [...prev, userMessage])
      
      // Add loading message
      const loadingMessage = {
        id: Date.now() + 1,
        text: "Analyzing your compliance question...",
        sender: "ai" as const,
        isLoading: true,
      }
      setMessages((prev) => [...prev, loadingMessage])
      
      const currentQuery = inputText
      setInputText("")
      setIsLoading(true)

      try {
        // Call your FastAPI endpoint
        const response = await fetch(`${API_BASE_URL}/query`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text: currentQuery
          })
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const data: APIResponse = await response.json()
        
        // Remove loading message and add AI response
        setMessages((prev) => {
          const withoutLoading = prev.filter(msg => !msg.isLoading)
          
          // Format the response nicely
          let formattedResponse = `**Domain:** ${data.routing.domain.replace('_', ' ').toUpperCase()}\n\n`
          
          if (data.routing.geo_hints.state || data.routing.geo_hints.city) {
            formattedResponse += `**Location:** ${[data.routing.geo_hints.city, data.routing.geo_hints.state].filter(Boolean).join(', ')}\n\n`
          }
          
          formattedResponse += `**Analysis:**\n${data.result.result}\n\n`
          formattedResponse += `**Routing Rationale:** ${data.routing.rationale}`
          
          const aiResponse = {
            id: Date.now() + 2,
            text: formattedResponse,
            sender: "ai" as const,
          }
          
          return [...withoutLoading, aiResponse]
        })

      } catch (error) {
        console.error('API call failed:', error)
        
        // Remove loading message and add error response
        setMessages((prev) => {
          const withoutLoading = prev.filter(msg => !msg.isLoading)
          const errorResponse = {
            id: Date.now() + 2,
            text: `I apologize, but I'm having trouble connecting to the compliance analysis service right now. Please try again later or contact support if the issue persists.\n\nError: ${error instanceof Error ? error.message : 'Unknown error'}`,
            sender: "ai" as const,
          }
          return [...withoutLoading, errorResponse]
        })
      } finally {
        setIsLoading(false)
      }
    }
  }

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const imageMessage = {
          id: Date.now(),
          text: "Uploaded an image for analysis",
          sender: "user" as const,
          image: e.target?.result as string,
        }
        setMessages((prev) => [...prev, imageMessage])

        // For now, keep the simulated response for images
        // You can extend your API to handle image analysis later
        setTimeout(() => {
          const aiResponse = {
            id: Date.now() + 1,
            text: "I can see your uploaded image. Image analysis for compliance documents is currently in development. For now, please describe what you'd like me to review about this document, and I can provide compliance guidance based on your description.",
            sender: "ai" as const,
          }
          setMessages((prev) => [...prev, aiResponse])
        }, 1500)
      }
      reader.readAsDataURL(file)
    }
  }

  // Function to render formatted text (handles markdown-style formatting)
  const renderFormattedText = (text: string) => {
    const lines = text.split('\n')
    return lines.map((line, index) => {
      if (line.startsWith('**') && line.endsWith('**')) {
        return <div key={index} className="font-semibold mb-2">{line.slice(2, -2)}</div>
      } else if (line.trim() === '') {
        return <br key={index} />
      } else {
        return <div key={index} className="mb-1">{line}</div>
      }
    })
  }

  if (currentView === "speak") {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <div className="container mx-auto px-6 py-8">
          <div className="flex items-center gap-4 mb-6">
            <Button variant="ghost" onClick={() => setCurrentView("home")}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
            <h1 className="text-2xl font-bold">AI Compliance Chat</h1>
          </div>

          <Card className="h-[600px] flex flex-col">
            <div className="flex-1 p-6 overflow-y-auto space-y-4">
              {messages.length === 0 ? (
                <div className="text-center text-muted-foreground py-12">
                  <MessageCircle className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p className="text-lg font-medium">Ask your compliance question...</p>
                  <p className="text-sm mt-2">
                    Ask about MLS rules, tenant/landlord regulations, transaction disclosures, or AML/fraud compliance
                  </p>
                </div>
              ) : (
                messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[80%] p-4 rounded-lg ${
                        message.sender === "user"
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted text-muted-foreground"
                      }`}
                    >
                      {message.image && (
                        <img
                          src={message.image || "/placeholder.svg"}
                          alt="Uploaded"
                          className="max-w-full h-auto rounded mb-2"
                        />
                      )}
                      
                      {message.isLoading ? (
                        <div className="flex items-center gap-2">
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span>{message.text}</span>
                        </div>
                      ) : (
                        <div className="whitespace-pre-wrap">
                          {renderFormattedText(message.text)}
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>

            <div className="border-t p-4">
              <div className="flex gap-2">
                <Button 
                  variant="outline" 
                  size="icon" 
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isLoading}
                >
                  <ImageIcon className="w-4 h-4" />
                </Button>
                <Input
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Ask about compliance requirements..."
                  onKeyPress={(e) => e.key === "Enter" && !e.shiftKey && handleSendMessage()}
                  className="flex-1"
                  disabled={isLoading}
                />
                <Button 
                  onClick={handleSendMessage} 
                  disabled={isLoading || !inputText.trim()}
                >
                  {isLoading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                </Button>
              </div>
              <div className="text-xs text-muted-foreground mt-2">
                Press Enter to send, Shift+Enter for new line
              </div>
              <input 
                ref={fileInputRef} 
                type="file" 
                accept="image/*" 
                onChange={handleImageUpload} 
                className="hidden" 
              />
            </div>
          </Card>
        </div>
      </div>
    )
  }

  if (currentView === "peek") {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <div className="container mx-auto px-6 py-8">
          <div className="flex items-center gap-4 mb-6">
            <Button variant="ghost" onClick={() => setCurrentView("home")}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
            <h1 className="text-2xl font-bold">Know Our Story</h1>
          </div>

          <div className="max-w-4xl mx-auto space-y-8">
            <Card className="p-8">
              <h2 className="text-3xl font-bold mb-6">About TERRACOMPLY.AI</h2>
              <div className="space-y-6 text-lg">
                <p>
                  TERRACOMPLY.AI revolutionizes real estate compliance with cutting-edge artificial intelligence. Our
                  platform streamlines complex regulatory processes, making compliance accessible and efficient for real
                  estate professionals.
                </p>
                <p>
                  Built with CrewAI for intelligent routing and multimodal analysis, we provide specialized AI agents
                  that understand MLS rules, zoning codes, transaction requirements, and regulatory frameworks.
                </p>
                <p>
                  Our mission is to eliminate compliance complexity, reduce regulatory risks, and empower real estate
                  professionals with AI-powered insights that ensure every transaction meets the highest standards.
                </p>
              </div>
            </Card>

            <div className="grid md:grid-cols-2 gap-6">
              <Card className="p-6">
                <h3 className="text-xl font-semibold mb-4">Our Technology</h3>
                <ul className="space-y-2 text-muted-foreground">
                  <li>• CrewAI-powered intelligent routing</li>
                  <li>• Multimodal document analysis</li>
                  <li>• Real-time compliance checking</li>
                  <li>• Automated regulatory updates</li>
                </ul>
              </Card>

              <Card className="p-6">
                <h3 className="text-xl font-semibold mb-4">Our Focus</h3>
                <ul className="space-y-2 text-muted-foreground">
                  <li>• MLS rules and regulations</li>
                  <li>• Zoning code compliance</li>
                  <li>• Transaction review automation</li>
                  <li>• Regulatory requirement tracking</li>
                </ul>
              </Card>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (currentView === "seek") {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <div className="container mx-auto px-6 py-8">
          <div className="flex items-center gap-4 mb-6">
            <Button variant="ghost" onClick={() => setCurrentView("home")}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
            <h1 className="text-2xl font-bold">Still Need Help? Reach Us</h1>
          </div>

          <div className="max-w-2xl mx-auto space-y-8">
            <Card className="p-8">
              <h2 className="text-2xl font-bold mb-6">Get Support</h2>
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Contact Information</h3>
                  <p className="text-muted-foreground">Email: support@terracomply.ai</p>
                  <p className="text-muted-foreground">Phone: 1-800-COMPLY</p>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-4">Send us a message</h3>
                  <div className="space-y-4">
                    <Input placeholder="Your Name" />
                    <Input placeholder="Your Email" />
                    <Textarea placeholder="How can we help you with compliance?" rows={4} />
                    <Button className="w-full">Send Message</Button>
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Frequently Asked Questions</h3>
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium">How does AI compliance checking work?</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    Our AI agents analyze documents and transactions against current regulatory requirements in
                    real-time.
                  </p>
                </div>
                <div>
                  <h4 className="font-medium">What types of compliance do you cover?</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    We specialize in MLS rules, zoning codes, transaction reviews, and regulatory requirements.
                  </p>
                </div>
                <div>
                  <h4 className="font-medium">Is my data secure?</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    Yes, we use enterprise-grade security and comply with all data protection regulations.
                  </p>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background text-foreground relative overflow-hidden">
      <div className="absolute inset-0 pointer-events-none">
        <div className="sketch-buildings-container">
          <div className="sketch-building sketch-building-1"></div>
          <div className="sketch-building sketch-building-2"></div>
          <div className="sketch-building sketch-building-3"></div>
          <div className="sketch-building sketch-building-4"></div>
          <div className="sketch-building sketch-building-5"></div>
          <div className="sketch-building sketch-building-6"></div>

          <div className="sketch-element sketch-element-1"></div>
          <div className="sketch-element sketch-element-2"></div>
          <div className="sketch-element sketch-element-3"></div>
          <div className="sketch-element sketch-element-4"></div>
          <div className="sketch-element sketch-element-5"></div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-12 relative z-10">
        <div className="text-center mb-16">
          <div className="flex items-center justify-center gap-3 mb-6">
            <Building2 className="w-8 h-8 text-foreground animate-pulse-gentle" />
            <h1 className="text-4xl font-bold text-balance">TERRACOMPLY.AI</h1>
          </div>
          <p className="text-xl text-muted-foreground text-pretty max-w-2xl mx-auto">
            AI-powered real estate compliance made simple
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto">
          {options.map((option) => {
            const Icon = option.icon
            return (
              <Card
                key={option.id}
                className={`p-8 transition-all duration-300 cursor-pointer border-2 ${
                  selectedOption === option.id
                    ? "border-primary shadow-lg scale-105"
                    : "border-border hover:border-primary/50"
                } ${option.color} hover:animate-card-lift`}
                onClick={() => handleOptionSelect(option.id)}
              >
                <div className="text-center space-y-4">
                  <div className="w-16 h-16 mx-auto bg-primary/10 rounded-full flex items-center justify-center animate-icon-float">
                    <Icon className="w-8 h-8 text-primary" />
                  </div>
                  <h3 className="text-2xl font-semibold text-card-foreground">{option.title}</h3>
                  <p className="text-muted-foreground text-pretty">{option.description}</p>
                  <Button variant="outline" className="w-full mt-4 bg-transparent">
                    Choose
                  </Button>
                </div>
              </Card>
            )
          })}
        </div>
      </div>
    </div>
  )
}