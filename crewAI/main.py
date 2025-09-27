# filename: crew_real_estate_router.py
# ------------------------------------------------------------
# Minimal CrewAI router for Tenant/Landlord vs Property/Zoning
# - Router agent (LLM) + deterministic fallback
# - Query refiner agent (LLM) to optimize for RAG
# - Domain "agents" that call your prebuilt RAG functions
# - Tools use CrewAI @tool decorator (fixes Pydantic errors)
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Literal
import json
import re
import os
from textwrap import dedent

from crewai import Agent, Task, Crew
from crewai.tools import tool

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


# =========================
# 1) YOUR RAG ENTRYPOINTS
# =========================
# Replace these with your real implementations (imports).
def rag_tenant_landlord(query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    # << Call your TL RAG pipeline here >>
    return {
        "domain": "tenant_landlord",
        "query": query,
        "meta": metadata,
        "result": "<<<Tenant/Landlord RAG ANSWER>>>",
    }

def rag_property_zoning(query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    # << Call your Zoning RAG pipeline here >>
    return {
        "domain": "property_zoning",
        "query": query,
        "meta": metadata,
        "result": "<<<Property/Zoning RAG ANSWER>>>",
    }

# =========================
# 2) HEURISTIC HELPERS
# =========================
def _extract_geo_hints(user_query: str) -> Dict[str, Optional[str]]:
    lower = user_query.lower()
    state = city = zipc = None

    for st in ["ca", "ny", "wa", "tx", "fl", "il", "ma", "nj", "pa", "az", "va", "co", "or"]:
        if f" {st} " in f" {lower} ":
            state = st.upper()
            break

    for c in [
        "seattle", "new york", "austin", "miami", "phoenix", "boston",
        "chicago", "los angeles", "san francisco", "richmond", "arlington",
        "fairfax", "houston", "dallas", "denver", "portland"
    ]:
        if c in lower:
            city = c.title()
            break

    m = re.search(r"\b(\d{5})(-\d{4})?\b", user_query)
    if m:
        zipc = m.group(1)

    return {"state": state, "city": city, "zip": zipc}

def _router_structured(user_query: str) -> Dict[str, Any]:
    text = user_query.lower()
    tl_kw = [
        "tenant", "landlord", "lease", "rent", "notice", "evict", "eviction",
        "security deposit", "habitability", "rent control", "nonpayment",
        "just cause", "termination", "unlawful detainer"
    ]
    z_kw = [
        "zoning", "setback", "str", "short-term", "airbnb", "vrbo", "adu",
        "permitted use", "parking", "balcony", "lot coverage", "parcel", "apn",
        "yard", "overlay", "variance", "historic district"
    ]
    tl_hit = any(k in text for k in tl_kw)
    z_hit  = any(k in text for k in z_kw)

    if tl_hit and not z_hit:
        domain = "tenant_landlord"
    elif z_hit and not tl_hit:
        domain = "property_zoning"
    elif tl_hit and z_hit:
        site_terms = ["setback", "balcony", "lot", "parcel", "apn", "adu", "str", "short-term", "overlay"]
        domain = "property_zoning" if any(t in text for t in site_terms) else "tenant_landlord"
    else:
        domain = None  # let LLM decide

    return {"domain": domain, "geo_hints": _extract_geo_hints(user_query)}

def _refine_query_for_domain(user_query: str, domain: str, geo_hints: Dict[str, Optional[str]]) -> Dict[str, Any]:
    base = user_query.strip()
    state, city, zipc = geo_hints.get("state"), geo_hints.get("city"), geo_hints.get("zip")
    jurisdiction = []
    if city:  jurisdiction.append(f"city:{city}")
    if state: jurisdiction.append(f"state:{state}")
    if zipc:  jurisdiction.append(f"zip:{zipc}")

    if domain == "tenant_landlord":
        synonyms = [
            "notice period", "security deposit", "habitability", "repair and deduct",
            "rent control", "nonpayment eviction", "just cause", "lease termination"
        ]
        filters  = ["doc_type:statute", "doc_type:ordinance", "doc_type:guidance", "jurisdiction_level:state,city"]
        optimized = f"""{base}

Focus:
- {", ".join(synonyms)}
Filters:
- {", ".join(filters)}
Jurisdiction:
- {", ".join(jurisdiction) if jurisdiction else "unspecified"}
Output:
- Timelines, thresholds, required notices; cite section/page/effective_date."""
    else:
        synonyms = [
            "zoning district", "setback", "yard encroachment", "lot coverage",
            "permitted use", "short-term rental", "parking minimum", "variances",
            "guardrail height", "balcony projection"
        ]
        filters  = ["doc_type:zoning_code", "doc_type:building_code", "doc_type:short_term_rental", "jurisdiction_level:city,county"]
        optimized = f"""{base}

Focus:
- {", ".join(synonyms)}
Filters:
- {", ".join(filters)}
Jurisdiction:
- {", ".join(jurisdiction) if jurisdiction else "unspecified"}
Output:
- Pass/Warn/Fail + reasons; cite section/page/effective_date; next steps/permits."""

    return {
        "optimized_query": optimized,
        "metadata": {
            "domain": domain,
            "geo": geo_hints,
            "required_citations": True,
            "require_effective_dates": True,
        },
    }

# =========================
# 3) CREWAI COMPATIBLE TOOLS
# =========================
@tool("router_structured")
def router_tool(user_query: str) -> str:
    """Classify query into tenant_landlord or property_zoning and extract geo hints."""
    result = _router_structured(user_query)
    return json.dumps(result)

@tool("extract_geo_hints")
def geo_hints_tool(user_query: str) -> str:
    """Extract state/city/zip hints from free-text query."""
    result = _extract_geo_hints(user_query)
    return json.dumps(result)

@tool("refine_query_for_domain")
def refine_tool(user_query: str, domain: str, geo_hints: str) -> str:
    """Remodel/optimize a query for the chosen domain with synonyms, filters, outputs.
    
    Args:
        user_query: Original end-user query
        domain: Target domain - 'tenant_landlord' or 'property_zoning'
        geo_hints: JSON string containing jurisdiction hints with keys: state/city/zip
    """
    try:
        geo_hints_dict = json.loads(geo_hints) if isinstance(geo_hints, str) else geo_hints
    except:
        geo_hints_dict = {}
    
    result = _refine_query_for_domain(user_query, domain, geo_hints_dict)
    return json.dumps(result)

# =========================
# 4) LLM SETUP - FIXED TO USE SAME AS TRIAL SCRIPT
# =========================
def _make_llms():
    """Create LLM instances using the same Google Genai setup as the working trial script"""
    from google import genai
    
    # Check API key just like in the trial script
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("❌ ERROR: The GOOGLE_API_KEY was not found in the environment (or .env file).")
    
    print("✅ GOOGLE_API_KEY successfully loaded from .env for CrewAI.")
    
    # Create a custom LLM wrapper that uses google.genai directly
    class GoogleGenaiLLM:
        def __init__(self, model_name="gemini-2.5-flash", temperature=0.2):
            self.client = genai.Client()  # Uses GOOGLE_API_KEY from environment automatically
            self.model_name = model_name
            self.temperature = temperature
            
        def invoke(self, messages, **kwargs):
            """CrewAI expects this method"""
            # Convert messages to a simple string prompt
            if isinstance(messages, list):
                prompt = "\n".join([str(msg.content) if hasattr(msg, 'content') else str(msg) for msg in messages])
            else:
                prompt = str(messages)
                
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                
                # Return in format expected by CrewAI
                class MockResponse:
                    def __init__(self, text):
                        self.content = text
                        
                return MockResponse(response.text.strip())
                
            except Exception as e:
                print(f"❌ Google Genai API call failed: {e}")
                # Return fallback response
                class MockResponse:
                    def __init__(self, text):
                        self.content = text
                return MockResponse(f"Error: Could not get response from model. {str(e)}")
        
        def __call__(self, prompt, **kwargs):
            """Alternative calling method"""
            return self.invoke(prompt, **kwargs)
    
    # Create three instances (router, refiner, domain)
    router_llm = GoogleGenaiLLM("gemini-2.5-flash", 0.2)
    refiner_llm = GoogleGenaiLLM("gemini-2.5-flash", 0.2) 
    domain_llm = GoogleGenaiLLM("gemini-2.5-flash", 0.2)
    
    return router_llm, refiner_llm, domain_llm

RouterLLM, RefinerLLM, DomainLLM = _make_llms()

# =========================
# 5) AGENTS
# =========================
def make_router_agent() -> Agent:
    return Agent(
        role="Compliance Router (Central)",
        backstory=dedent("""\
            You are the central routing analyst for a real-estate compliance assistant.
            You deeply understand two domains:
            (1) Tenant/Landlord regulations (statutes, notices, rent control, eviction timelines),
            (2) Property/Zoning (zoning codes, setbacks, STR rules, permitted uses).
            You classify the user's intent and jurisdiction, then hand off to the correct domain agent.
            You do not answer the legal question yourself; you route and prepare the best query.
        """),
        goal=dedent("""\
            Classify the incoming query as 'tenant_landlord' or 'property_zoning',
            extract jurisdiction hints (city/state/zip), and output a routing packet:
            {domain, geo_hints, rationale}. If ambiguous, choose the best domain.
        """),
        tools=[router_tool, geo_hints_tool],
        llm=RouterLLM,
        verbose=True,
        allow_delegation=False,
    )

def make_query_refiner_agent() -> Agent:
    return Agent(
        role="Query Normalizer",
        backstory=dedent("""\
            You reshape a user's raw query into an optimal retrieval query:
            add synonyms, expand acronyms, inject jurisdiction filters, and specify desired outputs.
            Your rewrites are faithful, not speculative, to maximize RAG precision & recall.
        """),
        goal=dedent("""\
            Given a domain and geo hints, produce a remodeled query + metadata for precise retrieval.
        """),
        tools=[refine_tool],
        llm=RefinerLLM,
        verbose=True,
        allow_delegation=False,
    )

def make_tenant_landlord_agent() -> Agent:
    return Agent(
        role="Tenant/Landlord Compliance Analyst",
        backstory=dedent("""\
            Specialist in landlord-tenant statutes, rent control ordinances, notice periods,
            habitability standards, and eviction timelines. Always answer strictly from RAG.
        """),
        goal=dedent("""\
            Produce accurate, citation-backed answers for the correct jurisdiction.
            Include timelines, thresholds, and required notices where applicable.
        """),
        llm=DomainLLM,
        verbose=True,
        allow_delegation=False,
    )

def make_property_zoning_agent() -> Agent:
    return Agent(
        role="Property & Zoning Compliance Analyst",
        backstory=dedent("""\
            Specialist in zoning/building codes: setbacks, encroachments, lot coverage,
            permitted uses, STR rules, parking minimums, and overlays. Always answer from RAG.
        """),
        goal=dedent("""\
            Provide Pass/Warn/Fail with reasons, next steps (permits/variances),
            and precise citations/effective dates for the relevant jurisdictional code.
        """),
        llm=DomainLLM,
        verbose=True,
        allow_delegation=False,
    )

# =========================
# 6) TASK BUILDERS
# =========================
def make_route_task(router: Agent, user_query: str) -> Task:
    return Task(
        description=dedent(f"""
        **Task**: Route Query to Correct Domain
        **Instructions**:
        1) Call tool `router_structured` with the user query: {json.dumps(user_query)}
        2) Use your own reasoning only if domain is null or ambiguous.
        **Return ONLY JSON**:
        {{
          "domain": "tenant_landlord" | "property_zoning",
          "geo_hints": {{"state":"..","city":"..","zip":".."}},
          "rationale": "one-sentence why"
        }}
        """),
        agent=router,
        expected_output="JSON object with domain classification and geo hints"
    )

def make_refine_task(refiner: Agent, domain: str, user_query: str, geo_hints: Dict[str, Optional[str]]) -> Task:
    return Task(
        description=dedent(f"""
        **Task**: Remodel Query for Retrieval
        **Instructions**:
        Call `refine_query_for_domain` with:
        - user_query: {json.dumps(user_query)}
        - domain: {domain}
        - geo_hints: {json.dumps(json.dumps(geo_hints))}  # Double-encoded for string parameter
        
        **Return ONLY the tool's JSON** with keys: optimized_query, metadata.
        """),
        agent=refiner,
        expected_output="JSON object with optimized query and metadata"
    )

def make_execute_task(domain_agent: Agent, domain: str, optimized_query: str, metadata: Dict[str, Any]) -> Task:
    # Marker task for auditability (we actually call the Python function directly below).
    return Task(
        description=dedent(f"""
        **Task**: Execute Domain RAG (marker)
        **Inputs**:
        - Domain: {domain}
        - Optimized Query: {json.dumps(optimized_query)}
        - Metadata: {json.dumps(metadata)}
        Do not re-route or re-refine.
        """),
        agent=domain_agent,
        expected_output="Acknowledgment that RAG execution is handled externally"
    )

# =========================
# 7) CREW RUNNER
# =========================
@dataclass
class RealEstateRouterCrew:
    tl_func: Callable[[str, Dict[str, Any]], Any] = rag_tenant_landlord
    z_func: Callable[[str, Dict[str, Any]], Any] = rag_property_zoning

    def run(self, user_query: str) -> Dict[str, Any]:
        router = make_router_agent()
        refiner = make_query_refiner_agent()
        tl_agent = make_tenant_landlord_agent()
        z_agent  = make_property_zoning_agent()

        # --- 1) ROUTE ---
        try:
            crew_route = Crew(agents=[router], tasks=[make_route_task(router, user_query)], verbose=True)
            route_out = str(crew_route.kickoff())
            
            packet = self._extract_json(route_out)
            if not packet or "domain" not in packet:
                # deterministic fallback
                fallback = _router_structured(user_query)
                domain = fallback["domain"] or "property_zoning"
                geo_hints = fallback["geo_hints"]
                rationale = "Fallback routing by heuristic."
            else:
                domain = packet.get("domain") or "property_zoning"
                geo_hints = packet.get("geo_hints", {})
                rationale = packet.get("rationale", "")
        except Exception as e:
            print(f"Router failed, using fallback: {e}")
            fallback = _router_structured(user_query)
            domain = fallback["domain"] or "property_zoning"
            geo_hints = fallback["geo_hints"]
            rationale = "Fallback routing due to error."

        # --- 2) REFINE ---
        try:
            crew_refine = Crew(
                agents=[refiner],
                tasks=[make_refine_task(refiner, domain, user_query, geo_hints)],
                verbose=True,
            )
            refined_out = str(crew_refine.kickoff())
            refined_packet = self._extract_json(refined_out) or {}
            optimized_query = refined_packet.get("optimized_query", user_query)
            metadata = refined_packet.get("metadata", {"domain": domain, "geo": geo_hints})
        except Exception as e:
            print(f"Refiner failed, using fallback: {e}")
            refined_packet = _refine_query_for_domain(user_query, domain, geo_hints)
            optimized_query = refined_packet.get("optimized_query", user_query)
            metadata = refined_packet.get("metadata", {"domain": domain, "geo": geo_hints})

        # --- 3) EXECUTE (call your RAG) ---
        if domain == "tenant_landlord":
            domain_agent = tl_agent
            result = self.tl_func(optimized_query, metadata)
        else:
            domain_agent = z_agent
            result = self.z_func(optimized_query, metadata)

        # Optional: marker task for audit trace
        try:
            crew_exec = Crew(agents=[domain_agent], tasks=[make_execute_task(domain_agent, domain, optimized_query, metadata)], verbose=True)
            _ = crew_exec.kickoff()
        except Exception as e:
            print(f"Execute task failed (non-critical): {e}")

        return {
            "routing": {"domain": domain, "geo_hints": geo_hints, "rationale": rationale},
            "refined": {"optimized_query": optimized_query, "metadata": metadata},
            "result": result,
        }

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        try:
            # Try to find JSON in the text
            json_match = re.search(r'\{[^}]*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            # If no braces found, try parsing the whole text
            return json.loads(text.strip())
        except Exception:
            return None

# =========================
# 8) SIMPLE CLI
# =========================
if __name__ == "__main__":
    print("=== Real-Estate Compliance Router (CrewAI) ===")
    print("Uses Gemini 2.5 Flash with native google.genai client.")
    print("Press Ctrl+C to exit.\n")

    while True:
        try:
            q = input("User: ").strip()
            if not q:
                continue
            app = RealEstateRouterCrew()
            out = app.run(q)
            print("\n--- ROUTING ---")
            print(json.dumps(out["routing"], indent=2))
            print("\n--- REFINED QUERY ---")
            print(out["refined"]["optimized_query"])
            print("\n--- ANSWER (RAG) ---")
            print(json.dumps(out["result"], indent=2))
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")