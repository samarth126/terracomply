# filename: crew_real_estate_router.py
# ------------------------------------------------------------
# Minimal CrewAI router for:
# - Tenant/Landlord
# - Listing/MLS Compliance
# - Transaction/Disclosures (VA)
# - AML / Fraud (FinCEN real estate rule)
# - Router agent (LLM) + deterministic fallback
# - Query refiner agent (LLM) to optimize for RAG
# - Domain "agents" that call your prebuilt RAG functions
# - Tools use CrewAI @tool decorator (fixes Pydantic errors)
# ------------------------------------------------------------



from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional
import json
import re
import os
from textwrap import dedent

from crewai import Agent, Task, Crew
from crewai.tools import tool

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from tenant_compliance.query_script import query_ollama_tenant_RAG, load_tenant_db
from Deal_listing_agent.query_script import query_ollama_deal_listing, load_listing_db
from residential_property.query_script import query_ollama_residential_property, load_residential_db
from AML_inference import query_fraud_model



from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# =========================
# 1) YOUR RAG ENTRYPOINTS
# =========================
# Replace these with your real implementations (imports).
def rag_tenant_landlord(query: str, metadata: Dict[str, Any], 
                       index_dir: str = "tenant_compliance/faiss_index/", 
                       embed_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                       ollama_model: str = "qwen2.5:3b", 
                       k: int = 3) -> Dict[str, Any]:
    """
    RAG pipeline for tenant/landlord compliance queries
    """
    # Load the tenant compliance FAISS database
    db, *_ = load_tenant_db(index_dir=index_dir, model_name=embed_model)
    
    # Query Ollama with the loaded db
    result = query_ollama_tenant_RAG(db, question=query, k=k, model=ollama_model)
    
    return {
        "domain": "tenant_landlord",
        "query": query,
        "meta": metadata,
        "result": result,
    }

def rag_listing_compliance(query: str, metadata: Dict[str, Any],
                          index_dir: str = "Deal_listing_agent/faiss_index/", 
                          embed_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                          ollama_model: str = "qwen2.5:3b", 
                          k: int = 3) -> Dict[str, Any]:
    """
    RAG pipeline for MLS/Listing Compliance queries
    """
    # Load the deal listing FAISS database
    db, *_ = load_listing_db(index_dir=index_dir, model_name=embed_model)
    
    # Query Ollama with the loaded db
    result = query_ollama_deal_listing(db, question=query, k=k, model=ollama_model)
    
    return {
        "domain": "listing_compliance",
        "query": query,
        "meta": metadata,
        "result": result,
    }

def rag_transaction_disclosures(query: str, metadata: Dict[str, Any],
                               index_dir: str = "residential_property/faiss_index/", 
                               embed_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                               ollama_model: str = "qwen2.5:3b", 
                               k: int = 3) -> Dict[str, Any]:
    """
    RAG pipeline for Transaction/Disclosures queries
    """
    # Load the residential property FAISS database
    db, *_ = load_residential_db(index_dir=index_dir, model_name=embed_model)
    
    # Query Ollama with the loaded db
    result = query_ollama_residential_property(db, question=query, k=k, model=ollama_model)
    
    return {
        "domain": "transaction_disclosures",
        "query": query,
        "meta": metadata,
        "result": result,
    }


def rag_aml_fraud(query: str, metadata: Dict[str, Any], 
                  model: str = "fraud_detect_llama:latest") -> Dict[str, Any]:
    """
    RAG pipeline for AML/FinCEN fraud detection queries
    Focus: FinCEN final rule (31 CFR Chapter X; RIN 1506-AB54) re: non-financed residential real estate
    transfers to entities/trusts (nationwide), reporting parties, contents, timing, retention, exceptions.
    """
    # Query the fraud detection model
    result = query_fraud_model(query, model=model)
    
    return {
        "domain": "aml_fraud",
        "query": query + "Help me",
        "meta": metadata,
        "result": result if result else "Error: Unable to get response from fraud detection model",
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

    # Tenant/Landlord indicators
    tl_kw = [
        "tenant", "landlord", "lease", "rent", "notice", "evict", "eviction",
        "security deposit", "habitability", "rent control", "nonpayment",
        "just cause", "termination", "unlawful detainer"
    ]

    # Listing/Dealing Compliance (MLS/IDX/VOW/Fair Housing/Lockbox) indicators
    lc_kw = [
        "mls", "listing compliance", "idx", "vow", "clear cooperation",
        "coming soon", "delayed entry", "photo entry date", "virtual staging",
        "accuracy of listing data", "listing duplication", "compensation terms",
        "structured compensation", "sentri", "sentrilock", "lockbox",
        "fine", "appeal", "compliance audit", "fair housing", "discriminatory ad",
        "remarks field", "open house information", "co-branding", "avm",
        "refresh listing information", "password use", "participant", "subscriber",
        "office exclusive", "delayed showing", "realtor code of ethics",
        "idx display", "vow policy", "marketing before mls", "coming soon status"
    ]

    # Transaction Review / Disclosures / Contract Addenda indicators (VA focus)
    td_kw = [
        "disclosure", "property disclosure", "virginia residential property disclosure act",
        "55.1-700", "55.1-701", "reb", "real estate board", "dpor",
        "electronic delivery", "notification", "ratification", "contract addendum",
        "addenda", "lead-based paint", "hoa disclosure", "poa disclosure", "septic addendum",
        "radon", "well", "as-is", "buyer beware", "seller disclosure", "acknowledgment",
        "receipt of disclosures", "rescission period", "delivery of disclosures",
        "real estate contract"
    ]

    # AML / Fraud (FinCEN) indicators
    aml_kw = [
        "fincen", "31 cfr", "chapter x", "bank secrecy act", "bsa",
        "aml", "anti-money laundering", "cft", "countering the financing of terrorism",
        "geographic targeting order", "gto", "non-financed transfer", "non financed transfer",
        "beneficial owner", "beneficial ownership", "entity buyer", "legal entity", "trust buyer",
        "nationwide rule", "reporting party", "closing agent", "settlement agent",
        "report due", "recordkeeping", "retention", "covered transaction", "exemption",
        "rin 1506-ab54", "residential real estate transfers"
    ]

    tl_hit  = any(k in text for k in tl_kw)
    lc_hit  = any(k in text for k in lc_kw)
    td_hit  = any(k in text for k in td_kw)
    aml_hit = any(k in text for k in aml_kw)

    # Decision priority:
    # - AML keywords → aml_fraud (these are federal, specialized)
    # - Transaction/Disclosures (VA) → transaction_disclosures
    # - MLS/IDX/VOW → listing_compliance
    # - Tenant/Landlord → tenant_landlord
    if aml_hit:
        domain = "aml_fraud"
    elif td_hit:
        domain = "transaction_disclosures"
    elif lc_hit:
        domain = "listing_compliance"
    elif tl_hit:
        domain = "tenant_landlord"
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
    elif domain == "listing_compliance":
        synonyms = [
            "MLS rules", "IDX display", "VOW policy", "Clear Cooperation Policy",
            "Coming Soon terms", "photo entry date", "virtual staging rules",
            "accuracy of listing data", "remarks field restrictions",
            "structured compensation terms", "office exclusive", "delayed entry/showing",
            "lockbox/SentriLock rules", "fine schedule", "appeal process",
            "fair housing advertising", "discriminatory ads", "AVM/comment policy",
            "refresh listing information", "listing duplication", "co-branding requirements"
        ]
        filters  = [
            "doc_type:mls_rules", "doc_type:policy", "doc_type:idx_vow",
            "doc_type:fair_housing", "doc_type:lockbox", "jurisdiction_level:mls,state"
        ]
        optimized = f"""{base}

Focus:
- {", ".join(synonyms)}
Filters:
- {", ".join(filters)}
Jurisdiction:
- {", ".join(jurisdiction) if jurisdiction else "unspecified"}
Output:
- Violation? (Yes/No) + Rule/Section cites + cure steps + deadlines + fine range; include effective_date & appendix references (e.g., Fine Chart, arbitration, lockbox)."""
    elif domain == "transaction_disclosures":
        synonyms = [
            "Virginia Residential Property Disclosure Act", "§55.1-700 et seq", "DPOR/REB notice",
            "electronic delivery proof", "notification/acknowledgment", "ratification timing",
            "buyer right to terminate", "lead-based paint disclosure", "HOA/POA disclosure",
            "septic/well addenda", "as-is clause", "property disclaimer vs disclosure",
            "form names/codes", "delivery methods", "recordkeeping"
        ]
        filters = [
            "doc_type:statute", "doc_type:disclosure_form", "doc_type:contract_addendum",
            "doc_type:guidance", "jurisdiction_level:state", "state:VA"
        ]
        optimized = f"""{base}

Focus:
- {", ".join(synonyms)}
Filters:
- {", ".join(filters)}
Jurisdiction:
- {", ".join(jurisdiction) if jurisdiction else "state:VA"}
Output:
- Checklist of required disclosures/addenda; who signs; when due (pre-contract vs by ratification vs post-ratification windows);
  delivery method (incl. electronic delivery) and proof; buyer termination rights & deadlines; exemptions; sample notification language;
  cite Code of Virginia sections (e.g., §55.1-700+) and any REB/DPOR references; include effective_date."""
    else:  # aml_fraud
        synonyms = [
            "FinCEN final rule", "31 CFR Chapter X", "BSA AML/CFT real estate",
            "non-financed residential real estate transfers", "entities and trusts buyers",
            "reporting party (closing/settlement agent)", "report contents and fields",
            "report due date", "recordkeeping/retention", "beneficial ownership info",
            "GTO history", "exemptions (e.g., individual transferees)", "covered transaction scope",
            "nationwide applicability", "penalties for noncompliance", "effective date December 1, 2025"
        ]
        filters = [
            "doc_type:regulation", "doc_type:final_rule", "agency:FinCEN",
            "framework:BSA", "31_CFR:Chapter_X", "jurisdiction_level:federal"
        ]
        optimized = f"""{base}

Focus:
- {", ".join(synonyms)}
Filters:
- {", ".join(filters)}
Jurisdiction:
- federal
Output:
- Does the transfer meet the 'covered' criteria (non-financed, residential, entity/trust buyer)? Who must file (reporting person)?
  What information is required (beneficial owners, transferee/transferor, property, consideration, settlement details)?
  Filing deadline, retention period, exemptions (e.g., individual transferees), and penalties. Include cites to FinCEN final rule (31 CFR Chapter X),
  BSA authority, and effective date (Dec 1, 2025). Provide a practical checklist and red flags."""

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
    """Classify query into tenant_landlord, listing_compliance, transaction_disclosures, or aml_fraud and extract geo hints."""
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
        domain: Target domain - 'tenant_landlord' | 'listing_compliance' | 'transaction_disclosures' | 'aml_fraud'
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
    
    class GoogleGenaiLLM:
        def __init__(self, model_name="gemini-2.5-flash", temperature=0.2):
            self.client = genai.Client()
            self.model_name = model_name
            self.temperature = temperature
            
        def invoke(self, messages, **kwargs):
            if isinstance(messages, list):
                prompt = "\n".join([str(msg.content) if hasattr(msg, 'content') else str(msg) for msg in messages])
            else:
                prompt = str(messages)
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                class MockResponse:
                    def __init__(self, text):
                        self.content = text
                return MockResponse(response.text.strip())
            except Exception as e:
                print(f"❌ Google Genai API call failed: {e}")
                class MockResponse:
                    def __init__(self, text):
                        self.content = text
                return MockResponse(f"Error: Could not get response from model. {str(e)}")
        
        def __call__(self, prompt, **kwargs):
            return self.invoke(prompt, **kwargs)
    
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
            You deeply understand four domains and act as the traffic controller that gets each
            query to the right specialist with the right retrieval context.

            (1) Tenant/Landlord regulations:
                You understand the full lifecycle of residential tenancies and the common disputes that arise.
                - Notices: pay-or-quit, cure-or-quit, termination/non-renewal, and service rules.
                - Rent control & just-cause: allowable increases, exemptions, relocation, and preemption issues.
                - Habitability & repairs: warranty of habitability, repair-and-deduct, entry rights, and emergency fixes.
                - Security deposits: caps, deadlines for return, allowable deductions, and penalties.
                - Enforcement: eviction/unlawful detainer timelines, mediation/ADR, right-to-counsel, and post-judgment.
                - Procedural protections: no lockouts or utility shutoffs, retaliation/harassment bans, and mandatory disclosures.
                - Jurisdictional nuance: distinguish between state statutes, local ordinances, and emergency rules.

            (2) Listing/MLS Compliance:
                MLS Rules, IDX/VOW policies, Clear Cooperation/Coming Soon, listing data accuracy, compensation display,
                lockbox/SentriLock, fair housing in advertising, fines/appeals.

            (3) Transaction Review / Disclosures / Contract Addenda (Virginia focus):
                Virginia Residential Property Disclosure Act (§55.1-700 et seq.), notifications, electronic delivery,
                ratification, buyer termination rights and deadlines, exemptions, forms/addenda, and recordkeeping proof.

            (4) AML / Fraud (FinCEN real estate rule):
                FinCEN final rule under the BSA (31 CFR Chapter X) for non-financed residential real estate transfers
                to entities/trusts, nationwide. You track covered transactions, reporting person, required report fields,
                filing deadlines, record retention, exemptions (e.g., individual transferees), penalties, and effective date.

            You classify the user's intent and jurisdictional hints (city, state, ZIP), then hand off
            to the correct domain agent with a compact routing packet and a retrieval-ready query plan.
            You never opine on the substantive legal outcome yourself; you route and prepare the best query.
        """),
        goal=dedent("""\
            Classify the incoming query as 'tenant_landlord' | 'listing_compliance' | 'transaction_disclosures' | 'aml_fraud',
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

def make_listing_compliance_agent() -> Agent:
    return Agent(
        role="Listing & MLS Compliance Analyst",
        backstory=dedent("""\
            Specialist in Multiple Listing Service (MLS) rules & regulations, IDX/VOW policies,
            Clear Cooperation and Coming Soon requirements, listing data accuracy & duplication controls,
            photo/virtual staging requirements, compensation/offer display standards, lockbox/SentriLock use,
            fines/appeals, and fair housing advertising rules.
            You interpret local MLS rulebooks (e.g., CVR MLS), appendices (fine charts), and related state regs
            (e.g., Virginia Fair Housing 18 VAC 135-50) to determine violations and remediation steps.
            Always answer strictly from RAG with pinpoint citations and effective dates.
        """),
        goal=dedent("""\
            Determine if a fact pattern complies with MLS/IDX/VOW/Fair Housing/Lockbox rules.
            Output: Violation? (Yes/No) + Rule/Section cites + cure steps + deadlines + fine range (if applicable)
            + links/appendix references; include effective_date and any notice/appeal windows.
        """),
        llm=DomainLLM,
        verbose=True,
        allow_delegation=False,
    )

def make_transaction_disclosures_agent() -> Agent:
    return Agent(
        role="Transaction / Disclosures Analyst (Virginia)",
        backstory=dedent("""\
            Specialist in Virginia Residential Property Disclosure Act (§55.1-700 et seq) and companion
            addenda and forms (lead-based paint, HOA/POA disclosures, well/septic, radon, as-is).
            You evaluate required notifications, acknowledgments, delivery (including electronic delivery and proof),
            ratification timing, buyer termination rights and deadlines, exemptions, and recordkeeping.
            You cite Code of Virginia sections and any Real Estate Board (REB/DPOR) guidance.
            Always answer strictly from RAG with precise citations and effective dates.
        """),
        goal=dedent("""\
            Produce a compliance checklist and timeline: which disclosures/addenda are required, who signs,
            how and when they must be delivered, buyer rights to terminate and deadlines, exemptions,
            and sample notification language. Include Code cites (e.g., §55.1-700+), DPOR/REB references,
            effective dates, and documentation/proof requirements.
        """),
        llm=DomainLLM,
        verbose=True,
        allow_delegation=False,
    )

def make_aml_fraud_agent() -> Agent:
    return Agent(
        role="AML / Fraud Analyst (FinCEN Real Estate Rule)",
        backstory=dedent("""\
            Specialist in the FinCEN final rule under the Bank Secrecy Act (31 CFR Chapter X; RIN 1506-AB54)
            governing non-financed transfers of residential real property to certain legal entities and trusts
            nationwide. You determine coverage, reporting responsibilities, required data elements, filing due dates,
            recordkeeping/retention, exemptions (e.g., individual transferees), penalties, and red flags.
            You place requirements in operational checklists for closings/settlements and coordinate with compliance teams.
            Always answer strictly from RAG with precise citations, definitions, and effective dates.
        """),
        goal=dedent("""\
            Decide whether a transfer is a 'covered transaction' and who must file;
            list required report fields, filing deadline, retention requirements, exemptions, penalties,
            and practical red flags. Include citations to the FinCEN final rule (31 CFR Chapter X),
            BSA authority, and the effective date (Dec 1, 2025).
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
          "domain": "tenant_landlord" | "listing_compliance" | "transaction_disclosures" | "aml_fraud",
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
    lc_func: Callable[[str, Dict[str, Any]], Any] = rag_listing_compliance
    td_func: Callable[[str, Dict[str, Any]], Any] = rag_transaction_disclosures
    af_func: Callable[[str, Dict[str, Any]], Any] = rag_aml_fraud

    def run(self, user_query: str) -> Dict[str, Any]:
        router = make_router_agent()
        refiner = make_query_refiner_agent()
        tl_agent = make_tenant_landlord_agent()
        lc_agent = make_listing_compliance_agent()
        td_agent = make_transaction_disclosures_agent()
        af_agent = make_aml_fraud_agent()

        # --- 1) ROUTE ---
        try:
            crew_route = Crew(agents=[router], tasks=[make_route_task(router, user_query)], verbose=True)
            route_out = str(crew_route.kickoff())
            packet = self._extract_json(route_out)
            if not packet or "domain" not in packet:
                # deterministic fallback
                fallback = _router_structured(user_query)
                domain = fallback["domain"] or "listing_compliance"
                geo_hints = fallback["geo_hints"]
                rationale = "Fallback routing by heuristic."
            else:
                domain = packet.get("domain") or "listing_compliance"
                geo_hints = packet.get("geo_hints", {})
                rationale = packet.get("rationale", "")
        except Exception as e:
            print(f"Router failed, using fallback: {e}")
            fallback = _router_structured(user_query)
            domain = fallback["domain"] or "listing_compliance"
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
        elif domain == "transaction_disclosures":
            domain_agent = td_agent
            result = self.td_func(optimized_query, metadata)
        elif domain == "aml_fraud":
            domain_agent = af_agent
            result = self.af_func(optimized_query, metadata)
        else:  # listing_compliance
            domain_agent = lc_agent
            result = self.lc_func(optimized_query, metadata)

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
            json_match = re.search(r'\{[^}]*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return json.loads(text.strip())
        except Exception:
            return None

# =========================
# 8) SIMPLE CLI
# =========================
# if __name__ == "__main__":
#     print("=== Real-Estate Compliance Router (CrewAI) ===")
#     print("Uses Gemini 2.5 Flash with native google.genai client.")
#     print("Press Ctrl+C to exit.\n")

#     while True:
#         try:
#             q = input("User: ").strip()
#             if not q:
#                 continue
#             app = RealEstateRouterCrew()
#             out = app.run(q)
#             print("\n--- ROUTING ---")
#             print(json.dumps(out["routing"], indent=2))
#             print("\n--- REFINED QUERY ---")
#             print(out["refined"]["optimized_query"])
#             print("\n--- ANSWER (RAG) ---")
#             print(json.dumps(out["result"], indent=2))
#         except KeyboardInterrupt:
#             print("\nGoodbye!")
#             break
#         except Exception as e:
#             print(f"Error: {e}")



# Add these Pydantic models for request/response
class QueryRequest(BaseModel):
    text: str
    
class QueryResponse(BaseModel):
    result: dict
    routing: dict
    refined_query: str

# Create FastAPI app instance
app = FastAPI(
    title="Real Estate Compliance Router API",
    description="API for routing and answering real estate compliance queries",
    version="1.0.0"
)

# Add CORS middleware - THIS IS THE KEY FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly allow OPTIONS
    allow_headers=["*"],
)

# Initialize the crew router
crew_router = RealEstateRouterCrew()

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a real estate compliance query and return the result.
    
    Args:
        request: QueryRequest containing the text query
        
    Returns:
        QueryResponse with the routing info, refined query, and RAG result
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Query text cannot be empty")
        
        # Process the query using your existing crew router
        result = crew_router.run(request.text)
        
        return QueryResponse(
            result=result["result"],
            routing=result["routing"],
            refined_query=result["refined"]["optimized_query"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Real Estate Compliance Router API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is operational"}

# Add explicit OPTIONS handler if needed (usually not required with proper CORS middleware)
@app.options("/query")
async def options_query():
    """Handle CORS preflight requests"""
    return {"message": "OK"}

# Add this function to run the server
def run_server():
    """Run the FastAPI server"""
    uvicorn.run(
        "crew_real_estate_router:app",  # module:app
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )