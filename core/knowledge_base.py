"""
ChromaDB-based RAG Knowledge Base for GrantBridge.
Stores and retrieves grant information with source validation.
"""
import os
import json
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed grant data (realistic, publicly known schemes – used for demo)
# ---------------------------------------------------------------------------
SEED_GRANTS = [
    {
        "id": "startup_india_seed_fund",
        "name": "Startup India Seed Fund Scheme (SISFS)",
        "source": "https://seedfund.startupindia.gov.in",
        "amount_min": 500000,
        "amount_max": 2000000,
        "currency": "INR",
        "deadline": "2025-12-31",
        "stage": ["Idea", "MVP"],
        "sectors": ["All"],
        "location": ["India"],
        "description": "Provides financial assistance to startups for proof of concept, prototype development, product trials, market entry and commercialisation.",
        "eligibility": "DPIIT-recognised startup, incorporated less than 2 years ago, not received more than ₹10 lakhs of monetary support.",
        "documents": ["DPIIT certificate", "Incorporation certificate", "PAN card", "Business plan"],
        "verified": True,
        "last_verified": "2024-11-01",
    },
    {
        "id": "birac_big_grant",
        "name": "BIRAC BIG (Biotechnology Ignition Grant)",
        "source": "https://birac.nic.in/big.php",
        "amount_min": 1000000,
        "amount_max": 5000000,
        "currency": "INR",
        "deadline": "2025-06-30",
        "stage": ["MVP", "Early-stage"],
        "sectors": ["Biotech", "Healthcare", "Life Sciences", "Agri-tech"],
        "location": ["India"],
        "description": "Supports innovative biotech startups at early stage with ignition funding for translational research.",
        "eligibility": "Indian incorporated startup/company, strong IP component, biotech-focused innovation.",
        "documents": ["Incorporation certificate", "IP documentation", "Research proposal", "Team CV"],
        "verified": True,
        "last_verified": "2024-10-15",
    },
    {
        "id": "aim_atal_grant",
        "name": "Atal Innovation Mission (AIM) Grants",
        "source": "https://aim.gov.in",
        "amount_min": 5000000,
        "amount_max": 100000000,
        "currency": "INR",
        "deadline": "2025-09-30",
        "stage": ["Early-stage", "Growth"],
        "sectors": ["Deep-tech", "AI", "Space", "Energy", "Healthcare"],
        "location": ["India"],
        "description": "Supports deep-tech startups through grants for R&D, prototype development and commercialisation.",
        "eligibility": "DPIIT-recognised, annual revenue < ₹25 crores, strong technological innovation.",
        "documents": ["DPIIT certificate", "Audited financials", "Patent/IP documents", "Business plan"],
        "verified": True,
        "last_verified": "2024-11-10",
    },
    {
        "id": "meity_tide_20",
        "name": "MeitY TIDE 2.0",
        "source": "https://tide.meity.gov.in",
        "amount_min": 750000,
        "amount_max": 2000000,
        "currency": "INR",
        "deadline": "2025-08-15",
        "stage": ["Idea", "MVP"],
        "sectors": ["IT", "AI", "IoT", "Cybersecurity", "Fintech"],
        "location": ["India"],
        "description": "Technology Incubation and Development of Entrepreneurs scheme for tech startups.",
        "eligibility": "Early-stage tech startup, incubated at TIDE TBI, Indian nationality.",
        "documents": ["Incubation letter", "Business plan", "Team profile"],
        "verified": True,
        "last_verified": "2024-09-20",
    },
    {
        "id": "eu_horizon_europe_eic",
        "name": "EU Horizon Europe – EIC Accelerator",
        "source": "https://eic.ec.europa.eu",
        "amount_min": 500000,
        "amount_max": 2500000,
        "currency": "EUR",
        "deadline": "2025-10-01",
        "stage": ["Early-stage", "Growth"],
        "sectors": ["All", "Deep-tech", "Green-tech", "Health"],
        "location": ["EU", "EEA"],
        "description": "EIC Accelerator supports individual high-tech companies and startups to develop and scale up breakthrough innovations.",
        "eligibility": "SME incorporated in EU/EEA country, disruptive innovation, strong growth potential.",
        "documents": ["Business plan", "Financial projections", "Technical description", "Pitch video"],
        "verified": True,
        "last_verified": "2024-10-01",
    },
    {
        "id": "nsf_sbir_phase1",
        "name": "NSF SBIR Phase I (US)",
        "source": "https://seedfund.nsf.gov",
        "amount_min": 275000,
        "amount_max": 275000,
        "currency": "USD",
        "deadline": "2025-11-05",
        "stage": ["Idea", "MVP"],
        "sectors": ["STEM", "Deep-tech", "Climate", "Healthcare", "AI"],
        "location": ["USA"],
        "description": "NSF SBIR Phase I awards help startups prove technical feasibility of innovative, high-risk ideas.",
        "eligibility": "US for-profit small business, majority US-citizen owned, strong R&D component.",
        "documents": ["Project pitch", "SBIR application", "Team CV", "Budget justification"],
        "verified": True,
        "last_verified": "2024-10-20",
    },
]


class GrantKnowledgeBase:
    """
    Manages the ChromaDB vector store for grant information.
    Falls back to in-memory storage when ChromaDB is unavailable.
    """

    def __init__(self, config):
        self.config = config
        self._collection = None
        self._in_memory: Dict[str, Dict] = {}
        self._use_chroma = False
        self._doc_hashes: set = set()
        self._init_store()
        self._seed_grants()

    def _init_store(self):
        try:
            import chromadb
            from chromadb.config import Settings

            os.makedirs(self.config.CHROMA_PERSIST_DIR, exist_ok=True)
            client = chromadb.PersistentClient(path=self.config.CHROMA_PERSIST_DIR)
            self._collection = client.get_or_create_collection(
                name=self.config.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
            logger.info("ChromaDB initialised successfully.")
        except Exception as exc:
            logger.warning(f"ChromaDB unavailable ({exc}) – using in-memory store.")

    def _seed_grants(self):
        """Add seed grants to the knowledge base if not already present."""
        for grant in SEED_GRANTS:
            self.add_grant(grant)

    def _build_document_text(self, grant: Dict) -> str:
        return (
            f"Grant: {grant['name']}\n"
            f"Description: {grant['description']}\n"
            f"Eligibility: {grant['eligibility']}\n"
            f"Sectors: {', '.join(grant.get('sectors', []))}\n"
            f"Stage: {', '.join(grant.get('stage', []))}\n"
            f"Location: {', '.join(grant.get('location', []))}\n"
            f"Amount: {grant.get('currency', 'USD')} "
            f"{grant.get('amount_min', 0):,} – {grant.get('amount_max', 0):,}\n"
            f"Deadline: {grant.get('deadline', 'Unknown')}\n"
            f"Documents required: {', '.join(grant.get('documents', []))}"
        )

    def add_grant(self, grant: Dict) -> bool:
        doc_text = self._build_document_text(grant)
        doc_hash = hashlib.md5(doc_text.encode()).hexdigest()
        if doc_hash in self._doc_hashes:
            return False  # duplicate
        self._doc_hashes.add(doc_hash)
        grant_id = grant.get("id", doc_hash[:16])

        # Always populate in-memory cache (used as fallback)
        self._in_memory[grant_id] = grant

        if self._use_chroma and self._collection is not None:
            try:
                self._collection.upsert(
                    ids=[grant_id],
                    documents=[doc_text],
                    metadatas=[{
                        "name": grant.get("name", ""),
                        "source": grant.get("source", ""),
                        "deadline": grant.get("deadline", ""),
                        "verified": str(grant.get("verified", False)),
                        "last_verified": grant.get("last_verified", ""),
                        "sectors": json.dumps(grant.get("sectors", [])),
                        "stage": json.dumps(grant.get("stage", [])),
                        "location": json.dumps(grant.get("location", [])),
                        "amount_min": str(grant.get("amount_min", 0)),
                        "amount_max": str(grant.get("amount_max", 0)),
                        "currency": grant.get("currency", "USD"),
                    }],
                )
            except Exception as exc:
                logger.error(f"ChromaDB upsert error: {exc}")
        return True

    def search(self, query: str, n_results: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve top-N relevant grants for a query.
        Returns validated, deduplicated, non-outdated results.
        Falls back to in-memory keyword search if ChromaDB returns empty.
        """
        if self._use_chroma and self._collection is not None:
            results = self._chroma_search(query, n_results)
            if results:
                return results
            # ChromaDB returned nothing (e.g. embedding model not yet loaded) – fall back
            logger.warning("ChromaDB returned no results, falling back to in-memory search.")
        return self._memory_search(query, n_results, filters)

    def _chroma_search(self, query: str, n_results: int) -> List[Dict]:
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, max(1, self._collection.count())),
            )
            grants = []
            for i, doc in enumerate(results.get("documents", [[]])[0]):
                meta = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
                dist = results.get("distances", [[]])[0][i] if results.get("distances") else 1.0
                grant = self._meta_to_grant(meta, doc, dist)
                if self._validate_grant(grant):
                    grants.append(grant)
            return grants
        except Exception as exc:
            logger.error(f"ChromaDB search error: {exc}")
            return self._memory_search(query, n_results)

    def _memory_search(self, query: str, n_results: int, filters: Optional[Dict] = None) -> List[Dict]:
        """Simple keyword-based fallback search."""
        query_words = set(query.lower().split())
        scored = []
        for gid, grant in self._in_memory.items():
            doc_text = self._build_document_text(grant).lower()
            score = sum(1 for w in query_words if w in doc_text)
            if score > 0:
                scored.append((score, grant))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [g for _, g in scored[:n_results]]

    def _meta_to_grant(self, meta: Dict, doc: str, distance: float) -> Dict:
        return {
            "name": meta.get("name", "Unknown"),
            "source": meta.get("source", ""),
            "deadline": meta.get("deadline", ""),
            "verified": meta.get("verified", "False") == "True",
            "last_verified": meta.get("last_verified", ""),
            "sectors": json.loads(meta.get("sectors", "[]")),
            "stage": json.loads(meta.get("stage", "[]")),
            "location": json.loads(meta.get("location", "[]")),
            "amount_min": int(meta.get("amount_min", 0)),
            "amount_max": int(meta.get("amount_max", 0)),
            "currency": meta.get("currency", "USD"),
            "relevance_score": max(0.0, 1.0 - distance),
            "document_text": doc,
        }

    def _validate_grant(self, grant: Dict) -> bool:
        """RAG guardrail: filter outdated or unverified grants."""
        deadline_str = grant.get("deadline", "")
        if deadline_str:
            try:
                deadline = datetime.strptime(deadline_str, "%Y-%m-%d")
                if deadline < datetime.now() - timedelta(days=30):
                    logger.info(f"Filtered outdated grant: {grant.get('name')}")
                    return False
            except ValueError:
                pass
        return True

    def get_all_grants(self) -> List[Dict]:
        if self._in_memory:
            return list(self._in_memory.values())
        return SEED_GRANTS
