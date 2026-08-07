"""Curated English news headlines for risk-relevance benchmarking."""

from __future__ import annotations

NEWS_RISK_BENCHMARK: list[dict[str, object]] = [
    {
        "id": "n1",
        "title": "Major water main burst floods downtown basement",
        "summary": "Aging cast iron pipe failed during freeze, injuring two workers.",
        "risk_relevant": True,
        "category": "infrastructure",
    },
    {
        "id": "n2",
        "title": "Storm surge warnings issued for coastal industrial zone",
        "summary": "Hurricane forecast prompts building closures and pipe inspections.",
        "risk_relevant": True,
        "category": "weather",
    },
    {
        "id": "n3",
        "title": "OSHA cites factory after forklift accident",
        "summary": "Worker injury leads to safety training mandate.",
        "risk_relevant": True,
        "category": "accident",
    },
    {
        "id": "n4",
        "title": "City council approves new art festival budget",
        "summary": "Local artists celebrate funding for summer exhibition.",
        "risk_relevant": False,
        "category": "culture",
    },
    {
        "id": "n5",
        "title": "Tech company reports quarterly earnings beat",
        "summary": "Stock rises after cloud revenue exceeds analyst expectations.",
        "risk_relevant": False,
        "category": "business",
    },
    {
        "id": "n6",
        "title": "Heat wave expected to stress HVAC systems across region",
        "summary": "14-day extreme heat advisory for commercial buildings.",
        "risk_relevant": True,
        "category": "weather",
    },
    {
        "id": "n7",
        "title": "Historic building inspection reveals structural concerns",
        "summary": "Engineers recommend immediate load-bearing review.",
        "risk_relevant": True,
        "category": "building",
    },
    {
        "id": "n8",
        "title": "Local team wins championship in overtime thriller",
        "summary": "Fans celebrate downtown parade scheduled for Monday.",
        "risk_relevant": False,
        "category": "sport",
    },
    {
        "id": "n9",
        "title": "Sewer line collapse triggers emergency road closure",
        "summary": "60-year-old pipe infrastructure under investigation.",
        "risk_relevant": True,
        "category": "infrastructure",
    },
    {
        "id": "n10",
        "title": "New restaurant opens on Main Street",
        "summary": "Chef brings fusion menu to renovated storefront.",
        "risk_relevant": False,
        "category": "society",
    },
    {
        "id": "n11",
        "title": "Chemical leak at refinery prompts evacuation",
        "summary": "Process safety incident under regulatory review.",
        "risk_relevant": True,
        "category": "accident",
    },
    {
        "id": "n12",
        "title": "Film festival announces award nominees",
        "summary": "Independent cinema highlights debut directors.",
        "risk_relevant": False,
        "category": "art",
    },
]

# Hugging Face Hub datasets used for capability benchmarking
HF_BENCHMARK_DATASETS = {
    "mining_safety_incidents": {
        "repo_id": "electricsheepafrica/africa-synth-mining-safety-incidents-all",
        "split": "train",
        "description": "Occupational mining safety incidents with injury severity labels",
        "license": "cc-by-4.0",
        "url": "https://huggingface.co/datasets/electricsheepafrica/africa-synth-mining-safety-incidents-all",
    },
}

HIGH_INJURY_SEVERITIES = frozenset(
    {"fatality", "days_away", "permanent_disability", "temporary_disability"}
)
