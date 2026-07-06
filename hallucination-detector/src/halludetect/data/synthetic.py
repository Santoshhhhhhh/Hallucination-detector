"""Synthetic RAGTruth-style dataset generator.

Used when the real RAGTruth dataset is unavailable (e.g. offline / no HF access).
It produces examples with *structured, learnable* signal so the whole probe
pipeline is exercisable end-to-end and tests are deterministic.

The generator builds each of the three classes from templated facts:

  * context_supported : the answer restates a fact that IS in the context.
  * common_knowledge   : the answer states a real-world fact NOT in the context.
  * hallucinated       : the answer states a fabricated fact contradicting/absent
                         from the context.

This is deliberately simple. It is NOT a substitute for real RAGTruth data, but
it lets the extractor -> sweep -> probe -> serve chain run and be validated.
"""
from __future__ import annotations

import random
from .schema import Example

# (topic, context_fact, question, common_knowledge_fact, hallucinated_fact)
_TEMPLATES = [
    ("solar", "The report states the new solar farm has a capacity of 480 megawatts.",
     "What is the capacity of the new solar farm?",
     "Solar panels convert sunlight into electricity using the photovoltaic effect.",
     "The report states the new solar farm has a capacity of 9,900 megawatts."),
    ("merger", "According to the filing, the merger was completed on March 3, 2021.",
     "When was the merger completed?",
     "Mergers combine two companies into a single legal entity.",
     "According to the filing, the merger was completed on August 19, 1812."),
    ("drug", "The trial enrolled 1,204 patients across twelve clinical sites.",
     "How many patients were enrolled in the trial?",
     "Clinical trials typically proceed through several sequential phases.",
     "The trial enrolled 6 million patients across twelve clinical sites."),
    ("river", "The passage notes the river flows for 340 kilometres before the delta.",
     "How long is the river before the delta?",
     "Rivers carry freshwater from higher elevations toward the sea.",
     "The passage notes the river flows for 9 light-years before the delta."),
    ("ceo", "The memo says Dana Ruiz was appointed interim chief executive in 2019.",
     "Who was appointed interim chief executive?",
     "A chief executive officer is the highest-ranking corporate officer.",
     "The memo says a talking dolphin was appointed interim chief executive."),
    ("battery", "Documentation lists the battery pack energy density at 260 Wh/kg.",
     "What is the battery pack energy density?",
     "Battery energy density measures stored energy per unit mass.",
     "Documentation lists the battery pack energy density at 40,000 Wh/kg."),
    ("treaty", "The article explains the treaty was ratified by 14 member states.",
     "How many member states ratified the treaty?",
     "Treaties are formal agreements between sovereign states under international law.",
     "The article explains the treaty was ratified by every planet in the galaxy."),
    ("harvest", "Records indicate the harvest yielded 3.2 tonnes per hectare last season.",
     "What was the harvest yield last season?",
     "Crop yield is commonly measured in tonnes per hectare.",
     "Records indicate the harvest yielded negative fifty tonnes per hectare."),
    ("bridge", "The survey reports the bridge spans 1.9 kilometres across the strait.",
     "How long is the bridge across the strait?",
     "Suspension bridges use cables anchored at each end to carry the deck load.",
     "The survey reports the bridge spans the entire Pacific Ocean."),
    ("vaccine", "The study found the vaccine was 87 percent effective after two doses.",
     "How effective was the vaccine after two doses?",
     "Vaccines prime the immune system to recognize specific pathogens.",
     "The study found the vaccine was 3000 percent effective after two doses."),
]

_ANSWER_PREFIXES = [
    "", "Based on the context, ", "The answer is that ", "In short, ",
    "To summarize, ", "According to the material, ",
]


def _make(topic_idx: int, cls: str, rng: random.Random, uid: int) -> Example:
    topic, ctx, q, ck, hallu = _TEMPLATES[topic_idx]
    prefix = rng.choice(_ANSWER_PREFIXES)
    if cls == "context_supported":
        ans = prefix + ctx.split("states ")[-1] if "states " in ctx else prefix + ctx
        ans = prefix + ctx.split(", ")[-1] if ", " in ctx and rng.random() < 0.5 else prefix + ctx
    elif cls == "common_knowledge":
        ans = prefix + ck
    else:  # hallucinated
        ans = prefix + hallu
    return Example(
        id=f"syn-{uid:06d}",
        context=ctx,
        question=q,
        answer=ans.strip(),
        label=cls,
        source="synthetic",
        meta={"topic": topic},
    )


def generate(n_per_class: int = 200, seed: int = 13) -> list[Example]:
    """Generate a balanced synthetic dataset.

    Args:
        n_per_class: number of examples for each of the three classes.
        seed: RNG seed for reproducibility.
    """
    rng = random.Random(seed)
    classes = ["context_supported", "common_knowledge", "hallucinated"]
    out: list[Example] = []
    uid = 0
    for cls in classes:
        for _ in range(n_per_class):
            topic_idx = rng.randrange(len(_TEMPLATES))
            out.append(_make(topic_idx, cls, rng, uid))
            uid += 1
    rng.shuffle(out)
    return out
