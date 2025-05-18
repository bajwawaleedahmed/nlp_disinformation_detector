import json
from pathlib import Path

output_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/knowledge_base.json")

knowledge_base = {
    "entities": [
        "IPCC", "NASA", "UNEP", "WMO", "CO2", "methane", "sea level rise", "1.5°C", "Paris Agreement",
        "Intergovernmental Panel on Climate Change", "glacier melt", "carbon emissions", "fossil fuels"
    ],
    "verified_facts": [
        "Human activity is the primary driver of recent global warming.",
        "The planet has warmed approximately 1.1°C since pre-industrial times.",
        "The IPCC is the leading international body for climate science consensus.",
        "The Arctic is warming roughly twice as fast as the global average.",
        "Sea levels are rising due to thermal expansion and glacial melt.",
        "Burning fossil fuels increases CO2 concentration in the atmosphere.",
        "CO2 and methane are greenhouse gases that trap heat in the atmosphere.",
        "The Paris Agreement aims to limit warming to below 2°C, ideally 1.5°C.",
        "NASA and NOAA both confirm long-term global temperature rise.",
        "Glaciers around the world have been retreating significantly since 1900."
    ]
}

with output_path.open("w", encoding="utf-8") as f:
    json.dump(knowledge_base, f, indent=2)

print(f"✅ Knowledge base saved to {output_path}")
