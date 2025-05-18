import json
from pathlib import Path

# Set output path
output_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_knowledge_base.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Define a simple knowledge base
knowledge_base = {
    "entities": {
        "IPCC": "Intergovernmental Panel on Climate Change",
        "NASA": "National Aeronautics and Space Administration",
        "UNEP": "United Nations Environment Programme",
        "CO2": "Carbon dioxide, a greenhouse gas contributing to global warming",
        "Methane": "A potent greenhouse gas with 25x the warming potential of CO2",
        "Paris Agreement": "2015 international treaty to limit global warming to below 2°C",
        "1.5°C": "Threshold identified to avoid the worst climate impacts",
        "Sea-level rise": "Result of ice melt and ocean thermal expansion",
        "Glaciers": "Losing mass globally due to warming",
        "El Niño": "Periodic climate pattern that raises global temps",
        "Fossil fuels": "Primary source of man-made greenhouse gases"
    },
    "facts": [
        "Human activities have caused approximately 1.1°C of global warming.",
        "CO2 concentrations exceeded 420 ppm in 2024.",
        "The past 9 years were the warmest on record.",
        "Sea levels are rising about 3.3 mm/year.",
        "Paris Agreement seeks to limit warming to 1.5–2°C.",
        "Glaciers have lost over 7 trillion tons of ice since 2000.",
        "Climate change drives more frequent extreme weather.",
        "Methane makes up 20% of anthropogenic emissions."
    ]
}

# Save it
with output_path.open("w", encoding="utf-8") as f:
    json.dump(knowledge_base, f, indent=2)

print(f"✅ Knowledge base saved to {output_path}")
