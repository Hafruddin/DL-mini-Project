"""
Disposal and Recycling Recommendation Engine for Smart Waste Classification.
Provides deterministic, domain-specific guidance for the six core waste classes.
"""

from typing import Dict, Any

RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    "Plastic": {
        "display_name": "Plastic",
        "icon": "🥤",
        "category_badge_color": "#2E7D32",
        "short_description": "Synthetic or semi-synthetic organic polymers commonly used in containers and packaging.",
        "examples": [
            "Beverage bottles (PET / PETE #1)",
            "Milk jugs & detergent bottles (HDPE #2)",
            "Rigid tubs and clear food containers",
            "Bottle caps and lids (separated if advised locally)"
        ],
        "preparation_instructions": [
            "Empty any residual liquid or food scraps completely.",
            "Rinse briefly with greywater or leftover rinse water to remove sticky residues.",
            "Leave labels intact unless local facilities specifically request label removal.",
            "Squash or flatten bottles to save bin collection space."
        ],
        "disposal_instructions": (
            "Place clean, rigid plastics into the dedicated dry recyclable bin. "
            "Never put soft plastic bags, film wrappers, or expanded polystyrene (Styrofoam) "
            "into standard curbside recycling bins unless explicitly accepted."
        ),
        "recycling_guidance": (
            "PET (#1) and HDPE (#2) bottles possess the highest recycling market rates worldwide. "
            "Check the resin identification code on the bottom of the container. If marked #3 (PVC), "
            "#6 (PS), or #7 (Other), check municipal guidance as these often require specialized processing."
        ),
        "safety_precautions": (
            "Never incinerate plastics in open domestic fires, as this releases toxic furans and dioxins. "
            "Discard chemical or pesticide containers through hazardous household waste channels, not standard recycling."
        )
    },
    "Paper": {
        "display_name": "Paper",
        "icon": "📄",
        "category_badge_color": "#1565C0",
        "short_description": "Cellulose fiber sheets derived from wood pulp, cotton, or recycled paper stocks.",
        "examples": [
            "Newspapers, magazines, and flyers",
            "Office copy paper, notebooks, and envelopes",
            "Paper bags and wrapping sheets (non-metallic)",
            "Paperback books and clean brochures"
        ],
        "preparation_instructions": [
            "Keep sheets dry and free from grease, food oils, and sauce stains.",
            "Remove metal spiral bindings, heavy plastic covers, and metallic glitter.",
            "Small staples and envelope address windows are generally acceptable in modern pulping mills.",
            "Stack sheets flat or fold neatly; avoid shredding unless necessary for data privacy."
        ],
        "disposal_instructions": (
            "Deposit dry, clean paper into the designated blue / paper recycling bin. "
            "Ensure the collection container lid is securely closed during rainy weather to avoid saturation."
        ),
        "recycling_guidance": (
            "Clean paper fibers can be recycled 5 to 7 times before fibers become too short for papermaking. "
            "Soiled paper napkins, greasy paper plates, and thermal receipts (BPA-coated) are not recyclable with dry paper."
        ),
        "safety_precautions": (
            "Keep paper away from open heating elements and wet kitchen waste. Wet paper molds quickly and "
            "can ruin entire batches in dry paper collection streams."
        )
    },
    "Metal": {
        "display_name": "Metal",
        "icon": "🥫",
        "category_badge_color": "#6A1B9A",
        "short_description": "Ferrous and non-ferrous metallic packaging, primarily aluminium and tin-coated steel.",
        "examples": [
            "Aluminium beverage cans",
            "Steel food cans (tinned soup, beans, vegetables)",
            "Clean aluminium foil sheets and pie trays",
            "Metal jar lids and crown bottle caps"
        ],
        "preparation_instructions": [
            "Scrape out and rinse off food remains to deter pests and odors.",
            "For tins with lids, push the lid into the empty tin or secure it inside.",
            "For aluminium foil, scrunch clean pieces into a ball at least the size of a golf ball so sorting machines can detect it.",
            "Do not crush cans if your automated local sorting facility uses optical shape sorters."
        ],
        "disposal_instructions": (
            "Place into your commingled or metal dry-recyclable bin. "
            "Separate bulky metal scrap (e.g., bicycle frames, cookware) for drop-off at municipal scrap centers."
        ),
        "recycling_guidance": (
            "Aluminium is infinitely recyclable without degradation; recycling aluminium saves 95% of the energy "
            "required to produce new metal from raw bauxite ore."
        ),
        "safety_precautions": (
            "Beware of sharp, ragged edges on opened can lids. Do not place pressurized aerosol cans that are "
            "not completely empty or puncture-free into curbside metal bins."
        )
    },
    "Glass": {
        "display_name": "Glass",
        "icon": "🍾",
        "category_badge_color": "#00838F",
        "short_description": "Silica-based rigid transparent or tinted container glass.",
        "examples": [
            "Glass beverage bottles (soda, beer, wine)",
            "Food jars (jam, pasta sauce, condiment jars)",
            "Cosmetic glass jars and small glass medicine vials"
        ],
        "preparation_instructions": [
            "Empty all liquids and rinse food jars thoroughly.",
            "Remove corks, plastic caps, and metal lids (recycle lids in the metal stream).",
            "Paper labels do not need to be scraped off; they burn off cleanly in high-temperature glass cullet furnaces.",
            "Keep different glass colors grouped together if municipal drop-off centers provide separated color bins."
        ],
        "disposal_instructions": (
            "Place intact bottles and jars into designated glass recycling receptacles or bottle banks. "
            "Deposit gently to avoid shattering during disposal."
        ),
        "recycling_guidance": (
            "Container glass is 100% recyclable repeatedly without loss of purity or quality. "
            "Note: Window glass, light bulbs, Pyrex, ceramics, and mirrors have different melting points and "
            "must never be mixed with container glass."
        ),
        "safety_precautions": (
            "CRITICAL SAFETY WARNING: If glass is shattered, wrap securely in heavy newspaper, label 'Caution: Broken Glass', "
            "and place in residual waste or dedicated municipal collection to prevent serious cuts to sanitation personnel."
        )
    },
    "Cardboard": {
        "display_name": "Cardboard",
        "icon": "📦",
        "category_badge_color": "#E65100",
        "short_description": "Heavy-duty paper stock, corrugated fiberboard, and paperboard used in cartons and shipping boxes.",
        "examples": [
            "Corrugated shipping boxes and delivery packages",
            "Cereal, dry grocery, and shoe boxes (paperboard)",
            "Cardboard tubes (toilet paper, paper towel rolls)",
            "Clean egg cartons (pressed pulp)"
        ],
        "preparation_instructions": [
            "Flatten and break down boxes completely to maximize bin space.",
            "Strip away excessive plastic packaging tape, bubble wrap, and styrofoam inserts.",
            "Keep completely dry; moisture weakens cardboard structure and causes fungal degradation.",
            "Cut oversized boxes down to standard municipal collection bundle dimensions."
        ],
        "disposal_instructions": (
            "Place flat inside dry cardboard bins or tie neatly in bundles for dry curbside recycling. "
            "Never leave outside in open rain."
        ),
        "recycling_guidance": (
            "Cardboard has high recycling value and produces fresh corrugated liners and paper pulp. "
            "Caution: Pizza boxes with heavy grease or melted cheese soaked into the fibers are contaminated; "
            "tear off clean tops for recycling and compost the greasy bottoms."
        ),
        "safety_precautions": (
            "Check boxes for packing staples or wire bands before compacting by hand to avoid puncture wounds."
        )
    },
    "Organic Waste": {
        "display_name": "Organic Waste",
        "icon": "🍎",
        "category_badge_color": "#558B2F",
        "short_description": "Biodegradable organic matter derived from plants, kitchens, and garden trimmings.",
        "examples": [
            "Fruit and vegetable peels, cores, and cuttings",
            "Coffee grounds, filters, and loose tea leaves",
            "Eggshells, nut shells, and bread crusts",
            "Garden clippings, leaves, and cut flowers"
        ],
        "preparation_instructions": [
            "Separate all plastic stickers from fruit and vegetable skins before discarding.",
            "Do not pack inside non-biodegradable polythene bags; use certified compostable liners or newspaper wraps.",
            "Drain excess liquid from food scraps to minimize leachate and odor.",
            "Chop bulky garden trimmings and thick stalks into smaller segments for faster decomposition."
        ],
        "disposal_instructions": (
            "Dispose in municipal green organic / compost bins or into backyard composting bins. "
            "Diverting organic waste from landfills directly mitigates methane greenhouse gas emissions."
        ),
        "recycling_guidance": (
            "Organic waste is recycled through aerobic composting or anaerobic digestion into nutrient-rich soil amendments. "
            "Guidance note: Backyard compost piles thrive on raw plant waste; meat, bones, and dairy usually require industrial high-temperature municipal composters."
        ),
        "safety_precautions": (
            "Wash hands thoroughly after handling decaying matter. Keep compost containers covered with tight lids "
            "to prevent vermin, fruit flies, and odors."
        )
    }
}


def get_recommendation(category: str) -> Dict[str, Any]:
    """
    Retrieve structured recommendation guidance for a predicted category.
    Falls back gracefully if an unknown class name is passed.
    """
    if category in RECOMMENDATIONS:
        return RECOMMENDATIONS[category]
    
    # Generic fallback
    return {
        "display_name": category,
        "icon": "♻️",
        "category_badge_color": "#245B45",
        "short_description": "Waste material requiring segregation and responsible disposal.",
        "examples": ["General item"],
        "preparation_instructions": ["Clean and segregate before disposal."],
        "disposal_instructions": "Consult local municipal guidelines for appropriate disposal channels.",
        "recycling_guidance": "Verify local recycling facility acceptance for this material class.",
        "safety_precautions": "Handle safely to avoid injury."
    }
