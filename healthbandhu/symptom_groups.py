"""Body-system grouping used only to help people browse 370+ symptoms in the app.

Grouping is keyword based and has no effect on any prediction. Rules are checked
in order and the first match wins; anything unmatched goes to "Other".
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, Sequence

GENERAL = {
    "fever", "chills", "fatigue", "weakness", "sweating", "feeling ill", "ache all over",
    "weight gain", "recent weight loss", "underweight", "thirst", "feeling cold", "feeling hot",
    "feeling hot and cold", "flu-like syndrome", "fluid retention", "stiffness all over",
    "swollen lymph nodes", "lymphedema", "excessive growth", "sleepiness", "decreased appetite",
    "excessive appetite", "pallor", "flushing", "allergic reaction", "peripheral edema",
    "poor circulation", "lower body pain",
}

RULES = [
    ("Children and infants", [r"infant", r"diaper", r"bedwetting", r"pulling at ears", r"lack of growth"]),
    ("Mental health and behaviour", [
        r"anxiety", r"depress", r"psychotic", r"insomnia", r"emotional", r"hostile", r"abusing alcohol",
        r"drug abuse", r"smoking", r"anger", r"delusions", r"temper", r"fears", r"self-esteem",
        r"obsessions", r"antisocial", r"hysterical", r"nightmares", r"sleepwalking", r"nailbiting",
        r"restlessness", r"stuttering",
    ]),
    ("General", []),  # exact names from GENERAL
    ("Brain and nerves", [
        r"dizziness", r"involuntary movements", r"fainting", r"headache", r"loss of sensation",
        r"focal weakness", r"slurring", r"memory", r"paresthesia", r"seizures", r"problems with movement",
        r"difficulty speaking", r"smell or taste",
    ]),
    ("Eyes", [r"\beye", r"vision", r"blindness", r"lacrimation", r"cross-eyed", r"pupils"]),
    ("Ear, nose, mouth and throat", [
        r"\bears?\b", r"hearing", r"\bnose", r"nasal", r"sinus", r"throat", r"hoarse", r"swallowing",
        r"tonsils", r"coryza", r"sneezing", r"mouth", r"tooth", r"\bgums?\b", r"tongue", r"\blips?\b",
        r"\bjaw\b", r"facial", r"\bface\b", r"nosebleed",
    ]),
    ("Heart, chest and breathing", [
        r"chest", r"breath", r"palpitations", r"heartbeat", r"heart rate", r"cough", r"sputum",
        r"wheezing", r"hemoptysis", r"apnea", r"\brib\b",
    ]),
    ("Digestion and abdomen", [
        r"abdom", r"stool", r"vomit", r"nausea", r"diarrhea", r"flatulence", r"jaundice", r"heartburn",
        r"regurgitation", r"bloating", r"melena", r"constipation", r"\banus\b", r"rectal",
        r"difficulty eating", r"stomach", r"belly button",
    ]),
    ("Urinary and kidney", [
        r"\burin", r"bladder", r"kidney", r"polyuria", r"suprapubic", r"hesitancy", r"side pain",
    ]),
    ("Sexual and reproductive health", [
        r"vagin", r"vulv", r"menstru", r"menopause", r"pregnan", r"intercourse", r"uterine", r"pelvic",
        r"infertility", r"breast", r"nipple", r"penis", r"penile", r"scrotum", r"testes", r"testicles",
        r"impotence", r"ejaculation", r"orgasm", r"sex drive", r"prostate", r"groin", r"hot flashes",
        r"postpartum",
    ]),
    ("Skin, hair and nails", [
        r"skin", r"rash", r"acne", r"itch", r"warts", r"moles?\b", r"hair", r"\bnails?\b", r"scalp",
        r"wrinkles",
    ]),
    ("Muscles, bones and joints", [
        r"pain", r"stiffness", r"swelling", r"weakness", r"cramps", r"spasms", r"lump", r"mass",
        r"joint", r"bones", r"muscle", r"knee", r"\bhip\b", r"ankle", r"wrist", r"elbow", r"shoulder",
        r"\barm\b", r"\bleg\b", r"\bfoot\b", r"\btoe\b", r"hand", r"finger", r"neck", r"back",
        r"bowlegged", r"feet turned in", r"posture",
    ]),
]

GROUP_ORDER = [name for name, _ in RULES] + ["Other"]
_COMPILED = [(name, [re.compile(p) for p in patterns]) for name, patterns in RULES]


def symptom_group(symptom: str) -> str:
    s = symptom.lower().strip()
    for name, patterns in _COMPILED:
        if name == "General":
            if s in GENERAL:
                return name
            continue
        if any(p.search(s) for p in patterns):
            return name
    return "Other"


def group_symptoms(symptoms: Sequence[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = OrderedDict((g, []) for g in GROUP_ORDER)
    for s in symptoms:
        groups[symptom_group(s)].append(s)
    for g in groups:
        groups[g].sort()
    return OrderedDict((g, items) for g, items in groups.items() if items)


def display_name(symptom: str) -> str:
    """Sentence-case label for the interface ('sharp chest pain' -> 'Sharp chest pain')."""
    return symptom[:1].upper() + symptom[1:] if symptom else symptom
