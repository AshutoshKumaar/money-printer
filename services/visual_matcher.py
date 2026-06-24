from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from core.models import Segment


@dataclass(slots=True)
class VisualPlan:
    """A relevance-first visual plan for one narration scene."""

    category: str
    keywords: list[str]
    query: str
    prompt: str
    confidence: float
    reason: str
    concept: str = ""

    def cache_key(self) -> str:
        payload = "|".join([self.category, self.query, self.prompt, self.concept])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def to_dict(self) -> dict:
        return asdict(self)


class VisualMatcher:
    """Maps scene narration to a relevant visual category, search query, and prompt."""

    CATEGORY_RULES: dict[str, dict[str, list[str] | str]] = {
        "everyday_science": {
            "keywords": [
                "empty room", "room", "wall", "walls", "echo", "sound", "noise", "loud",
                "quiet", "pen", "cap", "hole", "airway", "oxygen", "breathing", "safety",
                "object", "physics", "experiment", "reflection", "absorption", "furniture",
                "kamra", "khali", "awaaz", "shor", "dhwani", "ched", "suraksha",
            ],
            "query": "everyday science object close up physics experiment room",
            "prompt": "realistic everyday science visual, close-up object, simple physics experiment, clean room",
        },
        "space": {
            "keywords": [
                "outer space", "deep space", "cosmic", "galaxy", "planet", "stars",
                "black hole", "astronaut", "rocket", "telescope", "universe", "nebula",
                "milky way", "mars", "moon", "alien", "antariksh", "brahmand", "taare",
            ],
            "query": "deep space galaxy telescope cosmic mystery",
            "prompt": "deep space anomaly, galaxies, telescope, cosmic dust, mysterious object",
        },
        "ocean": {
            "keywords": [
                "ocean", "sea", "underwater", "deep sea", "marine", "creature", "submarine",
                "submersible", "trench", "mariana", "ship", "shark", "whale", "samundar",
                "sagar", "paani",
            ],
            "query": "deep ocean underwater creature trench",
            "prompt": "deep ocean trench, underwater darkness, sea creature silhouette, blue light rays",
        },
        "history": {
            "keywords": [
                "history", "ancient", "ruins", "empire", "king", "queen", "war", "artifact",
                "monument", "temple", "old map", "civilization", "historical", "itihas",
                "pracheen", "raja", "mandir", "yudh", "sabhyata", "kila",
            ],
            "query": "ancient ruins historical monument artifact map",
            "prompt": "ancient ruins, historical artifact, old map, dramatic museum lighting",
        },
        "person": {
            "keywords": [
                "scientist", "actor", "leader", "president", "inventor", "person", "man",
                "woman", "face", "portrait", "doctor", "teacher", "expert", "vaigyanik",
                "vyakti", "aadmi", "mahila", "chehra",
            ],
            "query": "person documentary cinematic silhouette science explainer",
            "prompt": "documentary-style person representation, silhouette, studio lighting, no fake celebrity face",
        },
        "technology": {
            "keywords": [
                "ai", "robot", "computer", "machine", "technology", "data", "internet",
                "server", "chip", "hacker", "artificial intelligence", "circuit", "software",
                "phone", "app", "device", "tech",
            ],
            "query": "technology device circuit computer data center",
            "prompt": "technology device close-up, glowing circuits, computer hardware, futuristic data visualization",
        },
        "nature": {
            "keywords": [
                "forest", "animal", "earth", "mountain", "volcano", "desert", "storm",
                "cloud", "jungle", "wildlife", "prakriti", "pahad", "toofan", "dharti",
            ],
            "query": "dramatic nature landscape storm mountain forest",
            "prompt": "dramatic natural landscape, storm clouds, cinematic light, high contrast",
        },
        "horror": {
            "keywords": [
                "creepy", "scary", "mystery", "mysterious", "dark", "haunted", "ghost",
                "fear", "dar", "darawna", "rahasya", "andhera",
            ],
            "query": "dark mysterious cinematic atmosphere",
            "prompt": "dark mysterious atmosphere, fog, dramatic shadows, suspenseful cinematic scene",
        },
    }

    STOPWORDS = {
        "the", "and", "for", "with", "this", "that", "from", "into", "about", "your", "you",
        "hai", "hain", "mein", "aur", "kya", "yeh", "ek", "ko", "ka", "ki", "ke", "se",
        "vertical", "cinematic", "realistic", "dramatic", "lighting", "close", "close-up",
    }

    def plan(self, segment: Segment, topic: str) -> VisualPlan:
        context_text = " ".join(
            [
                segment.text,
                segment.subtitle,
                segment.search_query,
                segment.image_prompt,
                topic,
            ]
        ).lower()
        
        # 1. Determine Category
        category = segment.visual_category.strip().lower()
        if category in self.CATEGORY_RULES:
            score = 5
        else:
            category_scores = self._score_categories(context_text)
            category, score = max(category_scores.items(), key=lambda item: item[1])
            if score == 0:
                category = self._topic_category(topic)
                score = 1

        keywords = self._extract_keywords(context_text)
        
        # 2. Determine scene-specific Query
        raw_query = segment.search_query.strip()
        cleaned_query = self._clean_query_from_generic(raw_query)
        
        if cleaned_query and len(cleaned_query.split()) >= 2:
            query_terms = cleaned_query
        else:
            # Fallback to English keywords + category
            english_keywords = [item for item in keywords if re.fullmatch(r"[a-zA-Z0-9-]+", item)]
            query_terms = " ".join(english_keywords[:5] + [category]).strip()
            
        rule = self.CATEGORY_RULES[category]
        prompt = self._prompt(segment, topic, category, keywords, str(rule["prompt"]))
        confidence = self._confidence(category, score, keywords, segment)
        reason = f"matched {category} (Gemini set: {bool(segment.visual_category)}) with score {score}"
        
        # 3. Use segment visual concept
        concept = segment.visual_concept.strip() or f"Visual for {category}"
        
        return VisualPlan(
            category=category,
            keywords=keywords,
            query=query_terms,
            prompt=prompt,
            confidence=confidence,
            reason=reason,
            concept=concept
        )

    def _clean_query_from_generic(self, query: str) -> str:
        words = re.findall(r"[a-zA-Z0-9-]+", query)
        generic_words = {
            "everyday", "science", "object", "close", "up", "close-up", "physics", "experiment", "room",
            "documentary", "cinematic", "silhouette", "explainer", "person", "man", "woman", "studio",
            "lighting", "concept", "background", "photo", "stock", "image", "video", "footage", "clip",
            "representation", "atmosphere", "mysterious", "dark", "dramatic", "realistic", "high",
            "contrast", "sharp", "subject", "composition", "vertical", "portrait"
        }
        clean = [w for w in words if w.lower() not in generic_words]
        return " ".join(clean)

    def _score_categories(self, text: str) -> dict[str, int]:
        scores: dict[str, int] = {}
        for category, rule in self.CATEGORY_RULES.items():
            score = 0
            for keyword in rule["keywords"]:  # type: ignore[index]
                keyword_text = str(keyword).lower()
                if " " in keyword_text:
                    if keyword_text in text:
                        score += 4
                    continue
                if re.search(rf"\b{re.escape(keyword_text)}\b", text):
                    score += 1
            scores[category] = score
        return scores

    def _topic_category(self, topic: str) -> str:
        scores = self._score_categories(topic.lower())
        category, score = max(scores.items(), key=lambda item: item[1])
        return category if score else "everyday_science"

    def _extract_keywords(self, text: str) -> list[str]:
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}|[\u0900-\u097F]{2,}", text)
        clean: list[str] = []
        for word in words:
            lowered = word.lower()
            if lowered in self.STOPWORDS or len(lowered) < 3:
                continue
            if lowered not in clean:
                clean.append(lowered)
            if len(clean) >= 10:
                break
        return clean

    def _query_terms(self, keywords: list[str], default_query: str) -> str:
        english_keywords = [item for item in keywords if re.fullmatch(r"[a-zA-Z0-9-]+", item)]
        selected = english_keywords[:5]
        return " ".join(selected + default_query.split()[:5]).strip() or default_query

    def _prompt(self, segment: Segment, topic: str, category: str, keywords: list[str], category_prompt: str) -> str:
        keyword_text = ", ".join(keywords[:6])
        return (
            f"{segment.image_prompt}. The visual must directly match the narration context: {category}. "
            f"Topic: {topic}. Key visual concepts: {keyword_text}. Include {category_prompt}. "
            "Avoid unrelated portraits, random people, text overlays, logos, and watermarks."
        )

    def _confidence(self, category: str, score: int, keywords: list[str], segment: Segment) -> float:
        prompt_text = f"{segment.image_prompt} {segment.search_query}".lower()
        category_bonus = 0.18 if category.replace("_", " ") in prompt_text else 0.0
        score_bonus = min(0.28, score * 0.04)
        keyword_bonus = min(0.16, len(keywords) * 0.015)
        return round(min(0.95, 0.48 + category_bonus + score_bonus + keyword_bonus), 2)
