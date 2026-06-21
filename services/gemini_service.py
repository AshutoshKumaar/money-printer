from __future__ import annotations

import json
import logging
import re
from typing import Any

from google import genai
from google.genai import types

from config import Settings
from core.models import Script, Segment
from core.retry import retry_call


class GeminiService:
    """Generates Shorts scripts, topics, and publishing metadata with Gemini."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self._quota_exhausted = False

    def generate_topic(self, recent_topics: list[str] | None = None) -> str:
        recent_topics = recent_topics or []
        if self._quota_exhausted:
            topic = self._fallback_topic(recent_topics)
            self.logger.warning("Gemini quota exhausted; using fallback topic: %s", topic)
            return topic

        recent_block = ""
        if recent_topics:
            recent_block = (
                "\nAvoid these recent topics and do not generate a close variation:\n"
                + "\n".join(f"- {topic}" for topic in recent_topics[:30])
            )
        prompt = """
        Generate one viral Hindi YouTube Shorts topic for a faceless educational channel.
        The topic should be curiosity-driven, safe for advertisers, and fit a 60-second video.
        Make it clearly different from recently used topics.
        Return only the topic text, no bullets.
        """ + recent_block
        try:
            response = retry_call(
                lambda: self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                ),
                attempts=self.settings.retry_attempts,
                backoff_seconds=self.settings.retry_backoff_seconds,
                logger=self.logger,
                label="Gemini topic generation",
            )
            topic = (response.text or "").strip().strip('"')
            if topic and self._normalize_topic(topic) not in {self._normalize_topic(item) for item in recent_topics}:
                return topic
        except Exception as exc:
            if self._is_quota_error(exc):
                self._quota_exhausted = True
                self.logger.warning("Gemini quota exhausted during topic generation; using local fallback topic")
            else:
                self.logger.warning("Gemini topic generation failed; using local fallback topic: %s", exc)
        return self._fallback_topic(recent_topics)

    def generate_script(self, topic: str) -> Script:
        if self._quota_exhausted:
            self.logger.warning("Gemini quota exhausted; using full fallback script")
            return self._fallback_script(topic)

        prompt = self._script_prompt(topic)
        try:
            response = retry_call(
                lambda: self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                ),
                attempts=self.settings.retry_attempts,
                backoff_seconds=self.settings.retry_backoff_seconds,
                logger=self.logger,
                label="Gemini script generation",
            )
            raw_text = self._clean_json_text(response.text or "")
            script = Script.from_dict(json.loads(raw_text), topic)
            script = self._normalize_script(script, topic)
            return self._fit_narration_budget(script)
        except Exception as exc:
            if self._is_quota_error(exc):
                self._quota_exhausted = True
            self.logger.error("Gemini script generation failed, using full fallback script: %s", exc)
            return self._fallback_script(topic)

    def improve_metadata(self, script: Script) -> Script:
        if self._quota_exhausted:
            self.logger.warning("Gemini quota exhausted; keeping fallback metadata")
            return script

        prompt = f"""
        Improve YouTube publishing metadata for this Hindi Shorts topic: {script.topic}
        Return JSON only with title, description, tags, hashtags.
        Keep title under 95 characters. Description should include 3-6 hashtags.
        Existing metadata:
        {json.dumps(script.to_dict(), ensure_ascii=False)}
        """
        try:
            response = retry_call(
                lambda: self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                ),
                attempts=self.settings.retry_attempts,
                backoff_seconds=self.settings.retry_backoff_seconds,
                logger=self.logger,
                label="Gemini metadata generation",
            )
            data = json.loads(self._clean_json_text(response.text or ""))
            script.title = str(data.get("title") or script.title).strip()
            script.description = str(data.get("description") or script.description).strip()
            script.tags = [str(tag).strip() for tag in data.get("tags", script.tags) if str(tag).strip()]
            script.hashtags = [
                str(tag).strip().lstrip("#")
                for tag in data.get("hashtags", script.hashtags)
                if str(tag).strip()
            ]
        except Exception as exc:
            if self._is_quota_error(exc):
                self._quota_exhausted = True
            self.logger.warning("Metadata improvement failed; keeping script metadata: %s", exc)
        return script

    def _normalize_script(self, script: Script, topic: str) -> Script:
        if len(script.segments) < self.settings.min_segments:
            fallback = self._fallback_script(topic)
            script.segments.extend(fallback.segments[len(script.segments):])
        script.segments = script.segments[: self.settings.max_segments]
        for index, segment in enumerate(script.segments, start=1):
            if not segment.image_prompt:
                segment.image_prompt = self._default_image_prompt(topic, index)
            segment.visual_type = "ai_image"
        return script

    def _fit_narration_budget(self, script: Script) -> Script:
        total_words = self._narration_word_count(script)
        if total_words <= self.settings.narration_max_words and all(
            len(segment.text.split()) <= self.settings.segment_max_words
            for segment in script.segments
        ):
            return script

        self.logger.warning(
            "Narration is over budget (%s words); compressing to at most %s words",
            total_words,
            self.settings.narration_max_words,
        )
        prompt = f"""
        Compress this Hindi Shorts script without changing its facts or visual fields.
        Return the complete JSON only.

        Hard requirements:
        - Keep all {len(script.segments)} scenes and preserve their order.
        - Keep the hook, story progression, final reveal, and CTA.
        - Spoken "text" must use at most {self.settings.segment_max_words} words per scene.
        - Total spoken "text" must use at most {self.settings.narration_max_words} words.
        - Use short, natural Devanagari Hindi sentences.
        - Preserve each scene's subtitle, search_query, and image_prompt, shortening subtitle if needed.

        Script:
        {json.dumps(script.to_dict(), ensure_ascii=False)}
        """
        try:
            response = retry_call(
                lambda: self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                ),
                attempts=self.settings.retry_attempts,
                backoff_seconds=self.settings.retry_backoff_seconds,
                logger=self.logger,
                label="Gemini narration compression",
            )
            compressed = Script.from_dict(
                json.loads(self._clean_json_text(response.text or "")),
                script.topic,
            )
            if len(compressed.segments) == len(script.segments):
                script = compressed
        except Exception as exc:
            if self._is_quota_error(exc):
                self._quota_exhausted = True
            self.logger.warning("Narration compression failed; applying deterministic word limits: %s", exc)

        return self._hard_limit_narration(script)

    def _hard_limit_narration(self, script: Script) -> Script:
        if not script.segments:
            return script
        per_scene_budget = min(
            self.settings.segment_max_words,
            max(6, self.settings.narration_max_words // len(script.segments)),
        )
        remaining = self.settings.narration_max_words
        for index, segment in enumerate(script.segments):
            scenes_left = len(script.segments) - index
            reserved = max(0, (scenes_left - 1) * 6)
            allowed = min(per_scene_budget, max(6, remaining - reserved))
            words = segment.text.split()
            if len(words) > allowed:
                if index == len(script.segments) - 1:
                    segment.text = "अपनी राय कमेंट करें, और ऐसे फैक्ट्स के लिए फॉलो करें।"
                else:
                    segment.text = " ".join(words[:allowed]).rstrip(",:;-") + "।"
            remaining -= len(segment.text.split())
        return script

    @staticmethod
    def _narration_word_count(script: Script) -> int:
        return sum(len(segment.text.split()) for segment in script.segments)

    def _fallback_topic(self, recent_topics: list[str]) -> str:
        recent = {self._normalize_topic(topic) for topic in recent_topics}
        topics = [
            "3 Unsolved Mysteries of the Deep Ocean",
            "3 Strange Facts About Black Holes",
            "3 Ancient Inventions That Were Ahead of Their Time",
            "3 Creepy Space Signals Scientists Still Cannot Explain",
            "3 Lost Cities Hidden From History",
            "3 Mind Blowing Facts About the Human Brain",
            "3 Mysterious Places Where Gravity Feels Broken",
            "3 Dangerous Animals With Secret Superpowers",
            "3 Future Technologies That May Change Your Life",
            "3 Historical Secrets Found Underwater",
        ]
        for topic in topics:
            if self._normalize_topic(topic) not in recent:
                return topic
        return f"3 Rare Science Mysteries From {len(recent) + 1} Forgotten Discoveries"

    @staticmethod
    def _normalize_topic(topic: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", topic.lower()).strip()

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "429" in text or "resource_exhausted" in text or "quota" in text

    def _script_prompt(self, topic: str) -> str:
        return f"""
        Create a professional Hindi YouTube Shorts script about: "{topic}".

        Requirements:
        - Target duration: 55 to 60 seconds.
        - Create {self.settings.min_segments} to {self.settings.max_segments} micro-scenes.
        - Total spoken narration must be at most {self.settings.narration_max_words} words.
        - Each scene's spoken "text" must contain 9 to {self.settings.segment_max_words} words.
        - Keep every scene concise so all scenes and the CTA fit in the final video.
        - First 3 seconds must be a very strong curiosity hook.
        - Use natural spoken Devanagari Hindi for "text".
        - Use clean Hinglish/Roman Hindi for "subtitle".
        - Every scene needs a vertical 9:16 cinematic "image_prompt".
        - Story should build curiosity and reveal information step by step.
        - End with a short CTA to follow/subscribe/comment.
        - Return JSON only.

        Schema:
        {{
          "title": "SEO friendly title",
          "description": "YouTube description with hashtags",
          "tags": ["shorts", "hindi", "facts"],
          "hashtags": ["Shorts", "HindiFacts"],
          "segments": [
            {{
              "text": "spoken Hindi line",
              "subtitle": "Roman Hindi subtitle",
              "visual_type": "ai_image",
              "search_query": "optional stock fallback query",
              "image_prompt": "vertical 9:16 cinematic realistic prompt"
            }}
          ]
        }}
        """

    def _fallback_script(self, topic: str) -> Script:
        templates = [
            ("रुकिए, यह सच आपको चौंका सकता है।", "Rukiye, yeh sach aapko chaunka sakta hai."),
            (f"{topic} के पीछे एक दिलचस्प कहानी छिपी है।", f"{topic} ke peeche ek dilchasp kahani chhupi hai."),
            ("पहली नजर में यह सामान्य लगता है।", "Pehli nazar mein yeh normal lagta hai."),
            ("लेकिन असली राज इसके अंदर है।", "Lekin asli raaz iske andar hai."),
            ("वैज्ञानिक भी इस पर लगातार खोज कर रहे हैं।", "Scientists bhi is par lagataar khoj kar rahe hain."),
            ("हर जवाब के साथ नया सवाल खुलता है।", "Har jawab ke saath naya sawaal khulta hai."),
            ("यही वजह है कि लोग इसे भूल नहीं पाते।", "Yahi wajah hai ki log ise bhool nahin paate."),
            ("सबसे हैरान करने वाली बात अभी बाकी है।", "Sabse hairan karne wali baat abhi baaki hai."),
            ("अगर यह सच है, तो हमारी सोच बदल सकती है।", "Agar yeh sach hai, toh hamari soch badal sakti hai."),
            ("ऐसे और रहस्यों के लिए फॉलो जरूर करें।", "Aise aur rahasyon ke liye follow zaroor karein."),
        ]
        segments = [
            Segment(
                text=text,
                subtitle=subtitle,
                image_prompt=self._default_image_prompt(topic, index),
                search_query=f"{topic} cinematic vertical",
                visual_type="ai_image",
            )
            for index, (text, subtitle) in enumerate(templates, start=1)
        ]
        return Script(
            title=f"{topic} in 60 Seconds",
            description=f"A fast Hindi Shorts explainer about {topic}. #Shorts #HindiFacts #AI",
            tags=["shorts", "hindi", "facts", "viral", topic],
            hashtags=["Shorts", "HindiFacts", "Viral"],
            segments=segments,
            topic=topic,
        )

    def _default_image_prompt(self, topic: str, index: int) -> str:
        return (
            f"Vertical 9:16 cinematic realistic scene about {topic}, scene {index}, "
            "dramatic lighting, high contrast, detailed subject, depth of field, no text"
        )

    def _clean_json_text(self, raw_text: str) -> str:
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text
