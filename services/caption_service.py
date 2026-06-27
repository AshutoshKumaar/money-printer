from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import Settings
HINDI_STOPWORDS = {
    "में", "से", "को", "ने", "का", "की", "के", "है", "हैं", "था", "थे", "थी",
    "और", "या", "लेकिन", "पर", "कि", "यह", "वह", "ये", "वे", "इस", "उस",
    "इन", "उन", "जो", "तो", "ही", "भी", "कर", "करके", "करना", "करते",
    "करता"
}


class CaptionService:
    """Builds caption overlay frames for phrase-timed Shorts subtitles."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def chunks(self, text: str) -> list[str]:
        clean_text = self._clean_caption_text(text)
        words = clean_text.split()
        if not words:
            return []
        chunk_size = max(2, self.settings.caption_words_per_chunk)
        chunks = [" ".join(words[index:index + chunk_size]) for index in range(0, len(words), chunk_size)]
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def make_overlay(self, caption: str, elapsed: float, total_seconds: float) -> np.ndarray:
        width, height = self.settings.video_resolution
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        self._draw_vignette(draw, width, height)
        self._draw_progress(draw, width, elapsed, total_seconds)
        self._draw_caption(image, width, height, caption)
        return np.array(image)

    def _load_font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        bundled_font = "NotoSansDevanagari-Bold.ttf" if bold else "NotoSansDevanagari-Regular.ttf"
        bundled_path = Path(self.settings.base_dir) / "assets" / "fonts" / bundled_font

        candidates = [
            str(bundled_path),
            "C:/Windows/Fonts/NirmalaB.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/impact.ttf",
            "C:/Windows/Fonts/ariblk.ttf",
            "C:/Windows/Fonts/Arialbd.ttf",
        ] if bold else [
            str(bundled_path),
            "C:/Windows/Fonts/Nirmala.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/Arial.ttf",
        ]
        candidates.extend(["arialbd.ttf", "arial.ttf"])
        for font_path in candidates:
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _wrap_lines(self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current: list[str] = []
        for word in words:
            test = " ".join(current + [word])
            bbox = draw.textbbox((0, 0), test, font=font, stroke_width=2)
            if current and bbox[2] - bbox[0] > max_width:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
        return lines

    def _draw_vignette(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
        for y in range(0, 360, 8):
            alpha = int(125 * (1 - y / 360))
            draw.rectangle([0, y, width, y + 8], fill=(0, 0, 0, alpha))
        for offset in range(0, 560, 8):
            alpha = int(150 * (1 - offset / 560))
            y = height - offset - 8
            draw.rectangle([0, y, width, y + 8], fill=(0, 0, 0, alpha))

    def _draw_progress(self, draw: ImageDraw.ImageDraw, width: int, elapsed: float, total_seconds: float) -> None:
        margin = 70
        bar_width = width - (margin * 2)
        bar_height = 10
        y = 62
        progress = min(1.0, max(0.0, elapsed / max(total_seconds, 0.01)))
        draw.rounded_rectangle([margin, y, margin + bar_width, y + bar_height], radius=5, fill=(255, 255, 255, 55))
        draw.rounded_rectangle(
            [margin, y, margin + int(bar_width * progress), y + bar_height],
            radius=5,
            fill=(255, 214, 72, 230),
        )

    def _draw_caption(self, image: Image.Image, width: int, height: int, text: str) -> None:
        if not text:
            return
        caption = text.upper()
        max_width = int(width * 0.88)
        font_size = max(46, int(width * 0.088))
        temp_draw = ImageDraw.Draw(Image.new("RGBA", (width, height)))
        lines = [caption]
        line_boxes = []

        while font_size > max(34, int(width * 0.052)):
            font = self._load_font(font_size, bold=True)
            lines = self._wrap_lines(temp_draw, caption, font, max_width)
            stroke_width = max(3, width // 210)
            line_boxes = [temp_draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width) for line in lines]
            total_height = sum(box[3] - box[1] for box in line_boxes) + max(0, len(lines) - 1) * 10
            if len(lines) <= 2 and total_height < int(height * 0.13):
                break
            font_size -= 3

        font = self._load_font(font_size, bold=True)
        stroke_width = max(3, width // 210)
        line_boxes = [temp_draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width) for line in lines]
        line_heights = [box[3] - box[1] for box in line_boxes]
        max_line_width = max((box[2] - box[0] for box in line_boxes), default=0)
        total_height = sum(line_heights) + max(0, len(lines) - 1) * 10
        box_padding_x = max(28, width // 28)
        box_padding_y = max(16, height // 72)
        box_width = min(width - max(44, width // 12), max_line_width + box_padding_x * 2)
        box_height = total_height + box_padding_y * 2
        box_left = (width - box_width) // 2
        box_top = int(height * 0.70) - box_height // 2
        box_right = box_left + box_width
        box_bottom = box_top + box_height

        caption_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(caption_image)
        draw.rounded_rectangle(
            [box_left + 8, box_top + 10, box_right + 8, box_bottom + 10],
            radius=16,
            fill=(0, 0, 0, 95),
        )
        draw.rounded_rectangle(
            [box_left, box_top, box_right, box_bottom],
            radius=16,
            fill=(3, 5, 9, 205),
            outline=(255, 255, 255, 35),
            width=1,
        )
        draw.rounded_rectangle(
            [box_left + 8, box_top + 8, box_left + 15, box_bottom - 8],
            radius=4,
            fill=(255, 214, 0, 240),
        )

        current_y = box_top + box_padding_y
        for line, line_height in zip(lines, line_heights):
            words = line.split()
            word_boxes = [draw.textbbox((0, 0), word, font=font, stroke_width=stroke_width) for word in words]
            word_widths = [box[2] - box[0] for box in word_boxes]
            gap = max(8, width // 65)
            line_width = min(max_width, sum(word_widths) + max(0, len(words) - 1) * gap)
            x = (width - line_width) // 2
            for word_index, word in enumerate(words):
                highlight = word_index == len(words) - 1 or any(char.isdigit() for char in word) or len(word) >= 8
                fill = (255, 223, 0, 255) if highlight else (255, 255, 255, 255)
                stroke = (25, 20, 0, 255) if highlight else (0, 0, 0, 245)
                draw.text((x, current_y), word, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke)
                x += word_widths[word_index] + gap
            current_y += line_height + 10

        image.alpha_composite(caption_image)

    @staticmethod
    def _clean_caption_text(text: str) -> str:
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"[#*_`~|<>]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def generate_ass_header(self) -> str:
        return """[Script Info]
; Script generated by Antigravity
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans Devanagari,90,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,3,2,100,100,384,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def format_ass_dialogue(self, text: str, start_time: float, end_time: float) -> list[str]:
        caption = text.upper()
        width, height = self.settings.video_resolution
        max_width = int(width * 0.88)
        font_size = max(46, int(width * 0.088))
        temp_draw = ImageDraw.Draw(Image.new("RGBA", (width, height)))
        
        while font_size > max(34, int(width * 0.052)):
            font = self._load_font(font_size, bold=True)
            lines = self._wrap_lines(temp_draw, caption, font, max_width)
            stroke_width = max(3, width // 210)
            line_boxes = [temp_draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width) for line in lines]
            total_height = sum(box[3] - box[1] for box in line_boxes) + max(0, len(lines) - 1) * 10
            if len(lines) <= 2 and total_height < int(height * 0.13):
                break
            font_size -= 3
            
        font = self._load_font(font_size, bold=True)
        
        # Collect all words and their line/word indices
        words_map = []
        for line_idx, line in enumerate(lines):
            for word_idx, w in enumerate(line.split()):
                words_map.append((line_idx, word_idx, w))
                
        if not words_map:
            start_str = self._to_ass_timestamp(start_time)
            end_str = self._to_ass_timestamp(end_time)
            return [f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,"]
            
        dialogue_lines = []
        chunk_duration = end_time - start_time
        N = len(words_map)
        word_dur = chunk_duration / N
        for i, (hl_line_idx, hl_word_idx, _) in enumerate(words_map):
            w_start = start_time + i * word_dur
            w_end = start_time + (i + 1) * word_dur
            
            # Format the lines with the i-th word highlighted
            ass_lines = []
            for line_idx, line in enumerate(lines):
                words = line.split()
                ass_words = []
                for word_idx, word in enumerate(words):
                    if line_idx == hl_line_idx and word_idx == hl_word_idx:
                        # Highlight in yellow (hex format BGR &H00D4FF&)
                        ass_words.append(f"{{\\c&H00D4FF&}}{word}{{\\c}}")
                    else:
                        ass_words.append(word)
                ass_lines.append(" ".join(ass_words))
                
            ass_text = "\\N".join(ass_lines)
            start_str = self._to_ass_timestamp(w_start)
            end_str = self._to_ass_timestamp(w_end)
            dialogue_lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{ass_text}")
            
        return dialogue_lines

    def _to_ass_timestamp(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int(round((seconds - int(seconds)) * 100))
        if centiseconds == 100:
            secs += 1
            centiseconds = 0
            if secs == 60:
                minutes += 1
                secs = 0
                if minutes == 60:
                    hours += 1
                    minutes = 0
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"
