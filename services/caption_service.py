from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import Settings


class CaptionService:
    """Builds caption overlay frames for phrase-timed Shorts subtitles."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def chunks(self, text: str) -> list[str]:
        clean_text = re.sub(r"\s+", " ", text).strip()
        words = clean_text.split()
        if not words:
            return []
        chunk_size = max(2, self.settings.caption_words_per_chunk)
        return [" ".join(words[index:index + chunk_size]) for index in range(0, len(words), chunk_size)]

    def make_overlay(self, caption: str, elapsed: float, total_seconds: float) -> np.ndarray:
        width, height = self.settings.video_resolution
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        self._draw_vignette(draw, width, height)
        self._draw_progress(draw, width, elapsed, total_seconds)
        self._draw_caption(image, width, height, caption)
        return np.array(image)

    def _load_font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        candidates = [
            "C:/Windows/Fonts/impact.ttf",
            "C:/Windows/Fonts/ariblk.ttf",
            "C:/Windows/Fonts/NirmalaB.ttf",
            "C:/Windows/Fonts/Arialbd.ttf",
        ] if bold else [
            "C:/Windows/Fonts/Nirmala.ttf",
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
        max_width = int(width * 0.84)
        font_size = 76
        temp_draw = ImageDraw.Draw(Image.new("RGBA", (width, height)))
        lines = [caption]
        line_boxes = []

        while font_size > 44:
            font = self._load_font(font_size, bold=True)
            lines = self._wrap_lines(temp_draw, caption, font, max_width)
            line_boxes = [temp_draw.textbbox((0, 0), line, font=font, stroke_width=4) for line in lines]
            total_height = sum(box[3] - box[1] for box in line_boxes) + max(0, len(lines) - 1) * 12
            if len(lines) <= 2 and total_height < 190:
                break
            font_size -= 4

        font = self._load_font(font_size, bold=True)
        line_heights = [box[3] - box[1] for box in line_boxes]
        max_line_width = max((box[2] - box[0] for box in line_boxes), default=0)
        total_height = sum(line_heights) + max(0, len(lines) - 1) * 12
        box_padding_x = 44
        box_padding_y = 26
        box_width = min(width - 80, max_line_width + box_padding_x * 2)
        box_height = total_height + box_padding_y * 2
        box_left = (width - box_width) // 2
        box_top = int(height * 0.68) - box_height // 2
        box_right = box_left + box_width
        box_bottom = box_top + box_height

        caption_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(caption_image)
        draw.rounded_rectangle(
            [box_left + 12, box_top + 12, box_right + 12, box_bottom + 12],
            radius=28,
            fill=(0, 0, 0, 110),
        )
        draw.rounded_rectangle(
            [box_left, box_top, box_right, box_bottom],
            radius=28,
            fill=(6, 9, 15, 215),
            outline=(255, 255, 255, 48),
            width=2,
        )
        draw.rounded_rectangle([box_left, box_top, box_left + 12, box_bottom], radius=5, fill=(255, 223, 0, 240))

        current_y = box_top + box_padding_y
        for line, line_height in zip(lines, line_heights):
            words = line.split()
            word_widths = [draw.textbbox((0, 0), word, font=font, stroke_width=4)[2] for word in words]
            gap = 20
            line_width = sum(word_widths) + max(0, len(words) - 1) * gap
            x = (width - line_width) // 2
            for word_index, word in enumerate(words):
                highlight = word_index == len(words) - 1 or any(char.isdigit() for char in word)
                fill = (255, 223, 0, 255) if highlight else (255, 255, 255, 255)
                stroke = (25, 20, 0, 255) if highlight else (0, 0, 0, 245)
                draw.text((x, current_y), word, font=font, fill=fill, stroke_width=4, stroke_fill=stroke)
                x += word_widths[word_index] + gap
            current_y += line_height + 12

        rotated = caption_image.rotate(-3, resample=Image.BICUBIC, center=(width // 2, box_top + box_height // 2))
        image.alpha_composite(rotated)
