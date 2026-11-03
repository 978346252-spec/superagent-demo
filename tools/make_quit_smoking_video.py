#!/usr/bin/env python3
"""Render a 10-second vertical quit-smoking short video with ffmpeg.

Everything is drawn procedurally (Pillow + numpy) and streamed to ffmpeg, so the
piece needs no footage, stock assets or external graphics.

Story (1080x1920, 30 fps, 10.0 s):
  0.0 - 4.2 s  a cigarette glows in the dark; smoke rises; "每一口，都在燃烧时间"
  4.2 - 4.8 s  the cigarette is stubbed out, the ember dies, the smoke thins
  4.8 - 7.4 s  clean morning light; "掐灭它，只需要 10 秒"
  7.4 - 10.0 s the closing card: "戒烟，从这一刻开始"

Usage:
  python make_quit_smoking_video.py --out out/quit-smoking-10s.mp4
"""

from __future__ import annotations

import argparse
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1080, 1920
FPS = 30
DURATION = 10.0
FRAMES = int(round(DURATION * FPS))

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
    r"C:\Windows\Fonts\SourceHanSansCN-Normal.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]


def load_font_path() -> str:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise SystemExit("No CJK font found; expected one of: " + ", ".join(FONT_CANDIDATES))


FONT_PATH = load_font_path()
_FONT_CACHE: dict[tuple[int, str], ImageFont.FreeTypeFont] = {}


def font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    key = (size, weight)
    if key not in _FONT_CACHE:
        f = ImageFont.truetype(FONT_PATH, size)
        try:
            f.set_variation_by_name(weight)
        except Exception:
            pass
        _FONT_CACHE[key] = f
    return _FONT_CACHE[key]


# --------------------------------------------------------------------------- #
# easing
# --------------------------------------------------------------------------- #

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def ease_out_cubic(x: float) -> float:
    x = clamp01(x)
    return 1.0 - (1.0 - x) ** 3


def ease_in_out(x: float) -> float:
    x = clamp01(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def smoothstep(a: float, b: float, x: float) -> float:
    if b == a:
        return 1.0 if x >= b else 0.0
    t = clamp01((x - a) / (b - a))
    return t * t * (3 - 2 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# --------------------------------------------------------------------------- #
# background
# --------------------------------------------------------------------------- #

def radial_background(center: tuple[float, float], inner: tuple[float, float, float],
                      outer: tuple[float, float, float], radius: float) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / radius
    t = np.clip(d, 0.0, 1.0) ** 0.85
    arr = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(3):
        arr[..., i] = inner[i] + (outer[i] - inner[i]) * t
    return arr.astype(np.uint8)


def vertical_background(top: tuple[int, int, int], bottom: tuple[int, int, int]) -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None, None]
    top_arr = np.array(top, dtype=np.float32)[None, None, :]
    bottom_arr = np.array(bottom, dtype=np.float32)[None, None, :]
    arr = top_arr + (bottom_arr - top_arr) * ramp
    return np.broadcast_to(arr, (H, W, 3)).astype(np.uint8).copy()


def cloud_texture(seed: int, size: tuple[int, int] = (512, 512), blur: float = 2.2) -> np.ndarray:
    """Fine-grained cloud texture, baked once at working resolution so puffs only need a crop/resize."""
    rng = np.random.default_rng(seed)
    small = (rng.random((size[1] // 8, size[0] // 8)) * 255).astype(np.uint8)
    img = Image.fromarray(small, "L").filter(ImageFilter.GaussianBlur(blur))
    img = img.resize(size, Image.Resampling.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(1.0))

    def boost(arr: np.ndarray) -> np.ndarray:
        return np.clip((arr / 255.0) ** 1.4, 0.0, 1.0)

    return boost(np.asarray(img).astype(np.float32))


# --------------------------------------------------------------------------- #
# smoke column
# --------------------------------------------------------------------------- #

_PUFF_CACHE: dict[int, np.ndarray] = {}


def puff_shape(texture: np.ndarray, radius: float) -> np.ndarray:
    """Soft elliptical puff shape (0..1) at a quantized radius; cached so frames stay cheap."""
    bucket = max(12, min(320, int(round(radius / 8.0)) * 8))
    cached = _PUFF_CACHE.get(bucket)
    if cached is not None:
        return cached
    th, tw = texture.shape
    w = bucket * 2
    h = int(bucket * 2 * 1.25)
    # Crop the baked texture to the puff aspect, then resize: cheaper and more detailed than
    # upscaling a tiny noise tile.
    crop_w = min(tw, th)
    crop_h = min(th, int(round(crop_w / w * h)))
    crop = Image.fromarray((texture[:crop_h, :crop_w] * 255).astype(np.uint8), "L")
    mask = np.asarray(crop.resize((w, h), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    norm = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    mask *= np.clip(1.0 - norm, 0.0, 1.0) ** 0.75
    _PUFF_CACHE[bucket] = mask
    return mask


@dataclass
class Puff:
    """A particle of a rising smoke column; its x is the offset from the column axis at birth."""
    x_off: float
    birth: float
    life: float
    r0: float
    growth: float
    drift: float
    sway_amp: float
    sway_phase: float
    alpha: float


class SmokeColumn:
    def __init__(self, seed: int, origin: tuple[float, float], length: float = 640.0,
                 count: int = 110, life: tuple[float, float] = (1.4, 2.4),
                 radius: tuple[float, float] = (34.0, 62.0), alpha: float = 0.55):
        rng = random.Random(seed)
        self.origin = origin
        self.length = length
        self.puffs = [
            Puff(
                x_off=rng.uniform(-14.0, 14.0),
                birth=rng.uniform(-3.0, DURATION - 0.2),
                life=rng.uniform(*life),
                r0=rng.uniform(*radius),
                growth=rng.uniform(0.7, 1.6),
                drift=rng.uniform(-0.06, 0.06),
                sway_amp=rng.uniform(6.0, 22.0),
                sway_phase=rng.uniform(0, math.tau),
                alpha=rng.uniform(0.55, 1.0) * alpha,
            )
            for _ in range(count)
        ]

    def draw(self, mask: np.ndarray, t: float, texture: np.ndarray, fade: float = 1.0,
             scale: int = 2) -> None:
        """Accumulate the column into an `H/scale x W/scale` float mask; tint once afterwards."""
        if fade <= 0.01:
            return
        h2, w2 = mask.shape
        ox, oy = self.origin[0] / scale, self.origin[1] / scale
        span = self.length / scale
        for p in self.puffs:
            age = t - p.birth
            if age <= 0 or age >= p.life:
                continue
            life_t = age / p.life
            # Each particle travels the column over its lifetime; opacity fades in and out.
            y = oy - span * life_t
            x = ox + p.x_off / scale + (p.drift * life_t + math.sin(p.sway_phase + age * 1.9)
                                        * p.sway_amp * 0.015) * (span * 0.05)
            radius = p.r0 * (1.0 + p.growth * life_t) / scale
            opacity = math.sin(math.pi * life_t) ** 0.8
            a = p.alpha * opacity * fade
            if a <= 0.004:
                continue
            sprite = puff_shape(texture, radius)
            hh, ww = sprite.shape
            x0 = int(round(x - ww / 2))
            y0 = int(round(y - hh / 2))
            x1, y1 = x0 + ww, y0 + hh
            sx0, sy0 = max(0, -x0), max(0, -y0)
            sx1, sy1 = ww - max(0, x1 - w2), hh - max(0, y1 - h2)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            region = mask[max(0, y0):min(h2, y1), max(0, x0):min(w2, x1)]
            np.maximum(region, sprite[sy0:sy1, sx0:sx1] * a, out=region)


# --------------------------------------------------------------------------- #
# cigarette
# --------------------------------------------------------------------------- #

CIG_TOP = (474.0, 716.0)
CIG_TIP = (600.0, 1382.0)
CIG_HALF_WIDTH = 56.0
FILTER_FRACTION = 0.16

PAPER = (238, 232, 220)
PAPER_LIGHT = (248, 244, 234)
PAPER_DARK = (188, 180, 167)
FILTER_C = (208, 181, 144)
FILTER_LIGHT = (226, 201, 165)
FILTER_DARK = (178, 148, 110)


def _cyl_color(u: np.ndarray, light: tuple[int, int, int], base: tuple[int, int, int],
               dark: tuple[int, int, int]) -> np.ndarray:
    """u in 0..1 across the cylinder width -> RGB, with a light band left of centre and a dark edge."""
    u = np.clip(u, 0.0, 1.0)
    edge = 1.0 - np.clip(np.abs(u - 0.5) * 2.0, 0.0, 1.0) ** 0.7
    hot = np.clip(1.0 - np.abs(u - 0.3) * 4.0, 0.0, 1.0)
    lc = np.array(light, dtype=np.float32)
    bc = np.array(base, dtype=np.float32)
    dc = np.array(dark, dtype=np.float32)
    out = dc + (bc - dc) * edge[..., None]
    return out + (lc - out) * (hot[..., None] * 0.85)


def _cig_axes(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sink = ease_in_out(clamp01((t - 4.2) / 0.55))
    base_top = np.array(CIG_TOP, dtype=np.float32)
    base_tip = np.array(CIG_TIP, dtype=np.float32)
    pivot = np.array([555.0, 1050.0], dtype=np.float32)
    offset = np.array([-16.0, 54.0], dtype=np.float32) * sink
    scale = 1.0 - 0.05 * sink
    top = pivot + (base_top - pivot) * scale + offset
    tip = pivot + (base_tip - pivot) * scale + offset
    axis = (tip - top) / np.linalg.norm(tip - top)
    perp = np.array([-axis[1], axis[0]], dtype=np.float32)
    return top, tip, perp


def _distance_to_segment(px: np.ndarray, py: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab) or 1.0
    t = ((px - a[0]) * ab[0] + (py - a[1]) * ab[1]) / denom
    t = np.clip(t, 0.0, 1.0)
    dx = px - (a[0] + ab[0] * t)
    dy = py - (a[1] + ab[1] * t)
    return np.sqrt(dx * dx + dy * dy)


def _capsule(center: np.ndarray, axis: np.ndarray, perp: np.ndarray, length: float,
             half_width: float) -> list[tuple[float, float]]:
    """Polygon of a capsule (rectangle + both round caps), as a fine polygon."""
    start = center - axis * length / 2
    end = center + axis * length / 2
    pts: list[tuple[float, float]] = []
    for t in np.linspace(0.0, math.pi, 14):
        p = end + perp * math.cos(t) * half_width + axis * math.sin(t) * half_width
        pts.append((float(p[0]), float(p[1])))
    for t in np.linspace(math.pi, math.tau, 14):
        p = start + perp * math.cos(t) * half_width + axis * math.sin(t) * half_width
        pts.append((float(p[0]), float(p[1])))
    return pts


def _draw_cigarette_body(t: float) -> np.ndarray:
    """Cigarette layer (RGBA uint8, transparent outside) at time t."""
    top, tip, perp = _cig_axes(t)
    appear = ease_out_cubic(smoothstep(0.15, 0.9, t))
    axis = (tip - top) / np.linalg.norm(tip - top)
    length = float(np.linalg.norm(tip - top))
    half_width = CIG_HALF_WIDTH

    half = half_width + 4
    y0 = max(0, int(top[1] - half))
    y1 = min(H, int(tip[1] + half))
    x0 = max(0, int(min(top[0], tip[0]) - half))
    x1 = min(W, int(max(top[0], tip[0]) + half) + 1)

    body_len = length * (1.0 - FILTER_FRACTION) - half_width
    body_center = (top + tip) / 2 + axis * (length * (FILTER_FRACTION / 2) - half_width / 2)
    filter_center = top + axis * (length * FILTER_FRACTION / 2)
    filter_len = length * FILTER_FRACTION + 2

    mask_img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask_img).polygon(_capsule(body_center, axis, perp, body_len, half_width), fill=255)
    fmask_img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(fmask_img).polygon(_capsule(filter_center, axis, perp, filter_len, half_width), fill=255)
    mask = np.asarray(mask_img.crop((x0, y0, x1, y1))).astype(np.float32) / 255.0
    fmask = np.asarray(fmask_img.crop((x0, y0, x1, y1))).astype(np.float32) / 255.0
    mask = np.maximum(mask, fmask)

    # Cylinder shading along the perpendicular axis.
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    rel = (xx - body_center[0]) * perp[0] + (yy - body_center[1]) * perp[1]
    u = rel / (2 * half_width) + 0.5
    body_col = _cyl_color(u, PAPER_LIGHT, PAPER, PAPER_DARK)
    filt_col = _cyl_color(u, FILTER_LIGHT, FILTER_C, FILTER_DARK)
    col = body_col + (filt_col - body_col) * fmask[..., None]

    # Seam where the filter meets the paper.
    seam_a = _distance_to_segment(xx, yy, filter_center + axis * (filter_len / 2),
                                  filter_center + axis * (filter_len / 2) - axis * 4.0)
    seam = np.clip(1.0 - seam_a / 2.2, 0.0, 1.0)
    seam_k = (0.45 * seam)[..., None]
    col = col * (1.0 - seam_k) + np.array([150.0, 126.0, 96.0], dtype=np.float32) * seam_k

    layer = np.zeros((H, W, 4), dtype=np.uint8)
    region = layer[y0:y1, x0:x1].astype(np.float32)
    region[..., :3] = col
    region[..., 3:] = 255.0 * mask[..., None] * appear
    layer[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)
    return layer


_CIG_STEP = 0.1
_CIG_CACHE: dict[int, tuple[Image.Image, tuple[float, float]]] = {}


def cigarette(t: float) -> tuple[Image.Image, tuple[float, float]]:
    """Cached sprite + ember position (the ember only moves during the small sink)."""
    key = int(round(t / _CIG_STEP))
    hit = _CIG_CACHE.get(key)
    if hit is not None:
        return hit
    tt = key * _CIG_STEP
    img = Image.fromarray(_draw_cigarette_body(tt), "RGBA")
    top, tip, _ = _cig_axes(tt)
    value = (img, (float(tip[0]), float(tip[1])))
    _CIG_CACHE[key] = value
    return value


_EMBER: tuple[int, int, int, int, np.ndarray] | None = None


def _ember_patch() -> tuple[int, int, int, int, np.ndarray]:
    """Precomputed ember glow patch (x0, y0, x1, y1, rgb float) centred on the resting tip."""
    global _EMBER
    if _EMBER is not None:
        return _EMBER
    cx, cy = int(CIG_TIP[0]), int(CIG_TIP[1])
    pad = 420
    x0, x1 = max(0, cx - pad), min(W, cx + pad)
    y0, y1 = max(0, cy - pad), min(H, cy + pad)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    glow = (np.exp(-d2 / (2 * 46.0 ** 2)) * 0.50 + np.exp(-d2 / (2 * 92.0 ** 2)) * 0.32
            + np.exp(-d2 / (2 * 200.0 ** 2)) * 0.10)
    rgb = glow[..., None] * np.array([255.0, 88.0, 20.0], dtype=np.float32)[None, None, :]
    _EMBER = (x0, y0, x1, y1, rgb.astype(np.float32))
    return _EMBER


def draw_cigarette(base: np.ndarray, t: float, opacity: float = 1.0) -> None:
    """Composite the cigarette sprite and its ember glow over the background array."""
    img, (ex, ey) = cigarette(t)
    appear = ease_out_cubic(smoothstep(0.15, 0.9, t))
    if appear <= 0.01 or opacity <= 0.01:
        return
    px = np.asarray(img, dtype=np.float32)[..., :3]
    alpha = np.asarray(img.getchannel("A"), dtype=np.float32) / 255.0 * opacity
    ember_life = (1.0 - smoothstep(4.2, 4.66, t)) * opacity
    if ember_life > 0.01:
        gx0, gy0, gx1, gy1, rgb = _ember_patch()
        sx = int(round(ex - CIG_TIP[0]))
        sy = int(round(ey - CIG_TIP[1]))
        bx0, by0 = max(0, gx0 + sx), max(0, gy0 + sy)
        bx1, by1 = min(W, gx1 + sx), min(H, gy1 + sy)
        if bx1 > bx0 and by1 > by0:
            patch = rgb[by0 - gy0 - sy:by1 - gy0 - sy, bx0 - gx0 - sx:bx1 - gx0 - sx]
            breathe = 0.9 + 0.1 * math.sin(t * 5.1)
            base[by0:by1, bx0:bx1] += patch * (breathe * ember_life)
    base[:] = base * (1.0 - alpha[..., None]) + px * alpha[..., None]


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #

def draw_caption(draw: ImageDraw.ImageDraw, t: float, text: str, start: float, end: float,
                 y: int, size: int, color: tuple[int, int, int], weight: str = "Regular",
                 tracking: int = 0) -> None:
    """Fade + rise caption; `tracking` spreads glyphs for a calmer, poster-like line."""
    fade_in = smoothstep(start, start + 0.55, t)
    fade_out = 1.0 - smoothstep(end - 0.45, end, t)
    a = int(255 * fade_in * fade_out)
    if a <= 1:
        return
    rise = int(26 * (1.0 - ease_out_cubic(fade_in)))
    f = font(size, weight)
    if tracking <= 0:
        bbox = draw.textbbox((0, 0), text, font=f)
        draw.text(((W - (bbox[2] - bbox[0])) / 2 - bbox[0], y + rise), text, font=f, fill=color + (a,))
        return
    widths = [draw.textlength(ch, font=f) for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x = (W - total) / 2
    for ch, cw in zip(text, widths):
        draw.text((x, y + rise), ch, font=f, fill=color + (a,))
        x += cw + tracking


# --------------------------------------------------------------------------- #
# frame loop
# --------------------------------------------------------------------------- #

def render() -> Iterator[np.ndarray]:
    smoke_tex = cloud_texture(11)
    mote_tex = cloud_texture(29, blur=0.7)

    night = radial_background((540, 1180), (30, 32, 38), (6, 7, 10), 1500).astype(np.float32)
    day = vertical_background((226, 240, 243), (178, 208, 212)).astype(np.float32)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sun_d = np.clip(np.hypot(xx - W * 0.58, yy - H * 0.26) / 1250.0, 0, 1) ** 1.1
    sun_a = np.clip(1.0 - sun_d, 0, 1).astype(np.float32) ** 2.2
    sun_rgb = np.array([255.0, 250.0, 236.0], dtype=np.float32)

    plume = SmokeColumn(seed=3, origin=(CIG_TIP[0] - 6, CIG_TIP[1] - 34), length=700.0,
                        count=140, life=(1.5, 2.6), radius=(40.0, 74.0), alpha=0.85)
    motes = SmokeColumn(seed=17, origin=(560, 1080), length=260.0, count=22,
                        life=(4.0, 7.0), radius=(6.0, 14.0), alpha=0.18)

    scale = 2
    smoke_tint = np.array([224.0, 227.0, 236.0], dtype=np.float32)

    for i in range(FRAMES):
        t = i / FPS
        day_t = ease_in_out(smoothstep(4.55, 5.85, t))
        arr = night + (day - night) * day_t

        # A sunrise flash hides the muddy middle of the dark-to-day crossfade.
        bloom_p = clamp01((t - 4.5) / 1.05)
        bloom = math.sin(math.pi * bloom_p) ** 1.6
        if bloom > 0.002:
            arr = arr + (np.array([255.0, 252.0, 242.0], dtype=np.float32) - arr) * (bloom * 0.62)

        if day_t > 0:
            arr = arr + (sun_rgb - arr) * (sun_a * day_t)[..., None]

        smoke_fade = 1.0 - smoothstep(4.3, 5.35, t)
        if smoke_fade > 0.01:
            mask_small = np.zeros((H // scale, W // scale), dtype=np.float32)
            plume.draw(mask_small, t, smoke_tex, fade=smoke_fade, scale=scale)
            smoke_a = np.asarray(Image.fromarray(np.clip(mask_small * 255, 0, 255).astype(np.uint8), "L")
                                 .resize((W, H), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
            arr = arr + (smoke_tint - arr) * (smoke_a * 0.62)[..., None]

        if t > 5.3:
            mote_fade = smoothstep(5.3, 6.6, t)
            mask_small = np.zeros((H // scale, W // scale), dtype=np.float32)
            motes.draw(mask_small, t, mote_tex, fade=mote_fade, scale=scale)
            mote_a = np.asarray(Image.fromarray(np.clip(mask_small * 255, 0, 255).astype(np.uint8), "L")
                                .resize((W, H), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
            arr = arr + (255.0 - arr) * (mote_a * 0.3 * mote_fade)[..., None]

        # Cigarette sits on the darkness so the opening fade never dims it.
        if t < 4.9:
            draw_cigarette(arr, t, opacity=1.0 - smoothstep(4.3, 4.8, t))

        frame = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)

        draw_caption(d, t, "每一口，都在燃烧时间", 1.0, 4.2, y=1520, size=62,
                     color=(240, 235, 226), weight="Light", tracking=12)
        draw_caption(d, t, "掐灭它，只需要 10 秒", 5.45, 7.35, y=850, size=70,
                     color=(34, 48, 52), weight="Medium", tracking=6)
        draw_caption(d, t, "给肺一次重新呼吸的机会", 5.95, 7.5, y=975, size=52,
                     color=(86, 106, 110), weight="Regular", tracking=8)

        card_in = smoothstep(7.45, 8.2, t)
        card_out = 1.0 - smoothstep(9.55, 9.95, t)
        if card_in > 0 and card_out > 0:
            draw_caption(d, t, "戒烟", 7.45, 10.0, y=760, size=190,
                         color=(24, 38, 42), weight="Bold", tracking=22)
            draw_caption(d, t, "从这一刻开始", 7.85, 10.0, y=1010, size=76,
                         color=(38, 104, 90), weight="Medium", tracking=12)
            draw_caption(d, t, "把它，转给想戒烟的人", 8.35, 10.0, y=1440, size=42,
                         color=(54, 72, 76), weight="Medium", tracking=6)

        frame = Image.alpha_composite(frame, overlay).convert("RGB")
        out = np.asarray(frame, dtype=np.float32)
        fade = min(smoothstep(0.0, 0.5, t), 1.0 - smoothstep(9.6, 10.0, t))
        if fade < 0.999:
            out = out * fade
        yield out.astype(np.uint8)


def encode(out: Path, ffmpeg: str = "ffmpeg") -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
        "-profile:v", "high", "-level", "4.0", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "96k", "-shortest", str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in render():
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
    finally:
        code = proc.wait()
    if code != 0:
        raise SystemExit(f"ffmpeg exited with {code}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out/quit-smoking-10s.mp4"))
    ap.add_argument("--poster", type=Path, default=None, help="Optional still PNG path.")
    ap.add_argument("--at", type=float, default=8.8, help="Timestamp for the poster still.")
    ap.add_argument("--stills", type=Path, default=None,
                    help="Optional directory: write stills at a few key moments.")
    args = ap.parse_args()
    if args.stills:
        args.stills.mkdir(parents=True, exist_ok=True)
        marks = (0.6, 2.4, 4.45, 5.6, 6.8, 8.8)
        wanted = {int(round(m * FPS)) for m in marks}
        for i, frame in enumerate(render()):
            if i in wanted:
                Image.fromarray(frame, "RGB").save(args.stills / f"still_{i / FPS:04.1f}.png")
            if i > max(wanted):
                break
        return 0
    encode(args.out)
    if args.poster:
        frames = list(islice(render(), int(args.at * FPS) + 1))
        args.poster.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frames[-1], "RGB").save(args.poster)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
