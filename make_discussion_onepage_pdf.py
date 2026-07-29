"""Generate the one-page professor discussion handout.

The PDF intentionally uses only the hard-CRC primary figures and numbers.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output/pdf/PROFESSOR_DISCUSSION_ONEPAGE_KO.pdf"
FONT_PATH = Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf")

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2A6F97")
TEAL = colors.HexColor("#00798C")
RED = colors.HexColor("#D1495B")
GOLD = colors.HexColor("#E9C46A")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#5B6670")
LIGHT = colors.HexColor("#F4F7F9")
LINE = colors.HexColor("#D8E1E8")


def trimmed_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    background = Image.new("RGB", image.size, "white")
    difference = ImageChops.difference(image, background)
    bbox = difference.getbbox()
    if bbox is None:
        return image
    pad = 10
    left = max(0, bbox[0] - pad)
    top = max(0, bbox[1] - pad)
    right = min(image.width, bbox[2] + pad)
    bottom = min(image.height, bbox[3] + pad)
    return image.crop((left, top, right, bottom))


def draw_image_fit(c, image, x, y, width, height):
    iw, ih = image.size
    scale = min(width / iw, height / ih)
    dw, dh = iw * scale, ih * scale
    c.drawInlineImage(
        image,
        x + (width - dw) / 2,
        y + (height - dh) / 2,
        width=dw,
        height=dh,
    )


def text(c, value, x, y, size=9, color=INK, font="AppleGothic"):
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawString(x, y, value)


def centered(c, value, x, y, width, size=9, color=INK):
    c.setFont("AppleGothic", size)
    c.setFillColor(color)
    c.drawCentredString(x + width / 2, y, value)


def wrapped(c, lines, x, y, size=8.4, leading=11.5, color=INK):
    c.setFillColor(color)
    c.setFont("AppleGothic", size)
    cursor = y
    for line in lines:
        c.drawString(x, cursor, line)
        cursor -= leading
    return cursor


def box(c, x, y, width, height, title, title_color=BLUE):
    c.setFillColor(colors.white)
    c.setStrokeColor(LINE)
    c.setLineWidth(0.8)
    c.roundRect(x, y, width, height, 7, fill=1, stroke=1)
    c.setFillColor(title_color)
    c.roundRect(x, y + height - 24, width, 24, 7, fill=1, stroke=0)
    c.rect(x, y + height - 24, width, 7, fill=1, stroke=0)
    text(c, title, x + 10, y + height - 17, 10.2, colors.white)


def build():
    if not FONT_PATH.exists():
        raise FileNotFoundError(FONT_PATH)
    pdfmetrics.registerFont(TTFont("AppleGothic", str(FONT_PATH)))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    page_width, page_height = landscape(A4)
    c = canvas.Canvas(str(OUTPUT), pagesize=(page_width, page_height))
    c.setTitle("Hard-CRC learned-prior decoding - professor discussion")
    c.setAuthor("Belief-to-Knowledge project")

    # Header
    c.setFillColor(NAVY)
    c.rect(0, page_height - 67, page_width, 67, fill=1, stroke=0)
    text(
        c,
        "Hard-CRC learned-prior decoding: latency와 reliability의 교환",
        24,
        page_height - 29,
        18.5,
        colors.white,
    )
    text(
        c,
        "핵심: 같은 MCS를 약 0.6 dB 더 낮은 SNR까지 유지해 강등을 유예",
        24,
        page_height - 51,
        10.5,
        colors.HexColor("#D9EDF7"),
    )
    text(c, "교수 디스커션용 | 2026-07-29", page_width - 184, page_height - 50, 8.2, colors.white)

    # Two primary figures
    margin = 24
    gap = 14
    graph_y = 291
    graph_h = 214
    graph_w = (page_width - 2 * margin - gap) / 2
    left_x = margin
    right_x = margin + graph_w + gap

    for x, label in (
        (left_x, "1  Fixed latency SNR-BLER (L_den=50)"),
        (right_x, "2  Equal-BP budget codec duel"),
    ):
        c.setFillColor(colors.white)
        c.setStrokeColor(LINE)
        c.roundRect(x, graph_y, graph_w, graph_h + 24, 6, fill=1, stroke=1)
        c.setFillColor(LIGHT)
        c.roundRect(x, graph_y + graph_h, graph_w, 24, 6, fill=1, stroke=0)
        c.rect(x, graph_y + graph_h, graph_w, 6, fill=1, stroke=0)
        text(c, label, x + 10, graph_y + graph_h + 7, 9.7, NAVY)

    fixed = trimmed_image(ROOT / "fixed_latency_snr_hardcrc_lden50.png")
    duel = trimmed_image(ROOT / "low_budget_codec_duel_hardcrc.png")
    draw_image_fit(c, fixed, left_x + 5, graph_y + 4, graph_w - 10, graph_h - 7)
    draw_image_fit(c, duel, right_x + 5, graph_y + 4, graph_w - 10, graph_h - 7)

    # Main takeaway band
    band_y = 257
    c.setFillColor(colors.HexColor("#EAF3F8"))
    c.setStrokeColor(colors.HexColor("#B8D5E5"))
    c.roundRect(margin, band_y, page_width - 2 * margin, 25, 6, fill=1, stroke=1)
    text(c, "RX-only: PixelCNN 지배", margin + 12, band_y + 8, 9.1, NAVY)
    text(c, "TX 포함: PixelCNN 30,576 > latency budget", margin + 185, band_y + 8, 9.1, NAVY)
    text(c, "L=1100: ours/WebP 교차", margin + 500, band_y + 8, 9.1, NAVY)

    # Bottom boxes
    left_w = 383
    right_x = margin + left_w + gap
    right_w = page_width - margin - right_x

    box(c, margin, 139, left_w, 108, "성능 격차 분해 (BP-100, BLER 0.1)", BLUE)
    text(c, "raw + BP", margin + 15, 204, 8.7, MUTED)
    text(c, "-2.331 dB", margin + 126, 204, 9.2, INK)
    text(c, "ours", margin + 15, 186, 8.7, RED)
    text(c, "-2.940 dB", margin + 126, 186, 9.2, INK)
    text(c, "PixelCNN-MAX", margin + 15, 168, 8.7, TEAL)
    text(c, "-3.826 dB", margin + 126, 168, 9.2, INK)
    text(c, "source prior", margin + 225, 204, 8.4, MUTED)
    text(c, "+0.608 dB", margin + 307, 204, 9.4, RED)
    text(c, "compression/rate", margin + 225, 186, 8.4, MUTED)
    text(c, "+1.494 dB", margin + 307, 186, 9.4, TEAL)
    text(c, "잔여 격차", margin + 225, 168, 8.4, MUTED)
    text(c, "0.886 dB", margin + 307, 168, 9.4, NAVY)
    text(c, "rate 이득이 source prior보다 지배적 - 정면 역전보다 trade-off 지도", margin + 15, 150, 8.2, MUTED)

    box(c, margin, 30, left_w, 99, "Robustness와 한계", RED)
    wrapped(
        c,
        [
            "• LUT는 SNR·seed에 대체로 강건: budget 50 재현 양호",
            "• budget 20은 local plateau - 작은 이득은 source call 19→3의 구조적 결과",
            "• budget 50 권고 rho: 0.90→0.85 (독립 seed 0/8, 0/33 파괴/구제)",
            "• 한계: L_den 의존성; budget 100 knee에서 seed 분산 큼",
        ],
        margin + 12,
        91,
        8.0,
        13.0,
    )

    box(c, right_x, 174, right_w, 73, "링크 적응 프레임", TEAL)
    wrapped(
        c,
        [
            "기여: 같은 MCS를 약 0.6 dB 더 낮은 SNR까지 유지 - 강등 유예.",
            "가치: MCS 격자가 성기거나 feedback이 제한된 broadcast·위성 링크.",
            "한계: MCS 강등과 정면 비교하면 code-rate 이득이 지배해 이기기 어려움.",
        ],
        right_x + 12,
        207,
        7.8,
        12.2,
    )

    box(c, right_x, 30, right_w, 134, "교수님께 여쭐 세 가지", NAVY)
    wrapped(
        c,
        [
            "1. L_den=50 가정은 목표 하드웨어의 병렬성·메모리 이동까지 포함해",
            "   방어 가능한가? 실측 latency profiling이 논문 필수인가?",
            "2. 타겟 시나리오에서 송신 지연과 수신 지연 중 무엇이 지배적인가?",
            "   PixelCNN TX=30,576 vs ours TX=0을 시스템 기여로 인정할 수 있는가?",
            "3. 논문 방향은 메커니즘·한계를 밝히는 characterization인가,",
            "   broadcast/위성 등 특정 시나리오의 positive result인가?",
        ],
        right_x + 12,
        132,
        8.0,
        15.0,
    )

    # Footer
    c.setStrokeColor(LINE)
    c.line(margin, 19, page_width - margin, 19)
    text(
        c,
        "조건: AWGN + perfect CSI, payload 6272, N=12600, hard CRC, Wilson 95% CI. "
        "Latency는 추상 critical-path 모델; fading은 재측정 보류.",
        margin,
        7,
        6.9,
        MUTED,
    )
    text(c, "1 / 1", page_width - 44, 7, 6.9, MUTED)

    c.showPage()
    c.save()
    return OUTPUT


if __name__ == "__main__":
    print(build())
