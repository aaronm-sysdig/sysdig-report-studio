"""
PDF generation for Sysdig Report Studio.

Takes a report config and turns it into a proper PDF document. Charts are rendered
as images (via the shared charts module) so they look the same as the preview.
Tables get some special treatment to handle wide data without looking rubbish.
"""
import os
import tempfile
import logging
from datetime import datetime
from typing import Optional
import pandas as pd
import requests
import urllib.parse
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER, landscape, portrait
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader

# Import shared modules
from charts import chart_to_image_bytes, get_chart_dimensions
from config import get_sysdig_host

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('pdf_generator')

# =============================================================================
# WATERMARK CONFIGURATION - Adjust these values as needed
# =============================================================================
WATERMARK_PATH = "background.png"  # Path to watermark image
WATERMARK_OPACITY = 0.08           # 0.0 = invisible, 1.0 = fully opaque (try 0.05-0.15 for subtle)


class WatermarkCanvas(pdf_canvas.Canvas):
    """Custom canvas that draws watermark ON TOP of content (after page is rendered)."""

    def __init__(self, *args, watermark_info=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.watermark_info = watermark_info or {}

    def showPage(self):
        """Called when a page is complete - draw watermark on top before finalizing."""
        self._draw_watermark()
        super().showPage()

    def _draw_watermark(self):
        """Draw the watermark overlay."""
        if not os.path.exists(WATERMARK_PATH):
            return

        self.saveState()
        self.setFillAlpha(WATERMARK_OPACITY)

        # Get image dimensions
        img_reader = ImageReader(WATERMARK_PATH)
        img_width, img_height = img_reader.getSize()
        aspect_ratio = img_width / img_height

        page_width = self.watermark_info.get('page_width', 595)
        page_height = self.watermark_info.get('page_height', 842)
        orientation = self.watermark_info.get('orientation', 'portrait')

        if orientation == "landscape":
            # Scale to fit height (top to bottom)
            wm_height = page_height
            wm_width = wm_height * aspect_ratio
        else:
            # Scale to fit width (left to right)
            wm_width = page_width
            wm_height = wm_width / aspect_ratio

        # Centre the watermark on the page
        x = (page_width - wm_width) / 2
        y = (page_height - wm_height) / 2

        self.drawImage(
            WATERMARK_PATH,
            x, y,
            width=wm_width,
            height=wm_height,
            mask='auto',
            preserveAspectRatio=True
        )
        self.restoreState()


class PDFReportGenerator:
    """Generates PDF reports from report configurations."""

    def __init__(
        self,
        page_size: str = "A4",
        orientation: str = "portrait",
        margin: float = 0.4
    ):
        """Initialize the PDF generator."""
        base_size = A4 if page_size.upper() == "A4" else LETTER
        self.page_size = landscape(base_size) if orientation == "landscape" else portrait(base_size)
        self.margin = margin * inch
        self.page_width = self.page_size[0] - (2 * self.margin)
        self.page_height = self.page_size[1] - (2 * self.margin)
        self.orientation = orientation

        self.styles = getSampleStyleSheet()
        self._setup_styles()
        self.temp_dir = tempfile.mkdtemp()

    def _setup_styles(self):
        """Set up custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        self.styles.add(ParagraphStyle(
            name='ElementTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=2
        ))
        self.styles.add(ParagraphStyle(
            name='ElementDescription',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#555555'),
            spaceAfter=8
        ))
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        ))

    def generate(
        self,
        config: dict,
        output_path: str,
        api_token: str,
        region: str
    ) -> tuple[bool, Optional[str]]:
        """Generate a PDF report from a configuration."""
        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=self.page_size,
                leftMargin=self.margin,
                rightMargin=self.margin,
                topMargin=self.margin,
                bottomMargin=self.margin
            )

            story = []

            # Header with title and logo
            customer = config.get('global', {}).get('customer', '')  # Space for a suffix after customer if needs be
            logo_path = config.get('global', {}).get('logo_path', 'logo1.png')

            title_content = [
                Paragraph(f"{customer} - Security Report", self.styles['ReportTitle']),
                Paragraph(
                    f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
                    self.styles['Normal']
                )
            ]

            # Add logo if it exists, preserving aspect ratio
            if logo_path and os.path.exists(logo_path):
                # Get actual image dimensions to preserve aspect ratio
                from reportlab.lib.utils import ImageReader
                img_reader = ImageReader(logo_path)
                img_width, img_height = img_reader.getSize()
                aspect_ratio = img_width / img_height

                # Set max width for logo, calculate height to maintain ratio
                logo_max_width = 2.0 * inch
                logo_width = logo_max_width
                logo_height = logo_width / aspect_ratio

                logo_img = Image(logo_path, width=logo_width, height=logo_height)
                logo_img.hAlign = 'RIGHT'

                logo_col_width = logo_max_width + 0.2*inch
                header_table = Table(
                    [[title_content, logo_img]],
                    colWidths=[self.page_width - logo_col_width, logo_col_width]
                )
                header_table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                ]))
                story.append(header_table)
            else:
                story.extend(title_content)

            story.append(Spacer(1, 20))

            # Process template blocks, handling half-width pairs side-by-side
            blocks = config.get('template_blocks', [])
            i = 0
            while i < len(blocks):
                block = blocks[i]
                width = block.get('width', 'full')

                # Check if this and next block are both half-width
                if width == 'half' and i + 1 < len(blocks) and blocks[i + 1].get('width', 'full') == 'half':
                    # Render two half-width blocks side-by-side
                    left_content = self._render_block_content(block, api_token, region, i)
                    right_content = self._render_block_content(blocks[i + 1], api_token, region, i + 1)

                    # Create a 2-column table for side-by-side layout
                    half_width = self.page_width * 0.48
                    gap = self.page_width * 0.04
                    layout_table = Table(
                        [[left_content, right_content]],
                        colWidths=[half_width, half_width],
                        hAlign='LEFT'
                    )
                    layout_table.setStyle(TableStyle([
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 0),
                        ('RIGHTPADDING', (0, 0), (-1, -1), gap / 2),
                        ('TOPPADDING', (0, 0), (-1, -1), 0),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                    ]))
                    story.append(layout_table)
                    story.append(Spacer(1, 10))
                    i += 2
                else:
                    # Render single block full-width
                    block_content = self._render_block(block, api_token, region, i)
                    if block_content:
                        story.extend(block_content)
                    i += 1

            # Build with custom canvas that draws watermark on top of content
            watermark_info = {
                'page_width': self.page_size[0],
                'page_height': self.page_size[1],
                'orientation': self.orientation
            }

            def make_canvas(*args, **kwargs):
                return WatermarkCanvas(*args, watermark_info=watermark_info, **kwargs)

            doc.build(story, canvasmaker=make_canvas)
            self._cleanup()
            return True, None

        except Exception as e:
            self._cleanup()
            logger.error(f"PDF generation failed: {e}")
            return False, str(e)

    def _render_block_content(
        self,
        block: dict,
        api_token: str,
        region: str,
        index: int,
        target_width: float = None
    ) -> list:
        """Render block content as a list of flowables (for embedding in tables)."""
        content = []
        block_type = block.get('type', 'table')
        title = block.get('title', f'Element {index + 1}')
        description = block.get('description', '')

        # Use specified width or calculate from block setting
        if target_width is None:
            width_setting = block.get('width', 'full')
            target_width = self.page_width if width_setting == 'full' else self.page_width * 0.48

        # Add title
        content.append(Paragraph(title, self.styles['ElementTitle']))

        # Add description if present
        if description:
            content.append(Paragraph(f"<i>{description}</i>", self.styles['ElementDescription']))

        # Fetch data for this block
        if block_type == 'history':
            days = block.get('days', 7)
            data, error = self._fetch_history_data(api_token, region, days)
            if error:
                content.append(Paragraph(f"Error: {error}", self.styles['Normal']))
                return content
            if not data:
                content.append(Paragraph("No history data returned", self.styles['Normal']))
                return content
        else:
            query = block.get('query', '')
            data, error = self._fetch_data(api_token, region, query)
            if error:
                content.append(Paragraph(f"Error: {error}", self.styles['Normal']))
                return content
            if not data:
                content.append(Paragraph("No data returned", self.styles['Normal']))
                return content

        df = pd.DataFrame(data)
        if df.empty:
            content.append(Paragraph("No data to display", self.styles['Normal']))
            return content

        # Handle tables
        if block_type == 'table':
            table = self._render_table(df, target_width)
            if table:
                content.append(table)
                content.append(Paragraph(f"<i>Displaying {len(df)} records</i>", self.styles['ElementDescription']))
            return content

        # For charts, use the shared charts module to create PNG
        logger.info(f"Rendering {block_type} chart: {title}")
        img_bytes = chart_to_image_bytes(df, block_type, title)

        if img_bytes:
            img_path = os.path.join(self.temp_dir, f"chart_{index}.png")
            with open(img_path, 'wb') as f:
                f.write(img_bytes)

            render_width, render_height = get_chart_dimensions(block_type)
            aspect_ratio = render_width / render_height
            img_height = target_width / aspect_ratio

            content.append(Image(img_path, width=target_width, height=img_height))
        else:
            content.append(Paragraph("Could not render chart", self.styles['Normal']))

        return content

    def _render_block(
        self,
        block: dict,
        api_token: str,
        region: str,
        index: int
    ) -> list:
        """Render a single template block to reportlab flowables.

        Returns content wrapped in KeepTogether to prevent page breaks
        between title and content.
        """
        width_setting = block.get('width', 'full')
        pdf_width = self.page_width if width_setting == 'full' else self.page_width * 0.48

        content = self._render_block_content(block, api_token, region, index, pdf_width)
        content.append(Spacer(1, 10))
        return [KeepTogether(content)]

    def _fetch_data(self, api_token: str, region: str, query: str) -> tuple:
        """Fetch data from Sysdig API."""
        host = get_sysdig_host(region)
        base_url = f"https://{host}/api/sysql/v2/query"
        encoded_query = urllib.parse.quote(query)
        url = f"{base_url}?q={encoded_query}"

        headers = {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json"
        }

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()

            if 'items' in result:
                items = result['items']
                if 'entities' in result and items:
                    allowed_fields = list(result['entities'].keys())
                    items = [
                        {k: item.get(k) for k in allowed_fields if k in item}
                        for item in items
                    ]
                return items, None
            return None, "No items in response"

        except Exception as e:
            logger.error(f"API fetch failed: {e}")
            return None, str(e)

    def _fetch_history_data(self, api_token: str, region: str, days: int) -> tuple:
        """Fetch vulnerability history from the Sysdig analytics API."""
        host = get_sysdig_host(region)
        url = f"https://{host}/api/platform/analytics/v1/data/query"

        # Calculate epoch timestamps - end is today's midnight, start is N days before
        now = datetime.now()
        end_of_today = now.replace(hour=23, minute=59, second=59, microsecond=0)
        start_date = end_of_today - pd.Timedelta(days=days)
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

        end_epoch = int(end_of_today.timestamp())
        start_epoch = int(start_date.timestamp())

        # Get system timezone name (needs IANA format like "Australia/Melbourne")
        try:
            local_tz = datetime.now().astimezone().tzinfo
            if hasattr(local_tz, 'key'):
                tz_name = local_tz.key
            else:
                tz_name = "Etc/UTC"
        except Exception:
            tz_name = "Etc/UTC"

        payload = {
            "id": "011",
            "zoneIds": [],
            "scope": [
                {"rightOperand": {"name": "StartTime", "value": str(start_epoch)}},
                {"rightOperand": {"name": "EndTime", "value": str(end_epoch)}}
            ],
            "timezone": tz_name
        }

        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()

            if 'data' in result:
                return result['data'], None
            else:
                return None, "Unexpected response format (no 'data' key)"

        except requests.exceptions.HTTPError as e:
            return None, f"API Error: {e.response.status_code}"
        except requests.exceptions.RequestException as e:
            return None, f"Request failed: {str(e)}"
        except json.JSONDecodeError:
            return None, "Invalid JSON response"

    def _render_table(self, df: pd.DataFrame, width: float) -> Optional[Table]:
        """Render data as a ReportLab table with smart column sizing and truncation."""
        try:
            df = df.copy()

            # Format datetime columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    sample = df[col].dropna().head(1)
                    if len(sample) > 0:
                        val = str(sample.iloc[0])
                        if len(val) >= 19 and 'T' in val and val[4] == '-':
                            try:
                                df[col] = pd.to_datetime(df[col]).dt.strftime('%b %d %Y %H:%M')
                            except Exception:
                                pass

            # Calculate max content length per column (header + data)
            col_max_lens = []
            for col in df.columns:
                header_len = len(str(col))
                if len(df) > 0:
                    data_max = df[col].astype(str).str.len().max()
                else:
                    data_max = 0
                col_max_lens.append(max(header_len, data_max))

            # Truncate long content and calculate proportional widths
            max_chars_per_col = 40  # Max characters before truncation
            truncated_lens = [min(length, max_chars_per_col) for length in col_max_lens]

            # Truncate actual data
            for col in df.columns:
                df[col] = df[col].astype(str).apply(
                    lambda x: x[:max_chars_per_col-2] + '..' if len(x) > max_chars_per_col else x
                )

            # Calculate proportional column widths
            total_chars = sum(truncated_lens) or 1
            col_widths = [(length / total_chars) * width for length in truncated_lens]

            # Ensure minimum width per column
            min_col_width = width * 0.08
            col_widths = [max(w, min_col_width) for w in col_widths]

            # Scale to fit total width
            scale = width / sum(col_widths)
            col_widths = [w * scale for w in col_widths]

            # Build table data
            data = [df.columns.tolist()] + df.values.tolist()

            table = Table(data, colWidths=col_widths)

            # Compact styling with smaller fonts and tighter padding
            style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#333333')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 7),
                ('TOPPADDING', (0, 0), (-1, 0), 4),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 3),
                ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                ('FONTSIZE', (0, 1), (-1, -1), 6),
                ('TOPPADDING', (0, 1), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 2),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')])
            ])
            table.setStyle(style)
            return table

        except Exception as e:
            logger.error(f"Failed to render table: {e}")
            return None

    def _cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass


def generate_report(
    config: dict,
    output_path: str,
    api_token: str,
    region: str,
    page_size: str = "A4",
    orientation: str = "portrait"
) -> tuple[bool, Optional[str]]:
    """Convenience function to generate a PDF report."""
    generator = PDFReportGenerator(page_size=page_size, orientation=orientation)
    return generator.generate(config, output_path, api_token, region)
