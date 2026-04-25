# Sysdig 2026 Brand Token Reference

This file documents the mapping between Sysdig 2026 brand token names, their
CSS custom properties in `sysdig-brand.css`, and the corresponding Word style
names in `reference.docx`.

## Color Tokens

| Brand Name | Hex | CSS Custom Property | Usage |
|---|---|---|---|
| Lumin | `#BDF78B` | `--lumin` | Accent: H3 bottom border, horizontal rules only — NOT used as a background fill |
| White | `#FFFFFF` | `--white` | Light bg, text on Deep See |
| Black | `#000000` | `--black` | Body text, text on Lumin |
| Deep See | `#01353E` | `--deep-see` | H1/H2 background, table header background, H3/H4 text color, Title text color |
| Grey 10 | `#EAEBED` | `--grey-10` | Alternate table row bg, inline code bg |
| Grey 20 | `#D4D6D9` | `--grey-20` | Light borders |
| Grey 30 | `#BEC0C5` | `--grey-30` | Table cell borders |
| Grey 40 | `#A8ABB1` | `--grey-40` | Disabled states |
| Grey 50 | `#92959D` | `--grey-50` | Placeholder text |
| Grey 60 | `#6E7178` | `--grey-60` | H5/H6, blockquote text, footer text |
| Grey 70 | `#4A4D53` | `--grey-70` | Secondary body text |
| Grey 80 | `#26282E` | `--grey-80` | Dark backgrounds |
| Grey 90 | `#121217` | `--grey-90` | Code block background |
| Falco Blue | `#00CBE2` | `--falco-blue` | Informational highlights |
| Yellow | `#FDD835` | `--yellow` | Warning highlights |
| Orange | `#FFA940` | `--orange` | Caution highlights |
| Red | `#FF7774` | `--red` | Error/critical highlights |
| Purple | `#CA87DA` | `--purple` | Feature/capability highlights |

**Contrast rule**: minimum 3 grey-steps between background and text.
**Deep See rule**: always pair Deep See with Lumin in the same composition.

## Typography

| Element | Size | Weight | CSS selector |
|---|---|---|---|
| H1 | 24pt | 700 | `h1` |
| H2 | 18pt | 700 | `h2` |
| H3 | 14pt | 700 | `h3` |
| H4 | 12pt | 700 | `h4` |
| H5/H6 | 11pt | 700 | `h5, h6` |
| Body | 11pt | 400 | `body` |
| Code | 0.9em | 400 | `code` |

Font family: **Inter** (Google Fonts, weights 300–700).

## Cover Page Style Exceptions

The document body uses **Light Mode** (white `#FFFFFF` background, black `#000000` text). The cover page is the **one permitted exception**: it uses **Deep See Mode** — a full Deep See `#01353E` background with white foreground text and the white Sysdig logo.

| Element | Rule | Exception |
|---|---|---|
| Body pages | White background, black text (Light Mode) | — |
| Cover page | Deep See background, white text (Deep See Mode) | Only exception to Light Mode |
| Cover page logo | White variant (`sysdig-logo-white.png`) | Use white logo on Deep See bg only |
| Body/light bg logo | Black variant (`sysdig-logo-black.png`) | Use black logo on white/light bg |

### Logo Usage Rules

- **White logo** (`templates/sysdig-logo-white.png`, `templates/sysdig-logo-white.svg`): use exclusively on Deep See `#01353E` or other dark backgrounds.
- **Black logo** (`templates/sysdig-logo-black.png`, `templates/sysdig-logo-black.svg`): use on white, light grey, or Lumin `#BDF78B` backgrounds.
- Never place the white logo on a light background or the black logo on a dark background.
- If the logo file is absent at export time, the system falls back to the `sysdig` text wordmark and logs a warning.

## Word Style Name Mapping

When editing `reference.docx` in Word, apply tokens to these named styles:

| Word Style Name | Token Applied | Value | Scope |
|---|---|---|---|
| Title | Background | White `#FFFFFF` | Cover page only |
| Title | Font color | Deep See `#01353E` | Cover page only |
| Title | Font size | 36pt bold | Cover page only |
| Heading 1 | Background fill | Deep See `#01353E` | Body |
| Heading 1 | Font color | White `#FFFFFF` | Body |
| Heading 2 | Background fill | Deep See `#01353E` | Body |
| Heading 2 | Font color | White `#FFFFFF` | Body |
| Heading 3 | Font color | Deep See `#01353E` | Body |
| Heading 3 | Bottom border | Lumin `#BDF78B`, 2pt | Body |
| Heading 4 | Font color | Deep See `#01353E` | Body |
| Normal | Font | Inter, 11pt | Body |
| Table Header Row | Background fill | Deep See `#01353E` | Body |
| Table Header Row | Font color | White `#FFFFFF` | Body |
| Table Header Row | Font weight | Bold | Body |
| Footer | Content (left) | Sysdig Inc. Proprietary Information | All pages |
| Footer | Content (centre) | `sysdig` (lowercase, bold) | All pages |
| Footer | Content (right) | Page number | All pages |

## Generating reference.docx

```bash
# Requires Pandoc installed
pandoc --print-default-data-file reference.docx > templates/reference.docx
```

Then open `templates/reference.docx` in Word and apply the style mappings
from the table above. Save and commit the result.

**Checklist** (tick off each style after applying in Word — items marked `[ ]` have not yet been applied to the committed `reference.docx`):

- [x] Title — Deep See background, white text, Inter 24pt bold (cover page)
- [x] Subtitle — Deep See background, white text, Inter 16pt (cover page)
- [ ] Heading 1 — Deep See background, white text, Inter font
- [ ] Heading 2 — Deep See background, white text, Inter font
- [ ] Heading 3 — Deep See text, Lumin bottom border, Inter font
- [ ] Heading 4 — Deep See text, Inter font
- [ ] Normal — Inter 11pt
- [ ] Table Header Row — Lumin background, black bold text
- [ ] Footer — left/centre/right content, two thin horizontal bars flanking page number

## Severity Colour Palette (Vulnerability Branding)

Sysdig's vulnerability UI uses a distinct palette from the site-wide brand colours. This palette is used for severity pills, severity-stacked bars, and anywhere a finding's severity is visually encoded.

| Severity | Hex | CSS custom property | Chart colour constant |
|---|---|---|---|
| Critical | `#cb87da` | `--severity-critical` | `CHART_COLORS.severityCritical` |
| High | `#ff7875` | `--severity-high` | `CHART_COLORS.severityHigh` |
| Medium | `#ffaa40` | `--severity-medium` | `CHART_COLORS.severityMedium` |
| Low | `#fdd836` | `--severity-low` | `CHART_COLORS.severityLow` |
| Negligible | `#b5c4cc` | `--severity-negligible` | `CHART_COLORS.severityNegligible` |

**Note:** these are NOT the site-brand `--red`, `--orange`, `--yellow`, `--purple` tokens — those remain defined for non-severity contexts (feature highlights, informational pills, etc). The severity palette was hand-picked from the live Sysdig Vulnerability Findings UI to ensure visual consistency with the parent product.