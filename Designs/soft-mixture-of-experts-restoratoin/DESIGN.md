---
name: My Design System
colors:
  surface: '#f5fbf8'
  surface-dim: '#d6dbd9'
  surface-bright: '#f5fbf8'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff5f2'
  surface-container: '#eaefec'
  surface-container-high: '#e4e9e7'
  surface-container-highest: '#dee4e1'
  on-surface: '#171d1b'
  on-surface-variant: '#3c4945'
  inverse-surface: '#2c3230'
  inverse-on-surface: '#ecf2ef'
  outline: '#6c7a75'
  outline-variant: '#bbcac4'
  surface-tint: '#006b5b'
  primary: '#006b5b'
  on-primary: '#ffffff'
  primary-container: '#07ac93'
  on-primary-container: '#00382f'
  inverse-primary: '#57dbc0'
  secondary: '#3a665b'
  on-secondary: '#ffffff'
  secondary-container: '#bcedde'
  on-secondary-container: '#406d61'
  tertiary: '#835409'
  on-tertiary: '#ffffff'
  tertiary-container: '#c68d42'
  on-tertiary-container: '#472b00'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#77f8dc'
  primary-fixed-dim: '#57dbc0'
  on-primary-fixed: '#00201a'
  on-primary-fixed-variant: '#005144'
  secondary-fixed: '#bcedde'
  secondary-fixed-dim: '#a1d0c2'
  on-secondary-fixed: '#00201a'
  on-secondary-fixed-variant: '#214e44'
  tertiary-fixed: '#ffddb7'
  tertiary-fixed-dim: '#fabb6a'
  on-tertiary-fixed: '#2a1700'
  on-tertiary-fixed-variant: '#653e00'
  background: '#f5fbf8'
  on-background: '#171d1b'
  surface-variant: '#dee4e1'
typography:
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  label-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1rem
  margin: 1.5rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2rem
---

# Design System

## Brand & Style
The design system embraces a modern, clean fidelity style, utilizing the **Inter** typeface for all levels to ensure high legibility and a contemporary digital aesthetic. The personality is professional, approachable, and technically precise, evoking trust and efficiency.

## Colors
The color palette is built around a fresh and stable foundation using semantic color derivation.
- **Primary (`#07ac93`)**: A vibrant, confident teal used for primary actions, active states, and key interactive elements.
- **Secondary (`#538074`)**: A muted, sophisticated slate-teal providing supporting contrast and secondary actions.
- **Tertiary (`#ffbf6e`)**: A warm, luminous amber accent used sparingly for highlights and notifications.
- **Neutral (`#727876`)**: A balanced, cool grey utilized for text, borders, and subtle background surfaces.

## Typography
Typography relies exclusively on **Inter** to maintain visual harmony and clean rendering across all platforms. The scale utilizes clear hierarchical steps from large display headings down to compact interface labels.

## Layout & Spacing
A standard fluid grid system is employed, relying on a consistent 8px/16px spacing rhythm (`space-sm` to `space-xl`). Margins scale gracefully from compact mobile viewports to expansive desktop layouts, ensuring comfortable breathing room around all content containers.

## Elevation & Depth
Elevation is achieved primarily through tonal surface layering and soft, low-contrast ambient shadows tinted with the neutral color base. This maintains a clean, modern interface without harsh visual interruptions.

## Shapes
The design system adopts a roundedness level of `2` (Rounded). UI elements feature a standard 0.5rem border radius, with larger containers scaling up to 1rem and 1.5rem (`rounded-lg`, `rounded-xl`), creating an inviting and approachable user experience.

## Components
Components leverage the rounded shape language and primary teal color palette (`#07ac93`) for interactive states. 
- **Buttons**: Feature solid primary fills with 0.5rem corner radii, transitioning to secondary slate tones on hover.
- **Input Fields**: Utilize clean neutral borders (`#727876`) that highlight with the primary teal upon focus.
- **Cards**: Employ soft tonal backgrounds and rounded corners to group related information clearly.