# Fonts

Self-hosted on purpose: client data is on the screen, and a webfont request is a
request to somebody else's server carrying this page's URL. Nothing here calls
home (`docs/ui-product.md` §3).

| File | Family | Licence |
|---|---|---|
| `Satoshi-Medium/Bold/Black.woff2` | Satoshi — UI | Fontshare Free Font Licence, `Satoshi-LICENSE.txt` |
| `InstrumentSerif-Regular.woff2` | Instrument Serif — run and report titles | SIL Open Font Licence 1.1 |
| `JetBrainsMono-Regular.woff2` | JetBrains Mono — paths, ids, numbers | SIL Open Font Licence 1.1 |

The two OFL faces are Google Fonts' `latin` subsets, and JetBrains Mono is the
variable file, which is why one file covers both weights the UI asks for.

To refresh a face, download it again and keep the name — `tokens.css` names these
files, and the stacks fall back to system faces if one is missing, so a bad
download degrades to plain rather than to broken.
