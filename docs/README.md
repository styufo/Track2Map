# Track2Map project page

This folder contains the static GitHub Pages project page for Track2Map.

## Local preview

From the repository root:

```bash
python3 -m http.server 8000 --directory docs
```

Then open:

```text
http://localhost:8000
```

## GitHub Pages deployment

In the GitHub repository settings:

1. Go to `Settings` → `Pages`.
2. Set `Source` to `Deploy from a branch`.
3. Select branch `main` and folder `/docs`.
4. Save.

The expected project page URL is:

```text
https://styufo.github.io/Track2Map/
```

## Assets to update before final release

- Replace the placeholder PDF button in `index.html` with the arXiv/PDF link once available.
- Replace the BibTeX block in `index.html` once Springer/arXiv metadata is available.
- Optionally add an arXiv / DOI / dataset link button when available.
