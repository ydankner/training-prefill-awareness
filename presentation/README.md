# Talk: Training Prefill-Awareness in Small Models (~30 min)

Open `index.html` in a browser (works offline, straight from disk).
`slides.pdf` is a static export with every reveal already shown.

| Key | Action |
| --- | --- |
| → / space / click | next reveal or slide |
| ← / click left quarter | back |
| ↑ / ↓ | previous / next slide, skipping reveals |
| f | fullscreen |
| Home / End | first / last slide |

`#12` in the URL jumps to slide 12.

## Rough timing

| Slides | Part | Min |
| --- | --- | --- |
| 1–4 | Motivation: definition, Qwen 3.6 banana, train it then study it | 3 |
| 5–18 | Part I · Qwen3-32B, greedy, 1 plant: training scheme, spot the plant, trainability, self-knowledge, generalization, mechanism, surprisal | 12 |
| 19–27 | Part II · Qwen3.8-27B, sampled, 1–4 plants: wording sensitivity, marker, two training examples, results, haiku chat | 8 |
| 28–30 | Part III · Removed reasoning | 3 |
| 31–34 | Part IV · CIAware-Bench, if there's time | 3 |
| 35–36 | Takeaways, thank you | 1 |
| 37 | Backup: Qwen3.8 softmax heads | — |

## Regenerating figures

The heatmaps are drawn from the paper's data (`paper/figures/*.dat`), packed into `data.js`:

```
py -3 presentation/build_data.py
```

To re-export the PDF, serve the folder (`py -3 -m http.server 8765 --directory presentation`)
and print the page from a browser with margins set to none, or run headless Edge with `--print-to-pdf`.
