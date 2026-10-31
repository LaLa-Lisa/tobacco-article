# Local dataset inspection, 2026-10-31

Read `original_article.pdf` §2.1 and Table 1 for dataset facts only.
The eight campaigns, patch sizes, counts, and three semantic class values match
the provided V2 data. This is distinct from the repository's 110-image detector
subset. No weed-detection method from the article is implemented.

| Campaign | Pairs | New split |
| --- | ---: | --- |
| 1 | 864 | train |
| 2 | 936 | train |
| 3–5 | 120 each | train |
| 6 | 120 | validation |
| 7–8 | 120 each | test |

All 2,520 patches have matching 480×352 masks, containing only 0/1/2. All have an
approximate appearance-based match to a source frame (quarter-resolution
normalized template error below 0.02). This is measured association, not a claim
of bit-identical crops or uniquely resolved provenance for duplicate frames.
The original source folders contain 210 RGB images. V1 has 2,520 PNG files whose
file hashes match V2 input patches; versions must not be concatenated.

Campaign 5 contains 12 exact decoded-image duplicate pairs: patch numbers 13–24
are repeated as 49–60. The paired semantic mask files have different hashes.
The new export excludes the later duplicate copies, leaving at most **2,148
unique training images**, before unreviewed images are excluded. Human review of
the retained copies must resolve any conflicting semantic guides against RGB.
There are no exact decoded-image duplicates across the proposed split, and the
coarse source-thumbnail screen produced no cross-split overlap candidates. That
screen does not exclude partial scene overlap; source-scene review is still
required.

Full-resolution mask encodings vary. Campaign 1 includes 0/3/4/5/127/255 across
its masks; campaign 2 has 0/127/255; campaigns 3–8 have 0/1/2 in PixelLabelData.
These masks are inventoried, not automatically coerced into new labels.

Detailed local evidence is in ignored
`experiments/reproductions/sam3-audit-local-20261031-v2/audit.json`.
That audit is **unapproved**. No instance annotations, training results, or
accuracy claims have been fabricated. `transfer_manifest.json` in the same
folder lists local transfer files with byte counts and SHA-256 hashes; these
are hashes computed from this checkout, not publisher-provided checksums.
