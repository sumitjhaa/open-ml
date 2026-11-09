# Epocher

An interactive machine learning course: notes, worked figures, interactive sections, notebooks, slides and cheat sheets, in one site.

Written and organised by Sumit Jha. Source material is listed under [References](#references).

## Layout

```
open-ml/
├── README.md
├── notes/                     Course notes in Markdown. The source of truth.
│   └── images/                Figures and slide captures, one folder per note.
├── site/                      The Next.js website.
│   ├── app/                   Pages and routes.
│   ├── components/            Shared interface pieces.
│   ├── explorers/             Interactive sections, by topic.
│   ├── lib/                   Catalogue, guides, step-ups, themes and algorithms.
│   ├── public/                Files served by the site. Generated, do not edit.
│   └── scripts/               Sync, figures, PDFs, import and checks.
└── sources/                   Original course material. Never edited.
    ├── course-2026/           Lectures, tutorials and course documents.
    ├── course-2025/           Downloaded course repositories. Not in git, because of its size.
    ├── course-deep-learning/  Downloaded, not in git.
    └── course-probabilistic/  Downloaded, not in git.
```

## Running the site

```
cd site
yarn install
yarn dev
```

The site needs nothing outside this repository. Only the one-minute videos link elsewhere, and they open in a new tab.

## After changing content

Run these from `site/`.

| Changed | Run |
|---|---|
| Anything in `notes/` | `node scripts/sync.mjs` |
| A figure in `scripts/make-figures.mjs` | `node scripts/make-figures.mjs`, then sync |
| Material in `sources/course-2025/` | `python3 -I scripts/import-course.py` |
| PDFs of the notes, with the site running | `node scripts/make-pdf.mjs http://localhost:3000` |
| Before committing | `node scripts/check.mjs` |

## References

- ES 335 / ES 678 Machine Learning, Semester I 2026, IIT Gandhinagar. Prof. Udit Bhatia, with Prof. Krishna Prasad Miyapuram.
- ES 335 Machine Learning, Semester I 2025–26, IIT Gandhinagar. Prof. Nipun Batra.
- ES 667 Deep Learning, August 2026, IIT Gandhinagar. Prof. Nipun Batra.
- ES 661 Probabilistic Machine Learning, IIT Gandhinagar. Prof. Nipun Batra.
