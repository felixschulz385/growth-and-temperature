# GNT Analysis Results Website

This directory contains a Jekyll-based website that displays analysis results.

## Local Development

1. Install dependencies:
```bash
bundle install
pip install pandas
```

2. Generate table partials:
```bash
python generate_tables.py
```

3. Build and serve locally:
```bash
bundle exec jekyll serve
```

Visit http://localhost:4000

## Deployment

The site automatically deploys via GitHub Actions when you push to main/master branch.

The workflow:
1. Runs `generate_tables.py` to create HTML table partials in `_includes/`
2. Builds the Jekyll site
3. Deploys to GitHub Pages

## Adding New Tables

1. Edit `generate_tables.py`
2. Add a new function like `generate_my_table()`
3. Save output to `_includes/table_my_name.html`
4. Include in `index.md` with `{% include table_my_name.html %}`

## File Structure

```
output/webpage/
├── _config.yml           # Jekyll configuration
├── _includes/            # Generated HTML partials (git-ignored)
│   └── table_main.html
├── Gemfile              # Ruby dependencies
├── generate_tables.py   # Python script to generate tables
├── index.md            # Main page
└── README.md           # This file
```
