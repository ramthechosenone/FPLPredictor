#!/bin/bash
# Sync BLOG_JOURNAL.md from FPL predictor to personal-site
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SITE_DIR="$(dirname "$SCRIPT_DIR")/Personal Website"

if [ ! -d "$SITE_DIR" ]; then
    echo "Error: Personal Website directory not found at $SITE_DIR"
    exit 1
fi

cp "$SCRIPT_DIR/BLOG_JOURNAL.md" "$SITE_DIR/app/fpl/blog/BLOG_JOURNAL.md"
echo "Synced BLOG_JOURNAL.md to personal-site"
