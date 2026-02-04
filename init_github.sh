#!/bin/bash
# GitHub Repository Initialization Script
# Usage: ./init_github.sh [github_token]

set -e

REPO_NAME="music-skills"
REPO_DIR="/Users/mindstorm/.openclaw/workspace/music-skills"
GITHUB_USER="baikai-li"
GITHUB_TOKEN="${1:-$GH_TOKEN}"

echo "GitHub Repository Initialization"
echo "================================="
echo "Repository: $GITHUB_USER/$REPO_NAME"

if [ -z "$GITHUB_TOKEN" ]; then
    echo ""
    echo "Error: GitHub token not provided."
    echo "Please provide your GitHub token as argument or set GH_TOKEN environment variable."
    echo ""
    echo "To create a token:"
    echo "  1. Go to https://github.com/settings/tokens"
    echo "  2. Generate a new token with 'repo' scope"
    echo "  3. Run: ./init_github.sh YOUR_TOKEN"
    exit 1
fi

echo ""
echo "Step 1: Creating GitHub repository..."
# Create repository using GitHub API
curl -X POST -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    https://api.github.com/user/repos \
    -d "{\"name\":\"$REPO_NAME\",\"description\":\"Python CLI tool for music voice conversion and song covering\",\"private\":false,\"auto_init\":true}"

echo ""
echo "Step 2: Initializing local git repository..."
cd "$REPO_DIR"

# Initialize git if not already done
if [ ! -d .git ]; then
    git init
    git add -A
    git commit -m "Initial commit: Music Skills CLI v0.1.0"
fi

# Add remote
if ! git remote get-url origin &> /dev/null; then
    git remote add origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"
fi

# Update remote URL with credentials
git remote set-url origin "https://$GITHUB_TOKEN@github.com/$GITHUB_USER/$REPO_NAME.git"

echo ""
echo "Step 3: Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "====================================="
echo "GitHub repository initialized successfully!"
echo ""
echo "Repository URL: https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
echo "Next steps:"
echo "  1. Install dependencies: ./setup.sh"
echo "  2. Run tests: pytest"
echo "  3. Start using: music-skills --help"
