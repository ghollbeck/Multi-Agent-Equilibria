#!/bin/bash

# Remove .DS_Store files from git tracking but keep them locally
git rm --cached .DS_Store
git rm --cached **/.DS_Store

# Commit these changes
git commit -m "Remove .DS_Store files from git tracking"

# Push the changes
git push origin main 