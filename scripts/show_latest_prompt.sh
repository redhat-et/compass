#!/bin/bash
#
# Show the most recent prompt for easy copy-paste

PROMPTS_DIR="logs/prompts"

if [ ! -d "$PROMPTS_DIR" ]; then
    echo "No prompts directory found. Make a request first."
    exit 1
fi

# Find the most recent prompt file
LATEST=$(ls -t "$PROMPTS_DIR"/intent_extraction_*.txt 2>/dev/null | head -1)

if [ -z "$LATEST" ]; then
    echo "No prompt files found. Make a request first."
    exit 1
fi

echo "📋 Latest prompt file: $LATEST"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cat "$LATEST"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 To copy to clipboard:"
echo "   macOS:  cat '$LATEST' | grep -v '^=' | pbcopy"
echo "   Linux:  cat '$LATEST' | grep -v '^=' | xclip -selection clipboard"
echo ""
echo "📁 All prompts: ls -lht $PROMPTS_DIR/"
