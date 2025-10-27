#!/bin/bash
# Build CircleClust documentation and open in browser

set -e

echo "📚 Building CircleClust documentation..."

# Build the documentation
python -m sphinx -b html . _build/html

echo ""
echo "✅ Documentation built successfully!"
echo ""
echo "📍 To view the documentation:"
echo "   1. Open a browser and go to: http://localhost:8000"
echo "   2. Or run: open _build/html/index.html"
echo ""
echo "   To start a local server, run:"
echo "   python -m http.server --directory _build/html 8000"
echo ""

# Try to open in default browser (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🌐 Opening in browser..."
    open _build/html/index.html
fi

