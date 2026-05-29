#!/bin/bash
# Pre-deployment verification script

echo "🔍 Checking deployment requirements..."
echo ""

# Check Python files
echo "✓ Checking main files:"
[ -f "streamlit_app.py" ] && echo "  ✅ streamlit_app.py" || echo "  ❌ streamlit_app.py NOT FOUND"
[ -f "requirements.txt" ] && echo "  ✅ requirements.txt" || echo "  ❌ requirements.txt NOT FOUND"
[ -f "README.md" ] && echo "  ✅ README.md" || echo "  ❌ README.md NOT FOUND"

# Check src folder
echo ""
echo "✓ Checking src/ folder:"
[ -d "src" ] && echo "  ✅ src/ exists" || echo "  ❌ src/ NOT FOUND"
[ -f "src/preprocessing.py" ] && echo "  ✅ src/preprocessing.py" || echo "  ❌ NOT FOUND"
[ -f "src/inference.py" ] && echo "  ✅ src/inference.py" || echo "  ❌ NOT FOUND"
[ -f "src/topic_modeling.py" ] && echo "  ✅ src/topic_modeling.py" || echo "  ❌ NOT FOUND"

# Check artifacts folder
echo ""
echo "✓ Checking artifacts/ folder (CRITICAL):"
[ -d "artifacts" ] && echo "  ✅ artifacts/ exists" || echo "  ❌ artifacts/ NOT FOUND"
[ -f "artifacts/model.pkl" ] && echo "  ✅ artifacts/model.pkl" || echo "  ⚠️  artifacts/model.pkl NOT FOUND (NEEDED!)"
[ -f "artifacts/tfidf_vectorizer.pkl" ] && echo "  ✅ artifacts/tfidf_vectorizer.pkl" || echo "  ⚠️  artifacts/tfidf_vectorizer.pkl NOT FOUND (NEEDED!)"
[ -f "artifacts/slang_dict.json" ] && echo "  ✅ artifacts/slang_dict.json" || echo "  ⚠️  artifacts/slang_dict.json NOT FOUND"
[ -f "artifacts/custom_stopwords.json" ] && echo "  ✅ artifacts/custom_stopwords.json" || echo "  ⚠️  artifacts/custom_stopwords.json NOT FOUND"
[ -f "artifacts/protected_words.json" ] && echo "  ✅ artifacts/protected_words.json" || echo "  ⚠️  artifacts/protected_words.json NOT FOUND"
[ -f "artifacts/special_cases.json" ] && echo "  ✅ artifacts/special_cases.json" || echo "  ⚠️  artifacts/special_cases.json NOT FOUND"

# Check config files
echo ""
echo "✓ Checking deployment configs:"
[ -f ".gitignore" ] && echo "  ✅ .gitignore" || echo "  ❌ .gitignore NOT FOUND"
[ -f ".streamlit/config.toml" ] && echo "  ✅ .streamlit/config.toml" || echo "  ❌ .streamlit/config.toml NOT FOUND"
[ -f "DEPLOYMENT_GUIDE.md" ] && echo "  ✅ DEPLOYMENT_GUIDE.md" || echo "  ❌ DEPLOYMENT_GUIDE.md NOT FOUND"

echo ""
echo "✅ Pre-deployment check complete!"
echo ""
echo "⚠️  WARNING: Make sure model.pkl and tfidf_vectorizer.pkl exist in artifacts/"
echo "            These are REQUIRED for the app to run!"
