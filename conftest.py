"""Root conftest.py: pytest imports this and inserts the project root onto sys.path,
so `import digital_twin` / `import ai_module` / `import dashboard` resolve from any
test file without a src-layout or editable install.
"""
