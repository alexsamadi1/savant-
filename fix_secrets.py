import re
import os

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Add get_secret import if file uses st.secrets
    if 'st.secrets' not in content:
        return
    
    # Replace st.secrets["KEY"] with get_secret("KEY")
    content = re.sub(r'st\.secrets\["([^"]+)"\]', r'get_secret("\1")', content)
    
    # Add import at top if not already there
    if 'from tools.s3_utils import get_secret' not in content and 'from tools.secrets' not in content:
        if filepath == 'app.py':
            content = content.replace(
                'from tools.s3_utils import',
                'from tools.s3_utils import get_secret,\nfrom tools.s3_utils import'
            )
        elif 'import streamlit as st' in content:
            content = content.replace(
                'import streamlit as st',
                'import streamlit as st\ntry:\n    from tools.s3_utils import get_secret\nexcept ImportError:\n    pass'
            )
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed: {filepath}")

files = [
    'app.py',
    'tools/filename_generator.py',
    'tools/analytics_dashboard.py',
    'tools/log_utils.py',
    'tools/vectorstore_builder.py'
]

for f in files:
    if os.path.exists(f):
        fix_file(f)

print("Done")
