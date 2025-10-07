with open('ztb/utils/run_manifest.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
lines[13] = 'from typing import Dict, Any, Optional, List, cast\n'
with open('ztb/utils/run_manifest.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('Fixed')