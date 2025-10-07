import re

# Read the file
with open('ztb/utils/talib_wrapper.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace ndarray type parameters - be more specific to avoid over-replacement
content = re.sub(r'np\.ndarray\[Any, np\.dtype\[Any\]\]', 'np.ndarray[Any, np.dtype[np.floating[Any]]]', content)

# Write back
with open('ztb/utils/talib_wrapper.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed ndarray type parameters in talib_wrapper.py')