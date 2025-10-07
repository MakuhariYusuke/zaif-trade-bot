import re

# Read the file
with open('ztb/utils/talib_wrapper.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace floating type parameters
content = re.sub(r'np\.dtype\[np\.floating\]', 'np.dtype[np.floating[Any]]', content)
content = re.sub(r'np\.ndarray\[Any, np\.dtype\[np\.floating\]\]', 'np.ndarray[Any, np.dtype[np.floating[Any]]]', content)

# Write back
with open('ztb/utils/talib_wrapper.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed floating type parameters in talib_wrapper.py')