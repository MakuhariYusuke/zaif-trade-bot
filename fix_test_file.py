#!/usr/bin/env python3
"""Fix test_env_reward_settings.py by removing duplicate methods."""

with open('test_env_reward_settings.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Keep only lines up to line 255, then add the closing
new_content = ''.join(lines[:255])
new_content += '\n\nif __name__ == \'__main__\':\n    unittest.main()\n'

with open('test_env_reward_settings.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("File fixed. Lines trimmed to 255 + footer.")
