with open(
    "c:\\Users\\Admin\\dev\\zaif-trade-bot\\ztb\\trading\\environment\\heavy_env\\mixins\\initialization.py",
    "r",
) as f:
    content = f.read()
old_line = '        correlation_reduction = getattr(self.config, "correlation_reduction", True)'
new_line = (
    '    correlation_reduction = getattr(self.config, "correlation_reduction", True)'
)
content = content.replace(old_line, new_line)
with open(
    "c:\\Users\\Admin\\dev\\zaif-trade-bot\\ztb\\trading\\environment\\heavy_env\\mixins\\initialization.py",
    "w",
) as f:
    f.write(content)
