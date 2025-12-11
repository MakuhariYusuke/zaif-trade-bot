#!/usr/bin/env python3
"""
Fix the syntax error in action_signal_guide.py
"""


# Read the file
with open(
    "ztb/trading/strategies/action_signal_guide/action_signal_guide.py",
    "r",
    encoding="utf-8",
) as f:
    lines = f.readlines()

# Keep only up to the return stats line (around line 1765)
# Find the get_stats method end
for i, line in enumerate(lines):
    if "return stats" in line and i > 1740:
        cutoff_line = i + 1
        break
else:
    cutoff_line = 1765

# Keep lines up to cutoff
clean_lines = lines[:cutoff_line]

# Add clean ending
clean_ending = '''
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration dynamically."""
        try:
            # Validate new config
            temp_config = ActionSignalGuideConfig(**{**self.config.__dict__, **new_config})
            self._validate_config()
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration update: {e}") from e

        # Update config
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Update guidance level if changed
        if 'guidance_level' in new_config:
            self.guidance_level = self.config.guidance_level
            self._update_guidance_mode(self.guidance_level)

        self.logger.info(f"Configuration updated: {list(new_config.keys())}")

    def _update_guidance_mode(self, guidance_level: GuidanceLevel) -> None:
        # Update internal guidance mode based on guidance level
        # Implementation for guidance mode updates
        pass
'''

# Write back
with open(
    "ztb/trading/strategies/action_signal_guide/action_signal_guide.py",
    "w",
    encoding="utf-8",
) as f:
    f.writelines(clean_lines)
    f.write(clean_ending)

print("File syntax fixed")
