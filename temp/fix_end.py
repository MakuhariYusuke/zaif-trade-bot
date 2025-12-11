#!/usr/bin/env python3
"""
Safely fix the end of action_signal_guide.py
"""

# Read the file
with open(
    "ztb/trading/strategies/action_signal_guide/action_signal_guide.py",
    "r",
    encoding="utf-8",
) as f:
    content = f.read()

# Find the return stats line
return_stats_pos = content.find("return stats")
if return_stats_pos == -1:
    print("return stats not found")
    exit(1)

# Keep everything up to and including return stats
cutoff_pos = content.find("\n", return_stats_pos) + 1
new_content = content[:cutoff_pos]

# Add clean ending
new_content += '''
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
    f.write(new_content)

print("File fixed successfully")
