__LAZY_MODULE_ATTRS__ = {
    "collect_ci_metrics": ("ztb.utils.ci_utils", "collect_ci_metrics"),
    "notify_ci_results": ("ztb.utils.ci_utils", "notify_ci_results"),
    "DiscordNotifier": ("ztb.utils.notify", "DiscordNotifier"),
    "NotificationManager": (
        "ztb.utils.notify.notification_manager",
        "NotificationManager",
    ),
}

# CI trigger: no-op change to force GitHub Actions run


def __getattr__(name: str):
    if name in __LAZY_MODULE_ATTRS__:
        module_name, attr = __LAZY_MODULE_ATTRS__[name]
        mod = __import__(module_name, fromlist=[attr])
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(list(globals().keys()) + list(__LAZY_MODULE_ATTRS__.keys()))
