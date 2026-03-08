"""Re-export from canonical ztb.ops.alerts.alert_notifier."""

from ztb.ops.alerts.alert_notifier import (  # noqa: F401
    AlertRecord,
    load_alerts,
    main,
    send_webhook,
)
