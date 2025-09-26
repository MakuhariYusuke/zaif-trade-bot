""""""

Discord notification system for trading bot alerts.Discord notification system for trading bot alerts.

""""""



import jsonimport json

import loggingimport logging

import requestsimport requests

from typing import Dict, Any, Optionalfrom typing import Dict, Any, Optional



logger = logging.getLogger(__name__)logger = logging.getLogger(__name__)



class DiscordNotifier:class DiscordNotifier:

    """Discord webhook notifier for trading bot events"""    """Discord webhook notifier for trading bot events"""



    def __init__(self, webhook_url: Optional[str] = None):    def __init__(self, webhook_url: Optional[str] = None):

        self.webhook_url = webhook_url or self._get_webhook_url()        self.webhook_url = webhook_url or self._get_webhook_url()



    def _get_webhook_url(self) -> Optional[str]:    def _get_webhook_url(self) -> Optional[str]:

        """Get webhook URL from environment or config"""        """Get webhook URL from environment or config"""

        import os        import os

        return os.getenv('DISCORD_WEBHOOK_URL')        return os.getenv('DISCORD_WEBHOOK_URL')



    def notify(self, message: str, level: str = "info", data: Optional[Dict[str, Any]] = None):    def notify(self, message: str, level: str = "info", data: Optional[Dict[str, Any]] = None):

        """        """

        Send notification to Discord.        Send notification to Discord.



        Args:        Args:

            message: Main message text            message: Main message text

            level: Notification level ('info', 'warning', 'error', 'success')            level: Notification level ('info', 'warning', 'error', 'success')

            data: Additional data to include as JSON            data: Additional data to include as JSON

        """        """

        if not self.webhook_url:        if not self.webhook_url:

            logger.warning("Discord webhook URL not configured")            logger.warning("Discord webhook URL not configured")

            return            return



        # Create embed based on level        # Create embed based on level

        embed = self._create_embed(message, level, data)        embed = self._create_embed(message, level, data)



        payload = {        payload = {

            "embeds": [embed]            "embeds": [embed]

        }        }



        try:        try:

            response = requests.post(            response = requests.post(

                self.webhook_url,                self.webhook_url,

                json=payload,                json=payload,

                headers={'Content-Type': 'application/json'},                headers={'Content-Type': 'application/json'},

                timeout=10                timeout=10

            )            )

            response.raise_for_status()            response.raise_for_status()

            logger.info(f"Discord notification sent: {level} - {message[:50]}...")            logger.info(f"Discord notification sent: {level} - {message[:50]}...")

        except Exception as e:        except Exception as e:

            logger.error(f"Failed to send Discord notification: {e}")            logger.error(f"Failed to send Discord notification: {e}")



    def _create_embed(self, message: str, level: str, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:    def _create_embed(self, message: str, level: str, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:

        """Create Discord embed for the notification"""        """Create Discord embed for the notification"""

        colors = {        colors = {

            'info': 0x3498db,      # Blue            'info': 0x3498db,      # Blue

            'warning': 0xf39c12,   # Orange            'warning': 0xf39c12,   # Orange

            'error': 0xe74c3c,     # Red            'error': 0xe74c3c,     # Red

            'success': 0x2ecc71    # Green            'success': 0x2ecc71    # Green

        }        }



        embed = {        embed = {

            "title": f"Trading Bot {level.title()}",            "title": f"Trading Bot {level.title()}",

            "description": message,            "description": message,

            "color": colors.get(level, colors['info']),            "color": colors.get(level, colors['info']),

            "timestamp": None  # Will be set by Discord            "timestamp": None  # Will be set by Discord

        }        }



        if data:        if data:

            # Add fields for key metrics            # Add fields for key metrics

            fields = []            fields = []

            for key, value in data.items():            for key, value in data.items():

                if isinstance(value, (int, float)):                if isinstance(value, (int, float)):

                    fields.append({                    fields.append({

                        "name": key.replace('_', ' ').title(),                        "name": key.replace('_', ' ').title(),

                        "value": f"{value:.4f}" if isinstance(value, float) else str(value),                        "value": f"{value:.4f}" if isinstance(value, float) else str(value),

                        "inline": True                        "inline": True

                    })                    })

                elif isinstance(value, str) and len(value) < 100:                elif isinstance(value, str) and len(value) < 100:

                    fields.append({                    fields.append({

                        "name": key.replace('_', ' ').title(),                        "name": key.replace('_', ' ').title(),

                        "value": value,                        "value": value,

                        "inline": True                        "inline": True

                    })                    })



            if fields:            if fields:

                embed["fields"] = fields[:25]  # Discord limit                embed["fields"] = fields[:25]  # Discord limit



        return embed        return embed



def get_notifier() -> DiscordNotifier:def get_notifier() -> DiscordNotifier:

    """Get configured Discord notifier instance"""    """Get configured Discord notifier instance"""

    return DiscordNotifier()    return DiscordNotifier()

class DiscordNotifier:import queue

    """Discord webhook notifier for trading bot events"""import threading

import time

    def __init__(self, webhook_url: Optional[str] = None):

        self.webhook_url = webhook_url or self._get_webhook_url()# ローカルモジュールのインポート

sys.path.append(str(Path(__file__).parent.parent))

    def _get_webhook_url(self) -> Optional[str]:from pathlib import Path

        """Get webhook URL from environment or config"""

        import os

        return os.getenv('DISCORD_WEBHOOK_URL')class AsyncNotifier:

    """非同期Discord通知クラス（キュー + 集約送信）"""

    def notify(self, message: str, level: str = "info", data: Optional[Dict[str, Any]] = None):    

        """    def __init__(self, notifier, flush_sec=300):

        Send notification to Discord.        self.n = notifier

        self.q = queue.Queue()

        Args:        self.flush_sec = flush_sec

            message: Main message text        self.buf = []

            level: Notification level ('info', 'warning', 'error', 'success')        threading.Thread(target=self._loop, daemon=True).start()

            data: Additional data to include as JSON

        """    def enqueue(self, msg):

        if not self.webhook_url:        """通常メッセージをキューに追加"""

            logger.warning("Discord webhook URL not configured")        self.q.put(("info", msg))

            return    

    def error(self, msg):

        # Create embed based on level        """エラーメッセージを即時送信"""

        embed = self._create_embed(message, level, data)        self.q.put(("error", msg))

    

        payload = {    def flush(self):

            "embeds": [embed]        """バッファを即時送信"""

        }        if self.buf:

            body = "\n".join(self.buf)

        try:            self.buf.clear()

            response = requests.post(            self.n.send_custom_notification("📣 Training Update", body, color=0x00AAFF)

                self.webhook_url,

                json=payload,    def _loop(self):

                headers={'Content-Type': 'application/json'},        last = time.time()

                timeout=10        while True:

            )            try:

            response.raise_for_status()                kind, msg = self.q.get(timeout=1)

            logger.info(f"Discord notification sent: {level} - {message[:50]}...")                if kind == "error":

        except Exception as e:                    self.n.send_custom_notification("🚨 Error", msg, color=0xFF0000)

            logger.error(f"Failed to send Discord notification: {e}")                else:

                    self.buf.append(f"- {msg}")

    def _create_embed(self, message: str, level: str, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:                # 定期的に集約送信

        """Create Discord embed for the notification"""                if time.time() - last >= self.flush_sec and self.buf:

        colors = {                    body = "\n".join(self.buf)

            'info': 0x3498db,      # Blue                    self.buf.clear()

            'warning': 0xf39c12,   # Orange                    last = time.time()

            'error': 0xe74c3c,     # Red                    self.n.send_custom_notification("📣 Training Update", body, color=0x00AAFF)

            'success': 0x2ecc71    # Green            except queue.Empty:

        }                # タイムアウト時も集約送信チェック

                if time.time() - last >= self.flush_sec and self.buf:

        embed = {                    body = "\n".join(self.buf)

            "title": f"Trading Bot {level.title()}",                    self.buf.clear()

            "description": message,                    last = time.time()

            "color": colors.get(level, colors['info']),                    self.n.send_custom_notification("📣 Training Update", body, color=0x00AAFF)

            "timestamp": None  # Will be set by Discord

        }

class DiscordNotifier:

        if data:    """Discord通知クラス"""

            # Add fields for key metrics

            fields = []    def __init__(self, webhook_url: Optional[str] = None, test_mode: bool = False):

            for key, value in data.items():        """初期化"""

                if isinstance(value, (int, float)):        self.test_mode = test_mode

                    fields.append({        # 環境変数が設定されていない場合、.envファイルから読み込み

                        "name": key.replace('_', ' ').title(),        if not os.getenv('DISCORD_WEBHOOK'):

                        "value": f"{value:.4f}" if isinstance(value, float) else str(value),            env_path = Path(__file__).parent.parent.parent.parent / '.env'

                        "inline": True            # .envファイルを現在のディレクトリから親ディレクトリに向かって探索

                    })            env_path = None

                elif isinstance(value, str) and len(value) < 100:            search_path = Path(__file__).parent

                    fields.append({            for _ in range(5):  # 最大5階層まで探索

                        "name": key.replace('_', ' ').title(),                candidate = search_path / '.env'

                        "value": value,                if candidate.exists():

                        "inline": True                    env_path = candidate

                    })                    break

                search_path = search_path.parent

            if fields:            if env_path:

                embed["fields"] = fields[:25]  # Discord limit                with open(env_path, 'r') as f:

                    for line in f:

        return embed                        line = line.strip()

                        if line and not line.startswith('#') and '=' in line:

def get_notifier() -> DiscordNotifier:                            key, value = line.split('=', 1)

    """Get configured Discord notifier instance"""                            if key == 'DISCORD_WEBHOOK':

    return DiscordNotifier()                                os.environ[key] = value
                                break
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK')
        self.session_start_time = None
        self.session_id = None

        if not self.webhook_url:
            logging.warning("DISCORD_WEBHOOK not found in environment variables")

    def _send_notification(self, content: str, color: int = 0x0099ff) -> bool:
        """Discordに通知を送信（指数バックオフ付きリトライ）"""
        if not self.webhook_url:
            logging.warning("No webhook URL configured, skipping notification")
            return False

        max_retries = 3
        for attempt in range(max_retries):
            try:
                payload = {
                    "content": content,
                    "embeds": [{"color": color}],
                    "username": "Trading RL Bot",
                    "avatar_url": "https://i.imgur.com/4M34hi2.png"  # Trading bot avatar
                }

                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )

                if response.status_code in (200, 204):
                    logging.info("Discord notification sent successfully")
                    return True
                else:
                    logging.error(f"Failed to send Discord notification: {response.status_code}, {response.text}")
                    return False

            except Exception as e:
                wait_time = 2 ** attempt  # 指数バックオフ
                logging.warning(f"Discord notification attempt {attempt + 1} failed: {e}, retrying in {wait_time}s")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logging.error(f"Discord notification failed after {max_retries} attempts")
                    return False

        return False  # すべてのリトライが失敗した場合


    def start_session(self, session_type: str = "training", config_name: str = "default", prefix: str = "") -> str:
        """セッション開始通知"""
        self.session_start_time = datetime.now()
        self.session_id = self.session_start_time.strftime('%Y%m%d-%H%M%S')

        icon = "🧪" if self.test_mode else "🚀"
        title = f"{icon} **{session_type.title()} Session Started**"
        if prefix:
            title = f"{prefix} {title}"

        content = f"{title}\n\n**Session ID:** `{self.session_id}`\n**Config:** `{config_name}`\n**Start Time:** {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')} JST\n**Session Type:** {session_type.title()}"

        color = 0x00ff00  # Green for start
        self._send_notification(content, color)
        return self.session_id

    def end_session(self, results: Dict[str, Any], session_type: str = "training") -> None:
        """セッション終了通知"""
        if not self.session_start_time:
            logging.warning("Session not started, cannot send end notification")
            return

        end_time = datetime.now()
        duration = end_time - self.session_start_time

        # 結果データの解析
        reward_stats = results.get('reward_stats', {})
        pnl_stats = results.get('pnl_stats', {})
        trading_stats = results.get('trading_stats', {})

        # 勝率の計算
        total_trades = trading_stats.get('total_trades', 0)
        winning_trades = trading_stats.get('winning_trades', 0)
        win_rate_percent = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = trading_stats.get('profit_factor', 0)

        # 設定ファイルから閾値読み込み
        config_path = Path("../rl_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            win_rate_threshold = config.get('notifications', {}).get('win_rate_threshold_percent', 50.0)
        else:
            win_rate_threshold = 50.0

        # 閾値下回りチェック
        should_notify = win_rate_percent < win_rate_threshold

        # 損益単位の決定（BUY主体=BTC, SELL主体=JPY）
        buy_ratio = trading_stats.get('buy_ratio', 0)
        sell_ratio = trading_stats.get('sell_ratio', 0)
        pnl_unit = "BTC" if buy_ratio > sell_ratio else "JPY"
        total_pnl = pnl_stats.get('mean_total_pnl', 0)

        content = f"{'🧪' if self.test_mode else '✅'} **{session_type.title()} Session Completed**\n\n**Session ID:** `{self.session_id}`\n**Duration:** `{str(duration).split('.')[0]}`\n**End Time:** {end_time.strftime('%Y-%m-%d %H:%M:%S')} JST\n\n"

        # 数値系をコードブロックにまとめる
        content += "```\n"
        content += f"Win Rate: {win_rate_percent:.1f}% {'(参考)' if win_rate_percent < win_rate_threshold else ''}\n"
        content += f"Profit Factor: {profit_factor:.2f}\n"
        content += f"Total Trades: {total_trades}\n"
        content += f"Total PnL: {total_pnl:,.2f} {pnl_unit}\n"
        content += f"Max Drawdown: {pnl_stats.get('max_drawdown', 0):.4f}\n"
        content += f"Avg Hold Time: {trading_stats.get('mean_trades_per_episode', 0):.1f} trades/episode\n"
        content += f"Total Fees: {total_trades * 0.001:,.4f} BTC\n"
        content += f"Max Position Size: 1.0\n"
        content += f"Risk Reduction Triggers: 0\n"
        content += "```"

        # 色設定：勝率が閾値下回りなら黄、それ以外は緑
        color = 0xffff00 if win_rate_percent < win_rate_threshold else 0x00ff00

        # 勝率が閾値を下回った場合のみ通知
        if should_notify:
            self._send_notification(content, color)
        else:
            logging.info(f"Win rate {win_rate_percent:.1f}% is above threshold {win_rate_threshold}%, skipping notification")

    def send_error_notification(self, error_message: str, error_details: Optional[str] = None) -> None:
        """致命的エラー時の即時通知"""
        error_time = datetime.now()

        icon = "🧪" if self.test_mode else "🚨"
        title = f"{icon} **Critical Error Alert**"

        content = f"{title}\n\n**Session ID:** `{self.session_id or 'Unknown'}`\n**Error:** {error_message}\n**Error Time:** {error_time.strftime('%Y-%m-%d %H:%M:%S')} JST"
        if error_details:
            content += f"\n\n**Error Details:**\n```\n{error_details[:1000]}\n```"

        self._send_notification(content, 0xff0000)  # Red

    def send_custom_notification(self, title: str, message: str, color: int = 0x0099ff,
                               fields: Optional[List[Dict[str, Union[str, bool]]]] = None) -> None:
        """カスタム通知"""
        content = f"**{title}**\n\n{message}"
        if fields:
            for field in fields:
                name = field.get('name', '')
                value = field.get('value', '')
                content += f"\n\n**{name}:** {value}"

        self._send_notification(content, color)

    def send_experiment_results(self, result: Any) -> None:
        """実験結果を通知"""
        from ztb.experiments.base import ExperimentResult
        
        if not isinstance(result, ExperimentResult):
            logging.warning("Invalid result type for experiment notification")
            return

        icon = "🧪" if self.test_mode else "🔬"
        title = f"{icon} **Experiment Results: {result.experiment_name}**"

        status_emoji = {
            "success": "✅",
            "failed": "❌", 
            "partial": "⚠️"
        }.get(result.status, "❓")

        content = f"{title}\n\n**Status:** {status_emoji} {result.status.title()}\n**Timestamp:** {result.timestamp}"

        if result.execution_time_seconds:
            content += f"\n**Execution Time:** {result.execution_time_seconds:.1f}s"

        if result.error_message:
            content += f"\n\n**Error:** {result.error_message[:500]}"

        # メトリクス表示
        if result.metrics:
            content += "\n\n**Metrics:**\n```"
            for key, value in result.metrics.items():
                if isinstance(value, float):
                    content += f"\n{key}: {value:.4f}"
                else:
                    content += f"\n{key}: {value}"
            content += "\n```"

        # アーティファクト表示
        if result.artifacts:
            content += "\n\n**Artifacts:**"
            for name, path in result.artifacts.items():
                content += f"\n• {name}: `{path}`"

        color = {
            "success": 0x00ff00,
            "failed": 0xff0000,
            "partial": 0xffff00
        }.get(result.status, 0x0099ff)

        self._send_notification(content, color)


# DiscordNotifierインスタンスのファクトリ関数
def get_notifier(webhook_url: Optional[str] = None) -> DiscordNotifier:
    return DiscordNotifier(webhook_url)


def notify_session_start(session_type: str = "training", config_name: str = "default", notifier: Optional[DiscordNotifier] = None) -> str:
    """セッション開始を通知"""
    if notifier is None:
        # PRODUCTION=1でないならtest_mode=True
        test_mode = os.environ.get('PRODUCTION') != '1'
        notifier = get_notifier()
        notifier.test_mode = test_mode
    return notifier.start_session(session_type, config_name)


def notify_session_end(results: Dict[str, Any], session_type: str = "training", notifier: Optional[DiscordNotifier] = None) -> None:
    """セッション終了を通知"""
    notifier = notifier or get_notifier()
    notifier.end_session(results, session_type)


def notify_error(error_message: str, error_details: Optional[str] = None, notifier: Optional[DiscordNotifier] = None) -> None:
    """エラーを通知"""
    notifier = notifier or get_notifier()
    notifier.send_error_notification(error_message, error_details)


def notify_custom(title: str, message: str, color: int = 0x0099ff,
                 fields: Optional[List[Dict[str, Union[str, bool]]]] = None,
                 notifier: Optional[DiscordNotifier] = None) -> None:
    """カスタム通知"""
    notifier = notifier or get_notifier()
    notifier.send_custom_notification(title, message, color, fields)


# Dry-runテスト関数
def test_notifications() -> None:
    """通知機能のテスト（Dry-run）"""
    print("🧪 Testing Discord Notifications...")

    # テスト用のモックデータ
    mock_results = {
        'reward_stats': {
            'mean_total_reward': 1000.0,
            'std_total_reward': 100.0
        },
        'pnl_stats': {
            'mean_total_pnl': 500.0,
            'max_drawdown': 0.05
        },
        'trading_stats': {
            'total_trades': 50,
            'mean_trades_per_episode': 5.0,
            'buy_ratio': 0.6,
            'sell_ratio': 0.4
        }
    }

    # テスト用インスタンス
    test_notifier = get_notifier()

    # セッション開始テスト
    session_id = notify_session_start("test", "test_config", notifier=test_notifier)
    print(f"✅ Session start notification sent (ID: {session_id})")

    # カスタム通知テスト
    notify_custom(
        "🧪 Test Notification",
        "This is a test notification from Trading RL Bot",
        0x00ff00,
        [
            {"name": "Test Field 1", "value": "Value 1", "inline": True},
            {"name": "Test Field 2", "value": "Value 2", "inline": True}
        ],
        notifier=test_notifier
    )
    print("✅ Custom notification sent")

    # セッション終了テスト
    notify_session_end(mock_results, "test", notifier=test_notifier)
    print("✅ Session end notification sent")

    # エラーテスト
    notify_error("Test Error", "This is a test error message", notifier=test_notifier)
    print("✅ Error notification sent")

    print("🎉 All notification tests completed!")


if __name__ == "__main__":
    # コマンドラインからテスト実行
    test_notifications()