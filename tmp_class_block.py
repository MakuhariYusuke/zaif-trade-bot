class PPOTrainer:
    """PPOトレーニングマネージャー"""

    def __init__(
            self, 
            data_path: str, 
            config: Optional[Dict[str, Any]] = None, 
            checkpoint_interval: int = 10000, 
            checkpoint_dir: str = 'models/checkpoints'
        ) -> None:
        """
        コンストラクタ

        Args:
            data_path (str): トレーニングデータのパス
            config (Optional[dict]): トレーニング設定
            checkpoint_interval (int): チェックポイント保存の間隔（ステップ数）
            checkpoint_dir (str): チェックポイント保存ディレクトリ
        """
        # CPU最適化を最初に適用
        apply_cpu_tuning()
        
        self.data_path = Path(data_path)
        self.data_path = Path(data_path)
        self.config = config or self._get_default_config()
        
        # CPU最適化設定
        self._setup_cpu_optimization()
        
        # config をフラット化（training セクションをトップレベルに）
        if 'training' in self.config:
            self.config.update(self.config['training'])
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir)

        # ログディレクトリの設定
        self.log_dir = Path(self.config['log_dir'])
        self.log_dir.mkdir(exist_ok=True)
        # モデルの保存ディレクトリ
        self.model_dir = Path(self.config['model_dir'])
        self.model_dir.mkdir(exist_ok=True)

        # データの読み込み
        self.df = self._load_data()

        # 環境の作成
        self.env = self._create_env()

    def _setup_cpu_optimization(self) -> None:
        """CPU最適化設定"""
        from ..utils.perf.cpu_tune import auto_config_threads
        
        # 環境変数から設定取得
        num_processes = int(os.environ.get("PARALLEL_PROCESSES", "1"))
        pin_cores_str = os.environ.get("CPU_AFFINITY")
        pin_to_cores = [int(x) for x in pin_cores_str.split(",")] if pin_cores_str else None
        
        # 自動設定決定
        cpu_config = auto_config_threads(num_processes, pin_to_cores)
        
        # 環境変数設定
        for key, value in cpu_config.items():
            if key.startswith(('OMP_', 'MKL_', 'OPENBLAS_', 'NUMEXPR_')):
                os.environ[key] = str(value)
        
        # PyTorch設定
        torch.set_num_threads(cpu_config['torch_threads'])
        torch.backends.mkldnn.enabled = True  # type: ignore
        
        # ログ出力
        logging.info(f"CPU: phys={cpu_config['physical_cores']}, log={cpu_config['logical_cores']}, "
                     f"procs={cpu_config['num_processes']}, pin={cpu_config['pin_to_cores']}, "
                     f"torch={cpu_config['torch_threads']}, OMP={cpu_config['OMP_NUM_THREADS']}, "
                     f"MKL={cpu_config['MKL_NUM_THREADS']}, OPENBLAS={cpu_config['OPENBLAS_NUM_THREADS']}")

    def _get_default_config(self) -> Dict[str, Any]:
        """
        デフォルトのPPOトレーニング設定を返します。

        Returns:
            dict: デフォルト設定の辞書
        """
        return {
            'total_timesteps': 200000,  # 本番用と同じ値に統一
            'eval_freq': 5000,
            'n_eval_episodes': 5,
            'batch_size': 64,
            'n_steps': 2048,
            'gamma': 0.99,
            'learning_rate': 3e-4,
            'ent_coef': 0.01,
            'clip_range': 0.2,
            'n_epochs': 10,
            'gae_lambda': 0.95,
            'max_grad_norm': 0.5,
            'vf_coef': 0.5,
            'log_dir': './logs/',
            'model_dir': './models/',
            'tensorboard_log': './tensorboard/',
            'verbose': 1,
            'seed': 42,
        }

    def _load_data(self) -> pd.DataFrame:
        """
        指定されたパスからトレーニングデータを読み込みます。
        ワイルドカードを含むパスにも対応し、複数のファイルを結合できます。

        Returns:
            pd.DataFrame: 読み込まれたデータフレーム

        Raises:
            FileNotFoundError: データファイルが見つからない場合
            ValueError: サポートされていないファイル形式または有効なデータファイルがない場合
        """
        data_path = Path(self.data_path)

        # メモリ効率化: FeatureCacheチェック
        memory_config = self.config.get('memory', {})
        if memory_config.get('enable_cache', False):
            cache = FeatureCache(
                memory_config.get('cache_dir', 'data/cache'),
                memory_config.get('cache_max_mb', 1000),
                memory_config.get('max_age_days', 7),
                memory_config.get('compressor', 'zstd')
            )
            params = {
                "data_path": str(data_path),
                "version": "v1",
                "downcast": memory_config.get('downcast', True)
            }
            cached = cache.get(str(data_path), params)
            if cached is not None:
                df = pickle.loads(cached)
                # メモリ使用量計算
                cached_size_mb = len(cached) / (1024 * 1024)
                logging.info(f"[CACHE] Hit ({cached_size_mb:.1f} MB loaded) for {data_path}")
                # ダウンキャスト適用
                if memory_config.get('downcast', True):
                    df = downcast_df(df, 
                                   float_dtype=memory_config.get('float_dtype', 'float32'),
                                   int_dtype=memory_config.get('int_dtype', 'int32'))
                return cast(pd.DataFrame, df)
            else:
                logging.info(f"[CACHE] Miss for {data_path}")

        # ワイルドカードが含まれる場合
        if '*' in str(data_path):
            # globパターンでファイルを検索
            import glob
            file_paths = glob.glob(str(data_path))

            if not file_paths:
                raise FileNotFoundError(f"No files found matching pattern: {data_path}")
            logging.info(f"Found {len(file_paths)} files matching pattern: {data_path}")

            # すべてのファイルを読み込んで結合
            dfs = []
            for file_path_str in file_paths:
                file_path = Path(file_path_str)
                if file_path.suffix == '.parquet':
                    df = pd.read_parquet(file_path)
                elif file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    logging.warning(f"Skipping unsupported file: {file_path}")
                    continue

                dfs.append(df)
                logging.info(f"Loaded {file_path.name}: {len(df)} rows")
                print(f"Loaded {file_path.name}: {len(df)} rows")

            if not dfs:
                raise ValueError("No valid data files found")

            # データを結合
            df = pd.concat(dfs, ignore_index=True)

            # タイムスタンプでソート
            if 'ts' in df.columns:
                df = df.sort_values('ts').reset_index(drop=True)

        else:
            # 単一ファイルの場合
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")

            if data_path.suffix == '.parquet':
                df = pd.read_parquet(data_path)
            elif data_path.suffix == '.csv':
                df = pd.read_csv(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
        logging.info(f"Total loaded data: {len(df)} rows, {len(df.columns)} columns")
        logging.info(f"Columns: {list(df.columns)}")
        print(f"Columns: {list(df.columns)}")

        # メモリ効率化: ダウンキャスト
        memory_config = self.config.get('memory', {})
        if memory_config.get('downcast', True):
            df = downcast_df(df, 
                           float_dtype=memory_config.get('float_dtype', 'float32'),
                           int_dtype=memory_config.get('int_dtype', 'int32'))
            logging.info(f"[MEMORY] Downcast applied: float->{memory_config.get('float_dtype', 'float32')}, int->{memory_config.get('int_dtype', 'int32')}")

        # メモリ効率化: キャッシュ保存
        if memory_config.get('enable_cache', False):
            cache = FeatureCache(
                memory_config.get('cache_dir', 'data/cache'),
                memory_config.get('cache_max_mb', 1000)
            )
            params = {
                "data_path": str(data_path),
                "version": "v1",
                "downcast": memory_config.get('downcast', True)
            }
            # メモリ使用量計算
            data_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            compressed = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
            compressed_size_mb = len(compressed) / (1024 * 1024)
            cache.put(str(data_path), params, compressed)
            logging.info(f"[CACHE] Saved ({data_size_mb:.1f} MB -> {compressed_size_mb:.1f} MB compressed) for {data_path}")

        return df

    def _create_env(self) -> Any:
        """
        トレーニング用のHeavyTradingEnv環境を作成し、Monitorでラップします。
        設定ファイルから取引手数料を読み込みます。

        Returns:
            Monitor: モニターでラップされた環境オブジェクト
        """
        # rl_config.jsonから手数料設定を読み込み
        config_path: Optional[str] = os.environ.get("RL_CONFIG_PATH")
        if config_path is None:
            # プロジェクトルートの絶対パスをデフォルトに
            config_path = str(Path(__file__).parent.parent.parent.parent / "rl_config.json")
        config_path_obj = Path(config_path)
        transaction_cost = 0.001  # デフォルト
        if config_path_obj.exists():
            with open(config_path_obj, 'r') as f:
                rl_config = json.load(f)
            fee_config = rl_config.get('fee_model', {})
            transaction_cost = fee_config.get('default_fee_rate', 0.001)
        env_config = {
            'reward_scaling': 1.0,
            'transaction_cost': transaction_cost,
            'max_position_size': 1.0,
            'risk_free_rate': 0.0,
        }

        env: Union[HeavyTradingEnv, Monitor[HeavyTradingEnv, Any]] = HeavyTradingEnv(self.df, env_config)
        env = Monitor(env, str(self.log_dir / 'monitor.csv'))

        return env

    def train(self, notifier: Optional[Any] = None, session_id: Optional[str] = None) -> PPO:
        """
        設定に基づいてPPOモデルをトレーニングします。

        Args:
            notifier: 通知用のオプショナルなNotifierオブジェクト
            session_id: トレーニングセッションのID

        Returns:
            PPO: トレーニング済みのPPOモデル

        Raises:
            Exception: トレーニング中にエラーが発生した場合
        """
        # I/O最適化: ログバッファリングを設定
        buffer_handler = BufferingHandler(1000)  # 1000メッセージごとにフラッシュ
        buffer_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        buffer_handler.setFormatter(formatter)
        logging.getLogger().addHandler(buffer_handler)
        
        logging.info("Starting PPO training...")

        # モデルの作成
        model = PPO(
            'MlpPolicy',
            self.env,
            learning_rate=self.config['learning_rate'],
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_range=self.config['clip_range'],
            ent_coef=self.config['ent_coef'],
            vf_coef=self.config['vf_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            tensorboard_log=self.config['tensorboard_log'],
            verbose=self.config['verbose'],
            seed=self.config['seed'],
        )

        # コールバックの設定
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=str(self.model_dir / 'best_model'),
            log_path=str(self.log_dir / 'eval'),
            eval_freq=self.config['eval_freq'],
            n_eval_episodes=self.config['n_eval_episodes'],
            deterministic=True,
            render=False,
        )

        tensorboard_callback = TensorBoardCallback(eval_freq=self.config['eval_freq'])

        # チェックポイントコールバックの設定
        checkpoint_callback = CheckpointCallback(
            save_freq=self.checkpoint_interval,
            save_path=str(self.checkpoint_dir),
            name_prefix=f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            verbose=1,
            notifier=notifier,
            session_id=session_id,
            light_mode=self.config.get('training', {}).get('checkpoint_light', False)
        )

        # 安全策コールバックの設定
        safety_callback = SafetyCallback(max_zero_trades=1000, verbose=1)

        try:
            # トレーニングの実行
            logging.info(f"Training started with total_timesteps: {self.config['total_timesteps']}")
            model.learn(
                total_timesteps=self.config['total_timesteps'],
                callback=[eval_callback, tensorboard_callback, checkpoint_callback, safety_callback],
                progress_bar=True,
            )

            # 最終モデルの保存
            model.save(str(self.model_dir / 'final_model'))
            logging.info(f"Model saved to {self.model_dir / 'final_model'}")

            # トレーニング完了時のキャッシュ統計出力
            memory_config = self.config.get('memory', {})
            memory_config = self.config.get('memory', {})
            if memory_config.get('enable_cache', False):
                cache = FeatureCache(
                    memory_config.get('cache_dir', 'data/cache'),
                    memory_config.get('cache_max_mb', 1000)
                )
                stats = cache.get_stats()
                logging.info(f"[CACHE] Final stats: {stats['hits']} hits, {stats['misses']} misses, "
                            f"{stats['hit_rate']:.1f}% hit rate, {stats['evictions']} evictions, "
                            f"{stats['compression_ratio']:.1f}% compression ratio")

                # キャッシュ健全性チェック
                health = cache.monitor_cache_health()
                if health['warnings']:
                    for warning in health['warnings']:
                        logging.warning(f"[CACHE] {warning}")
                else:
                    logging.info(f"[CACHE] Health check passed - {health['size_mb']:.1f}MB used")

            # I/O最適化: バッファをフラッシュ
            buffer_handler.flush()
            
            return model

        except Exception as e:
            logging.exception(f"Training failed: {e}")
            # I/O最適化: エラー時もバッファをフラッシュ
            buffer_handler.flush()
            if notifier:
                notifier.send_error_notification("Training Failed", f"Session {session_id}: {str(e)}")
            raise

    def evaluate(self, model_path: Optional[str] = None, n_episodes: int = 10) -> Dict[str, Any]:
        """
        指定されたモデルを評価し、統計情報を返します。

        Args:
            model_path (Optional[str]): 評価するモデルのパス。Noneの場合は最良モデルを使用。
            n_episodes (int): 評価エピソード数

        Returns:
            dict: 評価結果の統計情報（平均報酬など）
        """
        if model_path is None:
            model_path = str(self.model_dir / 'best_model')

        # モデルの読み込み
        model = PPO.load(model_path)

        # 評価環境の作成
        eval_env = DummyVecEnv([lambda: self._create_env()])

        # 評価の実行
        episode_rewards = []
        episode_lengths: list[int] = []

        for episode in range(n_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                # obsがタプルの場合、最初の要素（観測データ）を使用
                predict_obs = obs[0] if isinstance(obs, tuple) else obs
                action, _ = model.predict(predict_obs, deterministic=True)
                obs, reward, done_vec, _ = eval_env.step(action)
                done = done_vec[0]
                episode_reward += reward[0]
                episode_length += 1

            episode_rewards.append(episode_reward)
            logging.info(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

        # 統計の計算
        stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'total_episodes': n_episodes,
        }

        # 結果の保存
        results_path = self.log_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logging.info(f"Evaluation results saved to {results_path}")
        return stats

    def visualize_training(self) -> None:
        """
        monitor.csvログからトレーニング結果を可視化し、画像を保存します。
        """
        # モニターログの読み込み
        monitor_file = self.log_dir / 'monitor.csv'
        if monitor_file.exists():
            # ヘッダー行数を自動判定
            with open(monitor_file, 'r', encoding='utf-8') as f:
                header_lines = 0
                for line in f:
                    if line.startswith('#'):
                        header_lines += 1
                    else:
                        break
            monitor_df = pd.read_csv(monitor_file, skiprows=header_lines)

            # プロットの作成
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # リワードの推移
            axes[0, 0].plot(monitor_df['r'], alpha=0.7)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)

            # エピソード長の推移
            axes[0, 1].plot(monitor_df['l'], alpha=0.7)
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Length')
            axes[0, 1].grid(True)

            # リワードのヒストグラム
            axes[1, 0].hist(monitor_df['r'], bins=50, alpha=0.7)
            axes[1, 0].set_title('Reward Distribution')
            axes[1, 0].set_xlabel('Reward')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)

            # 累積リワード
            axes[1, 1].plot(np.cumsum(monitor_df['r']), alpha=0.7)
            axes[1, 1].set_title('Cumulative Rewards')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Cumulative Reward')
            axes[1, 1].grid(True)

            plt.tight_layout()
            plt.show()

            logging.info(f"Training visualization saved to {self.log_dir / 'training_visualization.png'}")
            print(f"Training visualization saved to {self.log_dir / 'training_visualization.png'}")


