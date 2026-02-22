Deployment Guide
===============

This section covers model deployment, production setup, and operational considerations for Zaif Trade Bot.

Deployment Overview
-------------------

Deploying trading models to production requires careful consideration of reliability, monitoring, and risk management. Zaif Trade Bot provides tools for safe and efficient model deployment.

Deployment Workflow
-------------------

1. **Model Validation**: Ensure model meets performance criteria
2. **Infrastructure Setup**: Configure production environment
3. **Risk Controls**: Implement safety measures
4. **Monitoring Setup**: Configure logging and alerting
5. **Gradual Rollout**: Start with small position sizes
6. **Performance Tracking**: Monitor live performance

Production Environment
----------------------

Infrastructure Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recommended production setup:

**Hardware Requirements**
* CPU: 4+ cores (8+ recommended)
* RAM: 16GB+ (32GB recommended)
* Storage: 100GB+ SSD
* Network: Stable low-latency connection

**Software Requirements**
* Python 3.8+
* Docker (recommended)
* Monitoring tools (Prometheus, Grafana)
* Database (PostgreSQL, Redis)

Docker Deployment
~~~~~~~~~~~~~~~~~

Deploy using Docker for consistency:

.. code-block:: dockerfile

   # Dockerfile.production
   FROM python:3.11-slim

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       && rm -rf /var/lib/apt/lists/*

   # Create app directory
   WORKDIR /app

   # Copy requirements first for better caching
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy application code
   COPY . .

   # Create non-root user
   RUN useradd --create-home --shell /bin/bash app \
       && chown -R app:app /app
   USER app

   # Expose port for monitoring
   EXPOSE 8000

   # Health check
   HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
       CMD python -c "import requests; requests.get('http://localhost:8000/health')"

   # Run application
   CMD ["python", "run_production.py"]

Build and run the container:

.. code-block:: bash

   # Build production image
   docker build -f Dockerfile.production -t ztb-production .

   # Run container
   docker run -d \
     --name ztb-trading \
     -p 8000:8000 \
     -v /data:/app/data \
     -e CONFIG_PATH=/app/config/production.yaml \
     ztb-production

Kubernetes Deployment
~~~~~~~~~~~~~~~~~~~~~

Deploy to Kubernetes for scalability:

.. code-block:: yaml

   # deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: ztb-trading
   spec:
     replicas: 1
     selector:
       matchLabels:
         app: ztb-trading
     template:
       metadata:
         labels:
           app: ztb-trading
       spec:
         containers:
         - name: ztb
           image: ztb-production:latest
           ports:
           - containerPort: 8000
           env:
           - name: CONFIG_PATH
             value: "/app/config/production.yaml"
           resources:
             requests:
               memory: "2Gi"
               cpu: "1000m"
             limits:
               memory: "4Gi"
               cpu: "2000m"
           livenessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /ready
               port: 8000
             initialDelaySeconds: 5
             periodSeconds: 5
           volumeMounts:
           - name: config-volume
             mountPath: /app/config
           - name: data-volume
             mountPath: /app/data
         volumes:
         - name: config-volume
           configMap:
             name: ztb-config
         - name: data-volume
           persistentVolumeClaim:
             claimName: ztb-data-pvc

Risk Management
---------------

Position Size Limits
~~~~~~~~~~~~~~~~~~~~

Implement strict position size controls:

.. code-block:: yaml

   # config/production.yaml
   trading:
     max_position_size: 0.05      # Maximum 5% of portfolio
     max_portfolio_risk: 0.02     # Maximum 2% portfolio risk per trade
     max_daily_loss: 0.05         # Stop trading after 5% daily loss
     max_open_positions: 3         # Maximum 3 open positions

Circuit Breakers
~~~~~~~~~~~~~~~~

Implement automatic circuit breakers:

.. code-block:: python

   from ztb.risk import CircuitBreaker

   class ProductionCircuitBreaker(CircuitBreaker):
       def __init__(self):
           self.daily_loss_limit = 0.05
           self.volatility_threshold = 0.03
           self.liquidity_threshold = 0.7

       def should_stop_trading(self, portfolio, market_data):
           # Check daily loss
           if portfolio.daily_pnl < -self.daily_loss_limit:
               return True, "Daily loss limit exceeded"

           # Check volatility
           if market_data.volatility > self.volatility_threshold:
               return True, "High volatility detected"

           # Check liquidity
           if market_data.liquidity_score < self.liquidity_threshold:
               return True, "Low liquidity detected"

           return False, None

Emergency Stop
~~~~~~~~~~~~~~

Implement emergency stop functionality:

.. code-block:: python

   from ztb.trading import EmergencyStop

   # Initialize emergency stop
   emergency_stop = EmergencyStop(
       stop_file="/tmp/ztb_emergency_stop",
       api_key="your_api_key"
   )

   # Check for emergency stop
   if emergency_stop.should_stop():
       logger.critical("Emergency stop activated")
       # Close all positions
       portfolio.close_all_positions()
       # Shutdown trading
       trading_engine.shutdown()

Monitoring & Alerting
---------------------

Application Monitoring
~~~~~~~~~~~~~~~~~~~~~~~

Monitor application health and performance:

.. code-block:: python

   from ztb.monitoring import ApplicationMonitor

   # Initialize monitoring
   monitor = ApplicationMonitor()

   # Track key metrics
   monitor.track_metric('portfolio_value', portfolio.total_value)
   monitor.track_metric('open_positions', len(portfolio.positions))
   monitor.track_metric('daily_pnl', portfolio.daily_pnl)

   # Health check endpoint
   @app.get("/health")
   def health_check():
       return {
           "status": "healthy",
           "timestamp": datetime.now(),
           "portfolio_value": portfolio.total_value
       }

Prometheus Metrics
~~~~~~~~~~~~~~~~~~

Export metrics for Prometheus monitoring:

.. code-block:: python

   from prometheus_client import Gauge, Counter, Histogram
   import prometheus_client

   # Define metrics
   PORTFOLIO_VALUE = Gauge('ztb_portfolio_value', 'Current portfolio value')
   TRADES_TOTAL = Counter('ztb_trades_total', 'Total number of trades', ['outcome'])
   TRADE_LATENCY = Histogram('ztb_trade_latency_seconds', 'Trade execution latency')

   # Update metrics
   PORTFOLIO_VALUE.set(portfolio.total_value)
   TRADES_TOTAL.labels(outcome='profit').inc()
   TRADE_LATENCY.observe(trade_duration)

   # Expose metrics endpoint
   @app.get("/metrics")
   def metrics():
       return prometheus_client.generate_latest()

Alerting Rules
~~~~~~~~~~~~~~

Configure alerting for critical events:

.. code-block:: yaml

   # alert_rules.yaml
   groups:
   - name: ztb_alerts
     rules:
     - alert: HighPortfolioLoss
       expr: ztb_portfolio_daily_pnl < -0.05
       for: 5m
       labels:
         severity: critical
       annotations:
         summary: "Portfolio loss exceeds 5%"

     - alert: TradingStopped
       expr: up{job="ztb-trading"} == 0
       for: 5m
       labels:
         severity: critical
       annotations:
         summary: "Trading service is down"

     - alert: HighVolatility
       expr: ztb_market_volatility > 0.05
       for: 10m
       labels:
         severity: warning
       annotations:
         summary: "Market volatility is high"

Logging & Auditing
------------------

Production Logging
~~~~~~~~~~~~~~~~~~

Configure comprehensive logging for production:

.. code-block:: yaml

   # config/production.yaml
   logging:
     level: "INFO"
     format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
     handlers:
       - type: "file"
         filename: "/var/log/ztb/trading.log"
         maxBytes: 10485760  # 10MB
         backupCount: 5
       - type: "syslog"
         address: ("logs.papertrailapp.com", 12345)
     loggers:
       ztb.trading: "DEBUG"
       ztb.risk: "INFO"

Audit Logging
~~~~~~~~~~~~~

Log all trading decisions and actions:

.. code-block:: python

   import logging
   import json

   class AuditLogger:
       def __init__(self):
           self.logger = logging.getLogger('ztb.audit')
           self.logger.setLevel(logging.INFO)

       def log_trade(self, trade_data):
           audit_entry = {
               'timestamp': datetime.now().isoformat(),
               'action': 'trade_executed',
               'trade_id': trade_data['id'],
               'symbol': trade_data['symbol'],
               'side': trade_data['side'],
               'quantity': trade_data['quantity'],
               'price': trade_data['price'],
               'reason': trade_data.get('reason', 'automated')
           }
           self.logger.info(json.dumps(audit_entry))

       def log_risk_event(self, event_data):
           audit_entry = {
               'timestamp': datetime.now().isoformat(),
               'action': 'risk_event',
               'event_type': event_data['type'],
               'severity': event_data['severity'],
               'description': event_data['description']
           }
           self.logger.info(json.dumps(audit_entry))

Backup & Recovery
-----------------

Data Backup
~~~~~~~~~~~

Implement regular data backups:

.. code-block:: bash

   #!/bin/bash
   # backup.sh

   BACKUP_DIR="/backups"
   TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

   # Backup database
   pg_dump ztb_production > "$BACKUP_DIR/db_$TIMESTAMP.sql"

   # Backup models
   tar -czf "$BACKUP_DIR/models_$TIMESTAMP.tar.gz" /app/models/

   # Backup configuration
   cp /app/config/production.yaml "$BACKUP_DIR/config_$TIMESTAMP.yaml"

   # Clean old backups (keep last 30 days)
   find "$BACKUP_DIR" -name "*.sql" -mtime +30 -delete
   find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete

Recovery Procedures
~~~~~~~~~~~~~~~~~~~

Document recovery procedures:

.. code-block:: python

   from ztb.recovery import RecoveryManager

   class ProductionRecoveryManager(RecoveryManager):
       def recover_from_crash(self):
           """Recover from system crash"""
           # Load last known good state
           last_state = self.load_last_backup()

           # Reconcile positions with broker
           self.reconcile_positions(last_state)

           # Resume trading with reduced position sizes
           self.resume_trading(safe_mode=True)

       def recover_from_data_loss(self):
           """Recover from data corruption"""
           # Restore from backup
           self.restore_from_backup()

           # Validate data integrity
           self.validate_data_integrity()

           # Rebuild derived data
           self.rebuild_derived_data()

Performance Optimization
------------------------

Production Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~

Optimize for production performance:

.. code-block:: python

   # config/production.yaml
   optimization:
     use_gpu: true                    # Use GPU if available
     batch_size: 1024                 # Larger batch sizes for inference
     cache_predictions: true          # Cache model predictions
     async_execution: true            # Asynchronous trade execution
     memory_optimization: true        # Memory-efficient data structures

Model Serving Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimize model inference for low latency:

.. code-block:: python

   from ztb.serving import OptimizedModelServer

   # Initialize optimized server
   server = OptimizedModelServer(
       model_path='models/production_model.zip',
       batch_size=32,
       use_gpu=True,
       num_workers=4
   )

   # Warm up model
   server.warm_up()

   # Serve predictions
   predictions = server.predict_batch(features)

Scaling Considerations
----------------------

Horizontal Scaling
~~~~~~~~~~~~~~~~~~

Scale horizontally for high throughput:

.. code-block:: yaml

   # k8s/hpa.yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: ztb-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: ztb-trading
     minReplicas: 1
     maxReplicas: 10
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70

Load Balancing
~~~~~~~~~~~~~~

Distribute load across multiple instances:

.. code-block:: yaml

   # k8s/service.yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: ztb-service
   spec:
     selector:
       app: ztb-trading
     ports:
     - port: 80
       targetPort: 8000
     type: LoadBalancer

Testing & Validation
--------------------

Pre-Deployment Testing
~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive testing before deployment:

.. code-block:: python

   from ztb.testing import ProductionTester

   # Run production readiness tests
   tester = ProductionTester()

   # Test scenarios
   test_results = tester.run_tests([
       'performance_test',
       'stress_test',
       'failover_test',
       'data_integrity_test'
   ])

   # Validate results
   if test_results.all_passed():
       print("Ready for production deployment")
   else:
       print("Fix issues before deployment:")
       for failure in test_results.failures:
           print(f"- {failure}")

A/B Testing
~~~~~~~~~~~

Test new models against production:

.. code-block:: python

   from ztb.testing import ABTester

   # Setup A/B test
   ab_tester = ABTester(
       control_model='models/current_production.zip',
       variant_model='models/new_candidate.zip',
       traffic_split=0.1  # 10% to new model
   )

   # Run A/B test
   results = ab_tester.run_test(duration_days=7)

   # Analyze results
   if results.variant_better():
       print("New model performs better - consider deployment")
   else:
       print("Stick with current model")

Gradual Rollout
~~~~~~~~~~~~~~~~

Deploy new versions gradually:

.. code-block:: python

   from ztb.deployment import GradualRollout

   # Configure gradual rollout
   rollout = GradualRollout(
       new_version='v2.1.0',
       rollout_percentage=5,  # Start with 5%
       increase_interval_hours=24,
       max_percentage=100
   )

   # Monitor rollout
   while not rollout.is_complete():
       current_percentage = rollout.get_current_percentage()
       performance = rollout.monitor_performance()

       if performance.is_acceptable():
           rollout.increase_percentage(5)  # Increase by 5%
       else:
           rollout.rollback()

       time.sleep(3600)  # Check every hour

Compliance & Security
---------------------

Regulatory Compliance
~~~~~~~~~~~~~~~~~~~~~

Ensure regulatory compliance:

.. code-block:: python

   from ztb.compliance import ComplianceManager

   # Initialize compliance
   compliance = ComplianceManager(
       jurisdiction='japan',
       regulations=['fsa_guidelines', 'tax_reporting']
   )

   # Check trade compliance
   if compliance.is_trade_allowed(trade):
       execute_trade(trade)
   else:
       log_compliance_violation(trade)

Security Best Practices
~~~~~~~~~~~~~~~~~~~~~~~

Implement security measures:

.. code-block:: yaml

   # config/security.yaml
   security:
     api_keys_encrypted: true
     secrets_management: "vault"
     network_isolation: true
     audit_logging: true
     access_control: "rbac"

   authentication:
     method: "oauth2"
     mfa_required: true
     session_timeout: 3600

Troubleshooting
---------------

Common Deployment Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**High Latency**
* Optimize model inference
* Use GPU acceleration
* Implement caching

**Memory Issues**
* Monitor memory usage
* Implement memory limits
* Use memory-efficient data structures

**Trading Halts**
* Check circuit breaker conditions
* Verify market data connectivity
* Review risk limits

**Data Corruption**
* Implement data validation
* Regular backups
* Use transactional databases

Best Practices
--------------

1. **Start Small**: Begin with small position sizes in production
2. **Monitor Everything**: Implement comprehensive monitoring and alerting
3. **Automate Recovery**: Have automated procedures for common issues
4. **Regular Backups**: Backup data and models regularly
5. **Security First**: Implement proper security measures
6. **Compliance**: Ensure regulatory compliance
7. **Testing**: Thoroughly test before deployment
8. **Gradual Rollout**: Roll out changes gradually with monitoring

Next Steps
----------

* :doc:`../api/index` - Explore the API reference
* :doc:`../examples/index` - See deployment examples
* :doc:`../troubleshooting/index` - Get help with common issues
