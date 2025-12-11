from unittest.mock import Mock, patch

from ztb.trading.v433_integration_manager import ComponentManager

mock_v433_system = Mock()
mock_position_manager = Mock()
mock_risk_manager = Mock()

with patch(
    "ztb.trading.v433_integration_manager.V433IntegratedSystem",
    return_value=mock_v433_system,
), patch(
    "ztb.trading.position_manager.PositionManager", return_value=mock_position_manager
), patch("ztb.trading.risk_manager.RiskManager", return_value=mock_risk_manager):
    cm = ComponentManager("zaif")
    print("v433_system", cm.v433_system)
    print("execution_engine", cm.execution_engine)
    print("position_manager", cm.position_manager)
    print("risk_overlay", cm.risk_overlay)
    res = cm.initialize_components()
    print("init result", res)
    print("v433_system after init", cm.v433_system)
