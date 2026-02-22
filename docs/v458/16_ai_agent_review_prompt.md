# Doc16: AI Agent Review Prompt for Doc14 Implementation

## Overview

This document contains a comprehensive review prompt for an AI coding agent to evaluate the Doc14 implementation. The prompt is designed to elicit multi-perspective analysis from various angles including technical, architectural, performance, security, and user experience viewpoints.

## Review Prompt for AI Coding Agent

```
You are an expert AI coding agent tasked with conducting a comprehensive review of the Doc14 implementation in the zaif-trade-bot project. Your review must cover multiple perspectives and provide actionable insights.

## Context
- Project: zaif-trade-bot (Cryptocurrency trading bot using SAC reinforcement learning)
- Implementation: Doc14 - Walk-Forward Validation System Enhancement
- Key Features: 4-seed evaluation, baseline comparisons, AB testing, entry gates
- Technologies: Python, SAC, FastIntradayEnvV456, WalkForwardEvaluationPipeline

## Multi-Perspective Review Framework

### 1. Technical Accuracy Perspective
- Examine code correctness and algorithmic implementation
- Verify mathematical formulations (ROI calculations, Sharpe ratios, etc.)
- Check data flow integrity through the pipeline
- Validate feature engineering (MTF, regime, cyclical features)
- Assess numerical stability (overflow handling, NaN/inf checks)

### 2. Architectural Design Perspective
- Evaluate component coupling and cohesion
- Assess dependency injection patterns
- Review factory patterns and environment creation
- Analyze pipeline architecture (WalkForwardEvaluationPipeline)
- Check separation of concerns (evaluation vs reporting vs data processing)

### 3. Performance & Scalability Perspective
- Analyze computational complexity of Walk-Forward evaluation
- Evaluate memory usage patterns
- Assess parallelization opportunities
- Review feature calculation efficiency
- Check database/query optimization potential

### 4. Security & Robustness Perspective
- Identify potential security vulnerabilities
- Review error handling and exception management
- Assess input validation and sanitization
- Check for race conditions in multi-seed evaluation
- Evaluate data integrity and corruption handling

### 5. Code Quality & Maintainability Perspective
- Review code readability and documentation
- Assess test coverage and quality
- Check type hints and static analysis compliance
- Evaluate naming conventions and code organization
- Review configuration management

### 6. User Experience & Usability Perspective
- Assess command-line interface design
- Review logging and output clarity
- Evaluate configuration file usability
- Check error messages and debugging support
- Analyze result presentation and interpretation

### 7. Integration & Compatibility Perspective
- Verify compatibility with existing codebase
- Check API consistency and backward compatibility
- Assess integration with other components (training, backtesting)
- Review dependency management
- Evaluate version compatibility

### 8. Business Logic & Domain Perspective
- Validate trading logic implementation
- Assess risk management integration
- Review entry gate system effectiveness
- Check baseline comparison methodology
- Evaluate AB testing statistical validity

### 9. Testing & Validation Perspective
- Review unit test coverage and quality
- Assess integration test completeness
- Check validation test robustness
- Evaluate performance benchmarking
- Review edge case handling

### 10. Documentation & Knowledge Transfer Perspective
- Assess inline documentation quality
- Review external documentation completeness
- Check API documentation accuracy
- Evaluate onboarding documentation
- Assess knowledge transfer mechanisms

## Review Process

1. **Initial Assessment**: Provide overall quality score (1-10) with justification
2. **Strengths Analysis**: Identify 3-5 major strengths with specific examples
3. **Critical Issues**: List any blocking issues that prevent production deployment
4. **Improvement Opportunities**: Suggest 5-7 specific improvements with priority levels
5. **Risk Assessment**: Identify technical debt and long-term maintenance risks
6. **Recommendations**: Provide actionable next steps and implementation priorities

## Output Format

Structure your review response using the following format:

### Executive Summary
[Brief overview with quality score and key findings]

### Detailed Analysis by Perspective
[One section per perspective with specific findings and evidence]

### Critical Issues
[Blocking issues with severity levels]

### Recommendations
[Prioritized list of improvements with rationale]

### Risk Assessment
[Technical debt and maintenance concerns]

### Conclusion
[Final assessment and go/no-go recommendation]

## Evidence Requirements

For each finding:
- Reference specific files and line numbers
- Provide code examples where relevant
- Include quantitative metrics where applicable
- Cite relevant best practices or standards

## Review Criteria Weighting

- Technical Accuracy: 25%
- Architectural Quality: 20%
- Performance: 15%
- Security: 10%
- Code Quality: 10%
- User Experience: 5%
- Integration: 5%
- Business Logic: 5%
- Testing: 3%
- Documentation: 2%

Use this weighting to inform your overall assessment.
```

## Implementation Notes

This prompt is designed to:
1. **Comprehensive Coverage**: Cover all major aspects of software quality
2. **Structured Analysis**: Provide clear framework for systematic review
3. **Evidence-Based**: Require specific references and examples
4. **Actionable Insights**: Focus on practical recommendations
5. **Risk-Aware**: Include business impact considerations
6. **Balanced Perspective**: Weight different aspects appropriately

The AI agent should spend adequate time analyzing the codebase before providing the review, using tools like code search, file reading, and execution testing as needed.</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\docs\v458\16_ai_agent_review_prompt.md