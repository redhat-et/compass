# Logging Implementation Summary

## What Was Implemented

Comprehensive logging system for Compass that captures the entire user request flow from initial message through LLM interaction to final recommendation.

## Key Features

### 1. User Request Logging
- **Location**: `backend/src/api/routes.py`
- **What's logged**:
  - User's full message
  - Conversation history count
  - Request boundaries marked with visual separators

### 2. LLM Interaction Logging
- **Location**: `backend/src/llm/ollama_client.py`
- **What's logged**:
  - INFO level: Metadata (message length, model used, response length)
  - DEBUG level: Full prompts and responses (first 500 chars preview)

### 3. Intent Extraction Logging
- **Location**: `backend/src/context_intent/extractor.py`
- **What's logged**:
  - Prompt sent to LLM for intent extraction
  - Schema used for extraction
  - Raw extracted intent JSON
  - Parsed intent summary

### 4. Workflow Progress Logging
- **Location**: `backend/src/orchestration/workflow.py`
- **What's logged**:
  - Each workflow step (1-4)
  - Traffic profile generation
  - SLO target calculation
  - Model recommendation results
  - Capacity planning decisions

## Configuration

### Environment Variable
Set `COMPASS_DEBUG=true` to enable DEBUG level logging (includes full prompts/responses)

```bash
# Enable debug logging
export COMPASS_DEBUG=true
make start-backend
```

### Log Format
```
YYYY-MM-DD HH:MM:SS - module.name - LEVEL - message
```

### Log Tags
All logs use consistent tags for easy searching:
- `[USER REQUEST]` - User request start
- `[USER MESSAGE]` - User's actual message
- `[CONVERSATION HISTORY]` - Previous messages
- `[INTENT EXTRACTION]` - Intent extraction phase
- `[LLM REQUEST]` - Request to LLM
- `[LLM RESPONSE]` - Response from LLM
- `[LLM PROMPT]` - Full prompt text (DEBUG only)
- `[LLM RESPONSE CONTENT]` - Full response text (DEBUG only)
- `[FULL PROMPT]` - Complete prompt with schema (DEBUG only)
- `[EXTRACTED INTENT]` - Raw intent JSON
- `Step 1`, `Step 2`, etc. - Workflow progress

## Files Changed

1. **backend/src/llm/ollama_client.py**
   - Added logging to `chat()` method
   - Added logging to `generate_completion()` method
   - Logs both metadata (INFO) and full content (DEBUG)

2. **backend/src/api/routes.py**
   - Added environment variable support (`COMPASS_DEBUG`)
   - Added user request logging with visual separators
   - Configured log format and level

3. **backend/src/context_intent/extractor.py**
   - Added prompt logging before LLM call
   - Added extracted intent logging after LLM response

4. **docs/LOGGING.md** (NEW)
   - Comprehensive documentation
   - Usage examples
   - Troubleshooting guide
   - Privacy and security considerations

5. **backend/logging_config.py** (NEW)
   - Reusable logging configuration module
   - Support for file and console logging
   - Debug mode support

6. **README.md**
   - Added link to LOGGING.md documentation

## Usage Examples

### View all user requests
```bash
grep "\[USER MESSAGE\]" logs/backend.log
```

### View LLM interactions (metadata)
```bash
grep "\[LLM REQUEST\]\|\[LLM RESPONSE\]" logs/backend.log
```

### View full prompts (DEBUG mode only)
```bash
grep "\[LLM PROMPT\]" logs/backend.log
```

### Follow a complete request
```bash
# Look for the separator line and follow
grep -A 50 "========" logs/backend.log | tail -60
```

## Example Output

### INFO Level (Default)
```
2025-10-16 16:45:12 - src.api.routes - INFO - ================================================================================
2025-10-16 16:45:12 - src.api.routes - INFO - [USER REQUEST] New recommendation request
2025-10-16 16:45:12 - src.api.routes - INFO - [USER MESSAGE] I need a chatbot for 1000 users
2025-10-16 16:45:12 - src.api.routes - INFO - ================================================================================
2025-10-16 16:45:12 - src.orchestration.workflow - INFO - Step 1: Extracting deployment intent
2025-10-16 16:45:12 - src.context_intent.extractor - INFO - [INTENT EXTRACTION] Sending prompt to LLM
2025-10-16 16:45:12 - src.llm.ollama_client - INFO - [LLM REQUEST] Role: user, Content length: 450 chars
2025-10-16 16:45:14 - src.llm.ollama_client - INFO - [LLM RESPONSE] Model: llama3.1:8b, Response length: 185 chars
2025-10-16 16:45:14 - src.context_intent.extractor - INFO - [EXTRACTED INTENT] {'use_case': 'chatbot', 'user_count': 1000, ...}
```

### DEBUG Level (With COMPASS_DEBUG=true)
Includes all INFO logs plus:
```
2025-10-16 16:45:12 - src.llm.ollama_client - DEBUG - [LLM PROMPT] You are an expert assistant helping users deploy Large Language Models...
2025-10-16 16:45:14 - src.llm.ollama_client - DEBUG - [LLM RESPONSE CONTENT] {"use_case": "chatbot", "user_count": 1000, "latency_requirement": "high", ...}
```

## Benefits

1. **Debugging**: Quickly identify where in the workflow issues occur
2. **Monitoring**: Track user requests and system behavior
3. **Audit Trail**: Complete record of user interactions and system decisions
4. **LLM Prompt Engineering**: See exactly what prompts are being sent to the LLM
5. **Performance Analysis**: Identify slow steps in the workflow
6. **User Behavior Analysis**: Understand what users are asking for

## Privacy Considerations

⚠️ **Important**: DEBUG mode logs contain:
- Full user messages
- Complete conversation history
- All LLM prompts and responses

**Recommendations**:
- Only use DEBUG in development/testing
- Never commit log files to git (already in .gitignore)
- Implement log rotation
- Consider PII scrubbing for production

## Next Steps (Optional Enhancements)

1. **Structured Logging**: Use JSON format for machine parsing
2. **Log Aggregation**: Send logs to ELK/Splunk/etc.
3. **Metrics**: Add Prometheus metrics for request rates, latency
4. **Request ID**: Add correlation ID to track requests across components
5. **PII Scrubbing**: Automatically redact sensitive data
6. **Log Sampling**: Only log a percentage of requests in production

## Testing

To test the logging:

1. Start backend with debug enabled:
   ```bash
   COMPASS_DEBUG=true make start-backend
   ```

2. Make a request through the UI or API

3. Check logs:
   ```bash
   tail -f logs/backend.log
   ```

4. Verify you see:
   - `[USER MESSAGE]` with your request
   - `[LLM REQUEST]` and `[LLM RESPONSE]`
   - `[LLM PROMPT]` with full prompt text (DEBUG only)
   - Workflow step logs
   - Final recommendation result
