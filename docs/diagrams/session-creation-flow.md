# Session Creation Flow - Model Assignment Analysis

## Overview

This document traces the flow of session creation and model assignment to identify where the default model (`gpt-4o`) is being set instead of the environment variable `DSAGENT_DEFAULT_MODEL`.

## The Problem

When creating a session via the API:
- `DSAGENT_DEFAULT_MODEL=groq/openai/gpt-oss-120b` is set in the environment
- But the session gets saved with `model: "gpt-4o"`
- This causes validation to fail because `OPENAI_API_KEY` is not set

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         POST /api/sessions                                   │
│                         (CreateSessionRequest)                               │
│                                                                              │
│  request.model = None  (not specified in request)                           │
│  request.hitl_mode = "none"                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: session_manager.create_session(name=request.name)                  │
│  File: src/dsagent/session/manager.py:74-110                                │
│                                                                              │
│  Creates Session object with:                                                │
│    - id: auto-generated                                                      │
│    - name: from request or auto-generated                                    │
│    - model: None  (default from Session model)                              │
│    - hitl_mode: "none" (default from Session model)                         │
│                                                                              │
│  Saves to DB via self.store.save(session)                                   │
│  At this point: session.model = None                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: session.model = request.model                                       │
│  File: src/dsagent/server/routes/sessions.py:84                              │
│                                                                              │
│  session.model = None  (because request.model is None)                      │
│  session.hitl_mode = "none"                                                  │
│  session_manager.save_session(session)                                       │
│                                                                              │
│  At this point: session.model = None (saved to DB)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: connection_manager.get_or_create_agent(session.id, model=None)     │
│  File: src/dsagent/server/routes/sessions.py:89-93                           │
│                                                                              │
│  Calls with model=None, hitl_mode=None                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: _get_or_create_agent()                                              │
│  File: src/dsagent/server/manager.py:241-310                                 │
│                                                                              │
│  session = self._session_manager.get_or_create(session_id)                  │
│  # Returns existing session with model=None                                  │
│                                                                              │
│  effective_model = (                                                         │
│      model                              # None (from parameter)             │
│      or getattr(session, "model", None) # None (from session)               │
│      or self._default_model             # ??? (from AgentConnectionManager) │
│      or os.getenv("DSAGENT_DEFAULT_MODEL")  # "groq/openai/gpt-oss-120b"   │
│      or os.getenv("LLM_MODEL", "gpt-4o")    # fallback                      │
│  )                                                                           │
│                                                                              │
│  ⚠️  QUESTION: What is self._default_model at this point?                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: ConversationalAgent created with effective_model                    │
│  File: src/dsagent/server/manager.py:292-302                                 │
│                                                                              │
│  config = ConversationalAgentConfig(model=effective_model, ...)              │
│  agent = ConversationalAgent(config=config, session=session, ...)            │
│                                                                              │
│  Note: The session object passed here has model=None                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: agent.start()                                                       │
│  File: src/dsagent/agents/conversational.py:305-340                          │
│                                                                              │
│  validate_configuration(self.config.model)                                   │
│  # This is where the error occurs!                                           │
│                                                                              │
│  ⚠️  self.config.model should be effective_model, but error says "gpt-4o"   │
└─────────────────────────────────────────────────────────────────────────────┘

## Key Investigation Points

### 1. AgentConnectionManager Initialization

File: `src/dsagent/server/app.py:49-54`

```python
connection_manager = AgentConnectionManager(
    session_manager=session_manager,
    default_model=settings.default_model,  # <-- What is this value?
    default_hitl_mode=settings.default_hitl_mode,
)
```

### 2. ServerSettings.default_model

File: `src/dsagent/server/deps.py:21-44`

```python
class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="DSAGENT_",  # <-- Reads DSAGENT_DEFAULT_MODEL
    )

    default_model: Optional[str] = None  # <-- Maps to DSAGENT_DEFAULT_MODEL
```

### 3. Verified in Container

```bash
$ docker exec <container> python3 -c "
from dsagent.server.deps import get_settings
s = get_settings()
print(f'default_model from settings: {s.default_model!r}')
"
# Output: default_model from settings: 'groq/openai/gpt-oss-120b'
```

**Settings correctly reads the environment variable!**

## The Mystery

1. `settings.default_model` = `'groq/openai/gpt-oss-120b'` ✅
2. `AgentConnectionManager` is initialized with `default_model=settings.default_model`
3. Yet somehow `self.config.model` in `ConversationalAgent` is `'gpt-4o'`

## Hypothesis

The database shows `"model":"gpt-4o"` for the session. But we traced that:
1. Session is created with `model=None`
2. Session is saved with `model=None`
3. Agent is created separately

**Where does `gpt-4o` come from in the database?**

Looking at the DB record:
```json
{"model":"gpt-4o", ...}
```

This suggests something is WRITING `gpt-4o` to the session AFTER creation.

## Next Steps

1. Check if `ConversationalAgentConfig` has a default that overrides
2. Check if session is saved again somewhere with the model
3. Add logging to track exactly when/where model gets set

## ConversationalAgentConfig Default

File: `src/dsagent/agents/conversational.py:62-66`

```python
@dataclass
class ConversationalAgentConfig:
    """Configuration for the conversational agent."""
    model: str = "gpt-4o"  # <-- DEFAULT IS gpt-4o!
```

**FOUND IT!** The `ConversationalAgentConfig` has a default of `"gpt-4o"`.

But wait - we're explicitly passing `model=effective_model` when creating the config:
```python
config = ConversationalAgentConfig(
    model=effective_model,  # Should override the default
    hitl_mode=effective_hitl_mode,
)
```

So unless `effective_model` is somehow being set to `None` or empty string...

## Root Cause Candidates

1. **`self._default_model` is None** - AgentConnectionManager not getting the value from settings
2. **`os.getenv("DSAGENT_DEFAULT_MODEL")` returns None** - env var not available in the async context
3. **Session already has model=gpt-4o from a previous run** - stale data in DB

## Testing Required

1. Print `self._default_model` in `_get_or_create_agent`
2. Print `os.getenv("DSAGENT_DEFAULT_MODEL")` at runtime
3. Delete sessions.db and test fresh
