"""Tests for ConversationalAgent."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from dataclasses import dataclass

from dsagent.agents.conversational import (
    ConversationalAgent,
    ConversationalAgentConfig,
    ChatResponse,
)
from dsagent.prompts import PromptBuilder
from dsagent.session import Session, SessionManager, ConversationHistory
from dsagent.schema.models import ExecutionResult


class TestConversationalAgentConfig:
    """Tests for ConversationalAgentConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConversationalAgentConfig()

        assert config.model == "gpt-4o"
        assert config.temperature == 0.3
        assert config.max_tokens == 4096
        assert config.code_timeout == 300
        assert config.workspace == Path("./workspace")

    def test_custom_config(self, tmp_path):
        """Test custom configuration."""
        config = ConversationalAgentConfig(
            model="claude-3-sonnet",
            temperature=0.5,
            max_tokens=8192,
            code_timeout=600,
            workspace=tmp_path,
        )

        assert config.model == "claude-3-sonnet"
        assert config.temperature == 0.5
        assert config.max_tokens == 8192
        assert config.code_timeout == 600
        assert config.workspace == tmp_path


class TestChatResponse:
    """Tests for ChatResponse."""

    def test_minimal_response(self):
        """Test response with minimal data."""
        response = ChatResponse(content="Hello")

        assert response.content == "Hello"
        assert response.code is None
        assert response.execution_result is None
        assert response.has_answer is False
        assert response.answer is None
        assert response.thinking is None

    def test_full_response(self):
        """Test response with all fields."""
        exec_result = ExecutionResult(
            stdout="42",
            success=True
        )
        response = ChatResponse(
            content="<code>print(42)</code>",
            code="print(42)",
            execution_result=exec_result,
            has_answer=True,
            answer="The answer is 42",
            thinking="Let me calculate...",
        )

        assert response.content == "<code>print(42)</code>"
        assert response.code == "print(42)"
        assert response.execution_result == exec_result
        assert response.has_answer is True
        assert response.answer == "The answer is 42"
        assert response.thinking == "Let me calculate..."


class TestConversationalAgent:
    """Tests for ConversationalAgent."""

    def test_init_default(self):
        """Test agent initialization with defaults."""
        agent = ConversationalAgent()

        assert agent.config is not None
        assert agent.config.model == "gpt-4o"
        assert agent._session is None
        assert agent._executor is None
        assert agent.is_running is False

    def test_init_with_config(self, tmp_path):
        """Test agent initialization with custom config."""
        config = ConversationalAgentConfig(
            model="gpt-4-turbo",
            workspace=tmp_path,
        )
        agent = ConversationalAgent(config=config)

        assert agent.config.model == "gpt-4-turbo"
        assert agent.config.workspace == tmp_path

    def test_init_with_session(self, tmp_path):
        """Test agent initialization with existing session."""
        session = Session.new(workspace_path=str(tmp_path))
        agent = ConversationalAgent(session=session)

        assert agent._session == session

    def test_session_property(self):
        """Test session property."""
        session = Session.new()
        agent = ConversationalAgent(session=session)

        assert agent.session == session

    def test_is_running_false_when_not_started(self):
        """Test is_running returns False when not started."""
        agent = ConversationalAgent()

        assert agent.is_running is False

    def test_extract_code(self):
        """Test code extraction from response."""
        agent = ConversationalAgent()

        text = "Here's the code:\n<code>\nprint('hello')\n</code>\n"
        code = agent._extract_code(text)

        assert code == "print('hello')"

    def test_extract_code_incomplete_tag(self):
        """Test code extraction with incomplete tag (LLM stopped at </code>)."""
        agent = ConversationalAgent()

        # LLM stopped before closing tag
        text = "Here's the code:\n<code>\nprint('hello')\n"
        code = agent._extract_code(text)

        assert code == "print('hello')"

    def test_extract_code_markdown_python(self):
        """Test code extraction from markdown python block."""
        agent = ConversationalAgent()

        text = "Here's the code:\n```python\nprint('hello')\n```\n"
        code = agent._extract_code(text)

        assert code == "print('hello')"

    def test_extract_code_markdown_generic(self):
        """Test code extraction from generic markdown block."""
        agent = ConversationalAgent()

        text = "Here's the code:\n```\nprint('hello')\n```\n"
        code = agent._extract_code(text)

        assert code == "print('hello')"

    def test_extract_code_prefers_code_tags(self):
        """Test that <code> tags are preferred over markdown."""
        agent = ConversationalAgent()

        text = "```python\nmarkdown_code\n```\n<code>\nreal_code\n</code>"
        code = agent._extract_code(text)

        # Should extract from <code> tags first
        assert code == "real_code"

    def test_extract_code_no_tags(self):
        """Test code extraction when no tags present."""
        agent = ConversationalAgent()

        text = "Just a plain response"
        code = agent._extract_code(text)

        assert code is None

    def test_extract_answer(self):
        """Test answer extraction from response."""
        agent = ConversationalAgent()

        text = "<answer>\nThe result is 42.\n</answer>"
        answer = agent._extract_answer(text)

        assert answer == "The result is 42."

    def test_extract_answer_incomplete_tag(self):
        """Test answer extraction with incomplete tag (LLM stopped at </answer>)."""
        agent = ConversationalAgent()

        # LLM stopped before closing tag
        text = "<answer>\nThe result is 42.\n"
        answer = agent._extract_answer(text)

        assert answer == "The result is 42."

    def test_extract_answer_no_tags(self):
        """Test answer extraction when no tags present."""
        agent = ConversationalAgent()

        text = "Just a plain response"
        answer = agent._extract_answer(text)

        assert answer is None

    def test_extract_thinking(self):
        """Test thinking extraction from response."""
        agent = ConversationalAgent()

        text = "<think>\nLet me analyze this...\n</think>"
        thinking = agent._extract_thinking(text)

        assert thinking == "Let me analyze this..."

    def test_extract_thinking_no_tags(self):
        """Test thinking extraction when no tags present."""
        agent = ConversationalAgent()

        text = "Just a plain response"
        thinking = agent._extract_thinking(text)

        assert thinking is None

    def test_chat_without_start_raises(self):
        """Test that chat raises when agent not started."""
        agent = ConversationalAgent()

        with pytest.raises(RuntimeError, match="not started"):
            agent.chat("Hello")

    def test_chat_stream_without_start_raises(self):
        """Test that chat_stream raises when agent not started."""
        agent = ConversationalAgent()

        with pytest.raises(RuntimeError, match="not started"):
            list(agent.chat_stream("Hello"))

    def test_execute_code_directly_without_start_raises(self):
        """Test that execute_code_directly raises when not started."""
        agent = ConversationalAgent()

        with pytest.raises(RuntimeError, match="not started"):
            agent.execute_code_directly("print(1)")

    def test_context_manager(self, tmp_path):
        """Test using agent as context manager."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )

        with patch.object(ConversationalAgent, '_call_llm') as mock_llm:
            mock_llm.return_value = "Hello!"

            with ConversationalAgent(config=config) as agent:
                assert agent.is_running is True

            assert agent.is_running is False

    def test_get_kernel_state_not_running(self):
        """Test get_kernel_state when not running returns empty dict."""
        agent = ConversationalAgent()

        state = agent.get_kernel_state()

        assert state == {}


class TestConversationalAgentWithKernel:
    """Integration tests for ConversationalAgent with real kernel.

    These tests start a real kernel and test the full flow.
    """

    def test_start_creates_session(self, tmp_path):
        """Test that start creates a session if none provided."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        try:
            agent.start()

            assert agent.is_running is True
            assert agent.session is not None
        finally:
            agent.shutdown()

    def test_start_with_session(self, tmp_path):
        """Test starting with an existing session."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        session = Session.new(workspace_path=str(tmp_path))
        agent = ConversationalAgent(config=config, session=session)

        try:
            agent.start()

            assert agent.is_running is True
            assert agent.session == session
        finally:
            agent.shutdown()

    def test_start_idempotent(self, tmp_path):
        """Test that calling start multiple times is safe."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        try:
            agent.start()
            agent.start()  # Should not raise

            assert agent.is_running is True
        finally:
            agent.shutdown()

    def test_shutdown_idempotent(self, tmp_path):
        """Test that calling shutdown multiple times is safe."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        agent.start()
        agent.shutdown()
        agent.shutdown()  # Should not raise

        assert agent.is_running is False

    def test_execute_code_directly(self, tmp_path):
        """Test direct code execution."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        try:
            agent.start()
            result = agent.execute_code_directly("print('hello world')")

            assert result.success is True
            assert "hello world" in result.stdout
        finally:
            agent.shutdown()

    def test_execute_code_directly_with_error(self, tmp_path):
        """Test direct code execution with error."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        try:
            agent.start()
            result = agent.execute_code_directly("raise ValueError('test error')")

            assert result.success is False
            assert "ValueError" in result.error
        finally:
            agent.shutdown()

    def test_get_kernel_state(self, tmp_path):
        """Test getting kernel state."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        try:
            agent.start()
            agent.execute_code_directly("x = 42")
            agent.execute_code_directly("name = 'test'")

            state = agent.get_kernel_state()

            assert "variables" in state
            assert "x" in state["variables"]
            assert "name" in state["variables"]
        finally:
            agent.shutdown()

    def test_reset_kernel(self, tmp_path):
        """Test kernel reset."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        try:
            agent.start()
            agent.execute_code_directly("my_var = 123")
            agent.reset_kernel()

            result = agent.execute_code_directly("print(my_var)")

            assert result.success is False  # NameError
        finally:
            agent.shutdown()

    def test_state_persists_between_executions(self, tmp_path):
        """Test that state persists between executions."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        try:
            agent.start()
            agent.execute_code_directly("x = 10")
            result = agent.execute_code_directly("print(x * 2)")

            assert result.success is True
            assert "20" in result.stdout
        finally:
            agent.shutdown()


class TestConversationalAgentChat:
    """Tests for ConversationalAgent chat functionality."""

    def test_chat_simple_response(self, tmp_path):
        """Test chat with simple response (no code)."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = "Hello! How can I help you?"

            try:
                agent.start()
                response = agent.chat("Hi there")

                assert response.content == "Hello! How can I help you?"
                assert response.code is None
                assert response.execution_result is None
            finally:
                agent.shutdown()

    def test_chat_with_code(self, tmp_path):
        """Test chat with code execution."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = "Here's the result:\n<code>\nprint(2 + 2)\n</code>"

            try:
                agent.start()
                response = agent.chat("What's 2 + 2?")

                assert response.code == "print(2 + 2)"
                assert response.execution_result is not None
                assert response.execution_result.success is True
                assert "4" in response.execution_result.stdout
            finally:
                agent.shutdown()

    def test_chat_with_answer(self, tmp_path):
        """Test chat with answer tag."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = "<answer>\nThe answer is 42.\n</answer>"

            try:
                agent.start()
                response = agent.chat("What's the answer?")

                assert response.has_answer is True
                assert response.answer == "The answer is 42."
            finally:
                agent.shutdown()

    def test_chat_adds_to_history(self, tmp_path):
        """Test that chat adds messages to history."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        session = Session.new(workspace_path=str(tmp_path))
        agent = ConversationalAgent(config=config, session=session)

        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = "Response 1"

            try:
                agent.start()

                initial_count = len(session.history)
                agent.chat("Message 1")

                # Should have added user message and assistant response
                assert len(session.history) >= initial_count + 2
            finally:
                agent.shutdown()

    def test_chat_stream(self, tmp_path):
        """Test chat_stream yields responses."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = "Streamed response"

            try:
                agent.start()
                responses = list(agent.chat_stream("Hello"))

                assert len(responses) >= 1
                assert responses[0].content == "Streamed response"
            finally:
                agent.shutdown()


class TestConversationalAgentWithSessionManager:
    """Tests for ConversationalAgent with SessionManager."""

    def test_session_saved_on_shutdown(self, tmp_path):
        """Test that session is saved when agent shuts down."""
        manager = SessionManager(tmp_path)
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(
            config=config,
            session_manager=manager,
        )

        try:
            agent.start()
            session_id = agent.session.id
            agent.execute_code_directly("x = 42")
        finally:
            agent.shutdown()

        # Session should be saved
        saved_session = manager.load_session(session_id)
        assert saved_session is not None

        manager.close()

    def test_kernel_snapshot_updated_after_execution(self, tmp_path):
        """Test that kernel snapshot is updated after code execution."""
        manager = SessionManager(tmp_path)
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(
            config=config,
            session_manager=manager,
        )

        try:
            agent.start()
            agent.execute_code_directly("my_variable = 'test_value'")

            # Kernel snapshot should be updated
            assert agent.session.kernel_snapshot is not None
            assert "my_variable" in agent.session.kernel_snapshot.variables
        finally:
            agent.shutdown()

        manager.close()


class TestSystemPrompt:
    """Tests for system prompt generation."""

    def test_prompt_builder_creates_valid_prompt(self):
        """Test that PromptBuilder creates a prompt with expected sections."""
        prompt = PromptBuilder.build_conversational_prompt(
            kernel_context="test kernel",
            tools=None,
            skills_context=None,
        )
        # Check key sections are present
        assert "test kernel" in prompt
        assert "execute code" in prompt.lower()

    def test_build_system_prompt(self, tmp_path):
        """Test building system prompt with context."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        try:
            agent.start()
            prompt = agent._build_system_prompt()

            # Should contain the kernel context section
            assert "Data Science assistant" in prompt
        finally:
            agent.shutdown()

    def test_get_kernel_context_not_running(self):
        """Test kernel context when not running."""
        agent = ConversationalAgent()

        context = agent._get_kernel_context()

        assert "no kernel" in context.lower() or "not available" in context.lower()


class TestBuildMessages:
    """Tests for message building."""

    def test_build_messages_empty_history(self, tmp_path):
        """Test building messages with empty history."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        session = Session.new(workspace_path=str(tmp_path))
        agent = ConversationalAgent(config=config, session=session)

        try:
            agent.start()
            messages = agent._build_messages()

            # Should have at least the system message
            assert len(messages) >= 1
            assert messages[0]["role"] == "system"
        finally:
            agent.shutdown()

    def test_build_messages_with_history(self, tmp_path):
        """Test building messages with conversation history."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        session = Session.new(workspace_path=str(tmp_path))
        session.history.add_user("Hello")
        session.history.add_assistant("Hi there!")

        agent = ConversationalAgent(config=config, session=session)

        try:
            agent.start()
            messages = agent._build_messages()

            # Should have system + conversation messages
            assert len(messages) >= 3

            # Find user and assistant messages
            roles = [m["role"] for m in messages]
            assert "user" in roles
            assert "assistant" in roles
        finally:
            agent.shutdown()


class TestPlanExtraction:
    """Tests for plan extraction and tracking."""

    def test_extract_plan(self):
        """Test plan extraction from response."""
        agent = ConversationalAgent()

        text = """Here's my plan:
<plan>
1. [ ] Load data
2. [ ] Process data
3. [ ] Build model
</plan>
"""
        plan = agent._extract_plan(text)

        assert plan is not None
        assert len(plan.steps) == 3
        assert plan.steps[0].description == "Load data"
        assert plan.steps[0].completed is False

    def test_extract_plan_with_completed_steps(self):
        """Test plan extraction with completed steps."""
        agent = ConversationalAgent()

        text = """<plan>
1. [x] Load data
2. [x] Process data
3. [ ] Build model
</plan>
"""
        plan = agent._extract_plan(text)

        assert plan is not None
        assert plan.steps[0].completed is True
        assert plan.steps[1].completed is True
        assert plan.steps[2].completed is False

    def test_extract_plan_no_plan(self):
        """Test plan extraction when no plan present."""
        agent = ConversationalAgent()

        text = "Just a simple response without a plan."
        plan = agent._extract_plan(text)

        assert plan is None

    def test_is_plan_complete_no_plan(self):
        """Test is_plan_complete when no plan."""
        agent = ConversationalAgent()

        assert agent._is_plan_complete() is True

    def test_is_plan_complete_incomplete(self):
        """Test is_plan_complete with incomplete plan."""
        agent = ConversationalAgent()

        text = """<plan>
1. [x] Done
2. [ ] Not done
</plan>
"""
        agent._current_plan = agent._extract_plan(text)

        assert agent._is_plan_complete() is False

    def test_is_plan_complete_complete(self):
        """Test is_plan_complete with complete plan."""
        agent = ConversationalAgent()

        text = """<plan>
1. [x] Done
2. [x] Also done
</plan>
"""
        agent._current_plan = agent._extract_plan(text)

        assert agent._is_plan_complete() is True


class TestChatResponseIsComplete:
    """Tests for ChatResponse.is_complete logic."""

    def test_simple_response_is_complete(self, tmp_path):
        """Test that simple response (no plan, no code) is complete."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = "Just a simple answer."

            try:
                agent.start()
                response = agent.chat("What is Python?")

                assert response.is_complete is True
                assert response.plan is None
            finally:
                agent.shutdown()

    def test_response_with_incomplete_plan_not_complete(self, tmp_path):
        """Test that response with incomplete plan is not complete."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        response_text = """<plan>
1. [ ] Step one
2. [ ] Step two
</plan>

<code>
print("hello")
</code>
"""

        with patch.object(agent, '_call_llm') as mock_llm:
            # First call returns plan, second marks steps done, third provides answer
            mock_llm.side_effect = [
                response_text,
                """<plan>
1. [x] Step one
2. [ ] Step two
</plan>

<code>
print("step 2")
</code>
""",
                """<plan>
1. [x] Step one
2. [x] Step two
</plan>

<answer>
All done!
</answer>
"""
            ]

            try:
                agent.start()
                response = agent.chat("Do something")

                # Should complete after executing all steps
                assert response.is_complete is True
                assert response.has_answer is True
            finally:
                agent.shutdown()


class TestAutonomousExecution:
    """Tests for autonomous execution mode."""

    def test_autonomous_continues_until_complete(self, tmp_path):
        """Test that autonomous mode continues until plan is complete."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
            max_rounds=10,
        )
        agent = ConversationalAgent(config=config)

        responses = [
            """<plan>
1. [ ] Step 1
2. [ ] Step 2
</plan>

<code>
x = 1
</code>
""",
            """<plan>
1. [x] Step 1
2. [ ] Step 2
</plan>

<code>
y = 2
</code>
""",
            """<plan>
1. [x] Step 1
2. [x] Step 2
</plan>

<answer>
Done! x=1, y=2
</answer>
"""
        ]

        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.side_effect = responses

            try:
                agent.start()
                response = agent.chat("Do two steps")

                assert response.is_complete is True
                assert response.has_answer is True
                assert "Done" in response.answer
                # Should have called LLM 3 times
                assert mock_llm.call_count == 3
            finally:
                agent.shutdown()

    def test_autonomous_respects_max_rounds(self, tmp_path):
        """Test that autonomous mode stops at max_rounds."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
            max_rounds=3,
        )
        agent = ConversationalAgent(config=config)

        # LLM never completes the plan
        incomplete_response = """<plan>
1. [ ] Never done
</plan>

<code>
print("still going")
</code>
"""

        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = incomplete_response

            try:
                agent.start()
                response = agent.chat("Infinite task")

                # Should stop at max_rounds
                assert "Max rounds" in response.content
            finally:
                agent.shutdown()


class TestChatStream:
    """Tests for chat_stream functionality."""

    def test_chat_stream_yields_responses(self, tmp_path):
        """Test that chat_stream yields responses."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = "Simple response"

            try:
                agent.start()
                responses = list(agent.chat_stream("Hello"))

                assert len(responses) >= 1
                assert responses[0].content == "Simple response"
            finally:
                agent.shutdown()

    def test_chat_stream_yields_multiple_for_plan(self, tmp_path):
        """Test that chat_stream yields multiple responses for plan execution."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
            max_rounds=10,
        )
        agent = ConversationalAgent(config=config)

        responses = [
            """<plan>
1. [ ] Step 1
</plan>

<code>
x = 1
</code>
""",
            """<plan>
1. [x] Step 1
</plan>

<answer>
Done!
</answer>
"""
        ]

        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.side_effect = responses

            try:
                agent.start()
                yielded = list(agent.chat_stream("Do one step"))

                # Should yield 2 responses (one per round)
                assert len(yielded) == 2
            finally:
                agent.shutdown()


class TestNotebookExport:
    """Tests for notebook export functionality."""

    def test_chat_stream_initializes_notebook_builder(self, tmp_path):
        """Test that chat_stream initializes notebook builder on first message."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = "Simple response"

            try:
                agent.start()
                assert agent._notebook_builder is None

                # First message should initialize notebook builder
                list(agent.chat_stream("Hello"))

                assert agent._notebook_builder is not None
                assert agent._current_task == "Hello"
            finally:
                agent.shutdown()

    def test_code_execution_tracked_in_notebook(self, tmp_path):
        """Test that code executions are tracked in notebook builder."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = "<code>\nx = 42\nprint(x)\n</code>"

            try:
                agent.start()
                list(agent.chat_stream("Create a variable"))

                # Check notebook builder has tracked the execution
                assert agent._notebook_builder is not None
                assert len(agent._notebook_builder.tracker.records) == 1
                assert agent._notebook_builder.tracker.records[0].success
            finally:
                agent.shutdown()

    def test_export_notebook_returns_none_without_executions(self, tmp_path):
        """Test export_notebook returns None if no code was executed."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = "Just a text response, no code"

            try:
                agent.start()
                list(agent.chat_stream("Hello"))

                # No code executed, so export should return None
                result = agent.export_notebook()
                assert result is None
            finally:
                agent.shutdown()

    def test_export_notebook_creates_file(self, tmp_path):
        """Test export_notebook creates a notebook file."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = "<code>\nprint('hello')\n</code>"

            try:
                agent.start()
                list(agent.chat_stream("Say hello"))

                # Export should create a notebook file
                notebook_path = agent.export_notebook("test_notebook.ipynb")

                assert notebook_path is not None
                assert notebook_path.exists()
                assert notebook_path.name == "test_notebook.ipynb"
            finally:
                agent.shutdown()

    def test_shutdown_saves_notebook(self, tmp_path):
        """Test that shutdown saves the notebook."""
        config = ConversationalAgentConfig(
            workspace=tmp_path,
            code_timeout=30,
        )
        agent = ConversationalAgent(config=config)

        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = "<code>\nx = 1\n</code>"

            agent.start()
            list(agent.chat_stream("Create x"))

            # Shutdown should save notebook
            notebook_path = agent.shutdown(save_notebook=True)

            assert notebook_path is not None
            assert notebook_path.exists()
