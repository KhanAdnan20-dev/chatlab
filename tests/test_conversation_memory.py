from backend.app.services.conversation_memory import ConversationMemory


def test_conversation_memory_flow():
    mem = ConversationMemory(max_turns=3)
    session = mem.get_or_create(None)
    sid = session.session_id

    mem.add_user_message(sid, "Hello")
    mem.add_assistant_message(sid, "Hi there")

    hist = mem.get_history(sid)
    assert len(hist) == 2
    assert hist[0]["role"] == "user"
    assert hist[1]["role"] == "assistant"

    # Test rolling window pruning
    for i in range(10):
        mem.add_user_message(sid, f"msg {i}")
        mem.add_assistant_message(sid, f"reply {i}")

    pruned = mem.get_history(sid)
    assert len(pruned) <= 6  # max_turns=3 -> 6 messages max


def test_failed_stream_retry_deduplication_with_request_id():
    """Verify that a stream failure followed by a synchronous retry with request_id records only one user turn."""
    mem = ConversationMemory()
    session = mem.get_or_create("test-retry-session-1")
    sid = session.session_id

    # 1. Stream starts: user message registered with request_id="turn-req-abc"
    res1 = mem.add_user_message(sid, "Explain backpropagation", request_id="turn-req-abc")
    assert res1 is True

    # 2. Stream drops connection before assistant reply is generated/saved
    # Client catches error and falls back to synchronous /chat request with the same turn request_id="turn-req-abc"
    res2 = mem.add_user_message(sid, "Explain backpropagation", request_id="turn-req-abc")
    assert res2 is False  # Deduplicated!

    # 3. Synchronous request completes and writes assistant reply
    mem.add_assistant_message(sid, "Backpropagation calculates loss gradients via the chain rule.")

    # 4. Verify history: exactly ONE user turn and ONE assistant reply
    history = mem.get_history(sid)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Explain backpropagation"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Backpropagation calculates loss gradients via the chain rule."


def test_failed_stream_retry_deduplication_without_request_id():
    """Verify that an interrupted user turn with identical content is deduplicated even without explicit request_id."""
    mem = ConversationMemory()
    session = mem.get_or_create("test-retry-session-2")
    sid = session.session_id

    # Stream registers user message
    res1 = mem.add_user_message(sid, "What is dropout?")
    assert res1 is True

    # Stream fails before assistant answer; retry occurs with same prompt
    res2 = mem.add_user_message(sid, "What is dropout?")
    assert res2 is False  # Deduplicated!

    mem.add_assistant_message(sid, "Dropout randomly deactivates neurons during training.")

    history = mem.get_history(sid)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_legitimate_repeated_identical_messages_preserved():
    """Verify that identical messages sent across different turns (separated by assistant replies) are NOT deduplicated."""
    mem = ConversationMemory()
    session = mem.get_or_create("test-repeated-session")
    sid = session.session_id

    # Turn 1
    mem.add_user_message(sid, "Give me an example", request_id="turn-1")
    mem.add_assistant_message(sid, "Here is example 1: Linear Regression.")

    # Turn 2: User repeats the EXACT same request for another example
    mem.add_user_message(sid, "Give me an example", request_id="turn-2")
    mem.add_assistant_message(sid, "Here is example 2: Logistic Regression.")

    # Turn 3: User repeats again
    mem.add_user_message(sid, "Give me an example", request_id="turn-3")
    mem.add_assistant_message(sid, "Here is example 3: Neural Networks.")

    history = mem.get_history(sid)
    assert len(history) == 6
    assert [h["role"] for h in history] == ["user", "assistant", "user", "assistant", "user", "assistant"]
    assert all(h["content"] == "Give me an example" for h in history if h["role"] == "user")

