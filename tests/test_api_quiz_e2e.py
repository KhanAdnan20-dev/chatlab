"""End-to-End API Integration tests for Quiz endpoints."""
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_e2e_quiz_complete_flow():
    # 1. Start Quiz
    start_resp = client.post("/quiz/start", json={
        "topic": "all",
        "difficulty": "medium",
        "num_questions": 3,
    })
    assert start_resp.status_code == 201
    start_data = start_resp.json()
    session_id = start_data["session_id"]
    assert session_id is not None
    assert start_data["total_questions"] == 3
    assert start_data["first_question"] is not None
    assert len(start_data["first_question"]["options"]) == 4

    # 2. Submit Answer to Question 1
    ans_resp = client.post(f"/quiz/{session_id}/answer", json={"answer_index": 0})
    assert ans_resp.status_code == 200
    ans_data = ans_resp.json()
    assert "is_correct" in ans_data
    assert "explanation" in ans_data
    assert ans_data["current_question_index"] == 0

    # 3. Advance to Question 2
    next_resp = client.post(f"/quiz/{session_id}/next")
    assert next_resp.status_code == 200

    # 4. Check intermediate score
    score_resp = client.get(f"/quiz/{session_id}/score")
    assert score_resp.status_code == 200
    score_data = score_resp.json()
    assert score_data["total_questions"] == 3
    assert score_data["completed"] is False

    # 5. Finish Quiz early
    finish_resp = client.post(f"/quiz/{session_id}/finish")
    assert finish_resp.status_code == 200
    final_score = finish_resp.json()
    assert final_score["completed"] is True
    assert 0.0 <= final_score["score_percent"] <= 100.0


def test_quiz_session_isolation():
    # Start two independent sessions
    s1 = client.post("/quiz/start", json={"num_questions": 2}).json()["session_id"]
    s2 = client.post("/quiz/start", json={"num_questions": 2}).json()["session_id"]
    assert s1 != s2

    # Answer s1 only
    client.post(f"/quiz/{s1}/answer", json={"answer_index": 1})

    score1 = client.get(f"/quiz/{s1}/score").json()
    score2 = client.get(f"/quiz/{s2}/score").json()

    # s2 score should still be 0 correct answers
    assert score2["correct_count"] == 0


def test_quiz_api_answer_idempotency_and_conflict():
    # 1. Start session
    resp = client.post("/quiz/start", json={"num_questions": 2})
    s = resp.json()["session_id"]

    # 2. Submit answer
    ans_resp1 = client.post(f"/quiz/{s}/answer", json={"answer_index": 0})
    assert ans_resp1.status_code == 200
    data1 = ans_resp1.json()

    # 3. Repeat exact same submission (idempotent HTTP 200)
    ans_resp2 = client.post(f"/quiz/{s}/answer", json={"answer_index": 0})
    assert ans_resp2.status_code == 200
    assert ans_resp2.json() == data1

    # 4. Score is not multiplied
    score_resp = client.get(f"/quiz/{s}/score")
    score_data = score_resp.json()
    expected_correct = 1 if data1["is_correct"] else 0
    assert score_data["correct_count"] == expected_correct

    # 5. Conflicting answer submission triggers HTTP 409 Conflict
    conflict_resp = client.post(f"/quiz/{s}/answer", json={"answer_index": 1})
    assert conflict_resp.status_code == 409
    assert "already been answered" in conflict_resp.json()["detail"].lower()

