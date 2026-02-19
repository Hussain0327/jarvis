"""Tests for database session management."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import src.database.session as session_mod
from src.database.session import get_engine, get_session_factory, reset_engine


class TestSessionSingleton:
    def setup_method(self):
        reset_engine()

    def teardown_method(self):
        reset_engine()

    @patch("src.database.session.create_engine")
    def test_get_engine_creates_once(self, mock_create):
        """get_engine() should create only one engine across multiple calls."""
        mock_create.return_value = MagicMock()
        e1 = get_engine()
        e2 = get_engine()
        assert e1 is e2
        mock_create.assert_called_once()

    @patch("src.database.session.create_engine")
    def test_get_session_factory_creates_once(self, mock_create):
        """get_session_factory() should create only one factory."""
        mock_create.return_value = MagicMock()
        f1 = get_session_factory()
        f2 = get_session_factory()
        assert f1 is f2

    @patch("src.database.session.create_engine")
    def test_reset_engine_clears_singletons(self, mock_create):
        """reset_engine() should allow a fresh engine to be created."""
        mock_engine = MagicMock()
        mock_create.return_value = mock_engine

        get_engine()
        reset_engine()
        mock_engine.dispose.assert_called_once()

        # Next call should create a new engine
        get_engine()
        assert mock_create.call_count == 2

    @patch("src.database.session.create_engine")
    def test_thread_safe_engine_creation(self, mock_create):
        """Concurrent first calls to get_engine() should only create one engine."""
        mock_create.return_value = MagicMock()

        engines: list = []
        barrier = threading.Barrier(5)

        def get():
            barrier.wait()
            engines.append(get_engine())

        threads = [threading.Thread(target=get) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # All threads should get the same engine
        assert len(engines) == 5
        assert all(e is engines[0] for e in engines)
        mock_create.assert_called_once()

    @patch("src.database.session.create_engine")
    def test_reset_clears_session_factory_too(self, mock_create):
        """reset_engine() should also clear the session factory."""
        mock_create.return_value = MagicMock()
        get_session_factory()
        reset_engine()
        # After reset, _SessionLocal should be None
        assert session_mod._SessionLocal is None
