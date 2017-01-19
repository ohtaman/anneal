# -*- coding:utf-8 -*-

from unittest import mock

from anneal.annealers import Annealer


class TestAnnealer(object):
    class MockAnnealer(Annealer):
        def is_frozen(self):
            return True

        def update(self, state_is_updated):
            return True

    def test_can_create_annealer(self):
        TestAnnealer.MockAnnealer(mock.Mock())

    def test_anneal(self):
        testee = TestAnnealer.MockAnnealer(mock.Mock())
        testee.optimize = mock.MagicMock()
        testee.anneal(max_iter=10, iter_callback=None)
        testee.optimize.assert_called_with(max_iter=10, iter_callback=None)
